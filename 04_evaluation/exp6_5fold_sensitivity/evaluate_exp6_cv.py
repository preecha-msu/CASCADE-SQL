import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cnn_bilstm"))
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "ast_gnn"))

from cascade_onnx_engine import batch_encode_stage1
from feature_extractor_stage2 import batch_extract_stage2
from train_stage1_onnx import Stage1CNNBiLSTM
from train_stage2_onnx import Stage2ASTContextualVerifier

SEED = 42
TRAIN_CSV = ROOT / "01_datasets" / "processed" / "train.csv"
VAL_CSV = ROOT / "01_datasets" / "processed" / "val.csv"
TEST_CSV = ROOT / "01_datasets" / "processed" / "test.csv"
ROUTER_CFG = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"

LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "exp6_5fold_cv_sensitivity_audit.json"
SURF_NPZ = ROOT / "05_results_and_paper" / "raw_logs" / "exp6_threshold_surface.npz"
TAB12_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab12_exp6_5fold_cv.tex"


def log_step(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_utf8_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def summary_across_folds(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=np.float64)
    k = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=k - 1))
    half_w = t_crit * (sd / np.sqrt(k))
    return {
        "mean": round(mean, 6),
        "std": round(sd, 6),
        "ci95_low": round(mean - half_w, 6),
        "ci95_high": round(mean + half_w, 6),
    }


def main() -> None:
    t0 = time.perf_counter_ns()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads(ROUTER_CFG.read_text(encoding="utf-8"))
    tl_opt = float(cfg["tau_low"])
    th_opt = float(cfg["tau_high"])
    t1_ms = float(cfg["measured_mean_stage1_ms"])
    t2_ms = float(cfg["measured_mean_stage2_ms"])

    log_step("[1/3] Loading all 50,460 deduplicated SQL records and pre-extracting Stage-1 & Stage-2 tensors once...")
    df_all = pd.concat([
        pd.read_csv(TRAIN_CSV, low_memory=False),
        pd.read_csv(VAL_CSV, low_memory=False),
        pd.read_csv(TEST_CSV, low_memory=False),
    ], ignore_index=True)

    queries = df_all["query"].fillna("").astype(str).tolist()
    dialects = df_all["dialect"].fillna("mysql").astype(str).tolist()
    labels = df_all["label"].astype(np.int64).values
    groups = df_all["template_hash"].astype(str).values

    X1_all = batch_encode_stage1(queries)
    Xt_all, Xs_all = batch_extract_stage2(queries, dialects)

    gkf = GroupKFold(n_splits=5)
    fold_reports = []
    oof_p1 = np.zeros(len(df_all), dtype=np.float32)
    oof_p2 = np.zeros(len(df_all), dtype=np.float32)

    log_step("[2/3] Executing 5-Fold Group-Isolated Cross-Validation on GPU (0% template leakage per fold)...")
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X1_all, labels, groups=groups), start=1):
        f_t0 = time.perf_counter_ns()
        set_seed(SEED + fold_idx)

        # Mathematical Zero-Leakage Assertion
        g_tr = set(groups[tr_idx])
        g_te = set(groups[te_idx])
        assert len(g_tr & g_te) == 0, f"Template leakage detected in Fold {fold_idx}!"

        y_tr_f = labels[tr_idx].astype(np.float32)
        y_te_i = labels[te_idx]
        pos_w = torch.tensor([float((y_tr_f == 0).sum()) / max(1.0, float((y_tr_f == 1).sum()))], device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        # Train Stage-1
        m1 = Stage1CNNBiLSTM().to(device)
        opt1 = torch.optim.AdamW(m1.parameters(), lr=1.5e-3, weight_decay=1e-4)
        ld1 = DataLoader(TensorDataset(torch.from_numpy(X1_all[tr_idx]), torch.from_numpy(y_tr_f)), batch_size=256, shuffle=True)
        m1.train()
        for _ in range(4):
            for bx, by in ld1:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                opt1.zero_grad(set_to_none=True)
                loss = crit(m1(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(m1.parameters(), 1.0)
                opt1.step()

        # Train Stage-2
        m2 = Stage2ASTContextualVerifier().to(device)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1.2e-3, weight_decay=1e-4)
        ld2 = DataLoader(
            TensorDataset(torch.from_numpy(Xt_all[tr_idx]), torch.from_numpy(Xs_all[tr_idx]), torch.from_numpy(y_tr_f)),
            batch_size=256,
            shuffle=True,
        )
        m2.train()
        for _ in range(5):
            for btok, bstr, by in ld2:
                btok, bstr, by = btok.to(device, non_blocking=True), bstr.to(device, non_blocking=True), by.to(device, non_blocking=True)
                opt2.zero_grad(set_to_none=True)
                loss = crit(m2(btok, bstr), by)
                loss.backward()
                nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
                opt2.step()

        # Evaluate Out-Of-Fold predictions
        m1.eval()
        m2.eval()
        X1_te = X1_all[te_idx]
        Xt_te = Xt_all[te_idx]
        Xs_te = Xs_all[te_idx]
        with torch.no_grad():
            lg1 = np.concatenate([m1(torch.from_numpy(X1_te[i:i+512]).to(device)).cpu().numpy() for i in range(0, len(X1_te), 512)])
            lg2 = np.concatenate([
                m2(torch.from_numpy(Xt_te[i:i+512]).to(device), torch.from_numpy(Xs_te[i:i+512]).to(device)).cpu().numpy()
                for i in range(0, len(Xt_te), 512)
            ])

        p1 = 1.0 / (1.0 + np.exp(-np.clip(lg1, -50.0, 50.0)))
        p2 = 1.0 / (1.0 + np.exp(-np.clip(lg2, -50.0, 50.0)))
        oof_p1[te_idx] = p1
        oof_p2[te_idx] = p2

        esc = (p1 > tl_opt) & (p1 < th_opt)
        preds = np.zeros(len(y_te_i), dtype=np.int64)
        preds[p1 >= th_opt] = 1
        preds[esc] = (p2[esc] >= 0.50).astype(np.int64)
        scores = np.where(esc, p2, p1)

        tp = int(((preds == 1) & (y_te_i == 1)).sum())
        fp = int(((preds == 1) & (y_te_i == 0)).sum())
        fn = int(((preds == 0) & (y_te_i == 1)).sum())
        tn = int(((preds == 0) & (y_te_i == 0)).sum())

        acc = float((preds == y_te_i).mean())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
        fpr = fp / max(1, fp + tn)
        roc = float(roc_auc_score(y_te_i, scores))
        prauc = float(average_precision_score(y_te_i, scores))
        rho_pct = float(esc.mean() * 100.0)
        f_sec = (time.perf_counter_ns() - f_t0) / 1e9

        log_step(
            f"   -> [Fold {fold_idx}/5] Rows={len(y_te_i):,} | Templates={len(g_te):,} | "
            f"F1={f1:.4f} | FPR={fpr:.4f} | ROC-AUC={roc:.4f} | Esc={rho_pct:.2f}% ({f_sec:.1f}s)"
        )

        fold_reports.append({
            "fold": fold_idx,
            "test_rows": int(len(y_te_i)),
            "unique_templates": int(len(g_te)),
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
            "fpr": round(fpr, 6),
            "roc_auc": round(roc, 6),
            "pr_auc": round(prauc, 6),
            "escalation_rate_pct": round(rho_pct, 4),
        })

    # [3/3] Compute 10x10 2D Threshold Sensitivity Surface on Full Out-of-Fold Probabilities (N=50,460)
    log_step("[3/3] Computing 10x10 2D Threshold Sensitivity Surface across 50,460 out-of-fold predictions...")
    tl_grid = np.linspace(0.01, 0.45, 10)
    th_grid = np.linspace(0.55, 0.99, 10)
    f1_surf = np.zeros((10, 10), dtype=np.float64)
    fpr_surf = np.zeros((10, 10), dtype=np.float64)
    rho_surf = np.zeros((10, 10), dtype=np.float64)
    lat_surf = np.zeros((10, 10), dtype=np.float64)

    for i, tl in enumerate(tl_grid):
        for j, th in enumerate(th_grid):
            esc = (oof_p1 > tl) & (oof_p1 < th)
            pr = np.zeros(len(labels), dtype=np.int64)
            pr[oof_p1 >= th] = 1
            pr[esc] = (oof_p2[esc] >= 0.50).astype(np.int64)
            tp = int(((pr == 1) & (labels == 1)).sum())
            fp = int(((pr == 1) & (labels == 0)).sum())
            fn = int(((pr == 0) & (labels == 1)).sum())
            tn = int(((pr == 0) & (labels == 0)).sum())
            f1_surf[i, j] = (2.0 * tp) / max(1, 2 * tp + fp + fn)
            fpr_surf[i, j] = fp / max(1, fp + tn)
            rho_val = float(esc.mean())
            rho_surf[i, j] = rho_val * 100.0
            lat_surf[i, j] = t1_ms + rho_val * t2_ms

    np.savez_compressed(
        SURF_NPZ,
        tau_low_grid=tl_grid,
        tau_high_grid=th_grid,
        f1_surface=f1_surf,
        fpr_surface=fpr_surf,
        rho_pct_surface=rho_surf,
        expected_latency_ms_surface=lat_surf,
    )

    agg = {
        metric: summary_across_folds([f[metric] for f in fold_reports])
        for metric in ["accuracy", "precision", "recall", "f1_score", "fpr", "roc_auc", "pr_auc", "escalation_rate_pct"]
    }

    fold_rows_tex = []
    for f in fold_reports:
        fold_rows_tex.append(
            f"Fold {f['fold']} & {f['test_rows']:,} & {f['unique_templates']:,} & "
            f"{f['precision']:.4f} & {f['recall']:.4f} & {f['f1_score']:.4f} & "
            f"{f['fpr']:.4f} & {f['roc_auc']:.4f} & {f['escalation_rate_pct']:.2f}\\% \\\\"
        )
    joined_folds = "\n".join(fold_rows_tex)

    tab12_tex = rf"""% Auto-generated from 100% Real 5-Fold Group-Isolated Cross-Validation Audit (Exp-6, N=50,460)
\begin{{table*}}[htbp]
\centering
\caption{{5-Fold Group-Isolated Cross-Validation across all {len(df_all):,} deduplicated SQL queries ({df_all['template_hash'].nunique():,} canonical AST templates, 0\% cross-fold template overlap) with sample standard deviation ($\pm s$) and 95\% Student-$t$ confidence intervals.}}
\label{{tab:exp6_5fold_cv}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
\textbf{{Partition}} & \textbf{{Test Queries}} & \textbf{{Templates}} & \textbf{{Precision}} & \textbf{{Recall}} & \textbf{{F1-Score}} & \textbf{{FPR}} & \textbf{{ROC-AUC}} & \textbf{{Escalated ($\rho$)}} \\
\midrule
{joined_folds}
\midrule
\textbf{{Mean $\pm$ SD}} & {len(df_all)//5:,} & — & \textbf{{{agg['precision']['mean']:.4f} $\pm$ {agg['precision']['std']:.4f}}} & \textbf{{{agg['recall']['mean']:.4f} $\pm$ {agg['recall']['std']:.4f}}} & \textbf{{{agg['f1_score']['mean']:.4f} $\pm$ {agg['f1_score']['std']:.4f}}} & \textbf{{{agg['fpr']['mean']:.4f} $\pm$ {agg['fpr']['std']:.4f}}} & \textbf{{{agg['roc_auc']['mean']:.4f} $\pm$ {agg['roc_auc']['std']:.4f}}} & \textbf{{{agg['escalation_rate_pct']['mean']:.2f}\% $\pm$ {agg['escalation_rate_pct']['std']:.2f}\%}} \\
\textbf{{95\% Student-$t$ CI}} & — & — & [{agg['precision']['ci95_low']:.4f}, {agg['precision']['ci95_high']:.4f}] & [{agg['recall']['ci95_low']:.4f}, {agg['recall']['ci95_high']:.4f}] & [{agg['f1_score']['ci95_low']:.4f}, {agg['f1_score']['ci95_high']:.4f}] & [{agg['fpr']['ci95_low']:.4f}, {agg['fpr']['ci95_high']:.4f}] & [{agg['roc_auc']['ci95_low']:.4f}, {agg['roc_auc']['ci95_high']:.4f}] & [{agg['escalation_rate_pct']['ci95_low']:.2f}\%, {agg['escalation_rate_pct']['ci95_high']:.2f}\%] \\
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    write_utf8_lf(TAB12_TEX, tab12_tex)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_sec": round((time.perf_counter_ns() - t0) / 1e9, 3),
        "total_corpus_rows": int(len(df_all)),
        "total_unique_templates": int(df_all["template_hash"].nunique()),
        "zero_leakage_across_all_5_folds_verified": True,
        "folds": fold_reports,
        "aggregate_summary": agg,
        "sensitivity_surface_summary": {
            "f1_min_across_100_grid_points": round(float(np.min(f1_surf)), 6),
            "f1_max_across_100_grid_points": round(float(np.max(f1_surf)), 6),
            "fpr_min_across_100_grid_points": round(float(np.min(fpr_surf)), 6),
            "fpr_max_across_100_grid_points": round(float(np.max(fpr_surf)), 6),
            "rho_pct_range": [round(float(np.min(rho_surf)), 3), round(float(np.max(rho_surf)), 3)],
        },
        "sha256_verification": {
            "threshold_surface_npz": sha256_of(SURF_NPZ),
            "tab12_exp6_5fold_cv_tex": sha256_of(TAB12_TEX),
        },
    }

    write_utf8_lf(LOG_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()