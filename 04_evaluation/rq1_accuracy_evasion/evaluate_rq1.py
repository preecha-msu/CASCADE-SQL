import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "ast_gnn"))
sys.path.insert(0, str(ROOT / "03_cloud_testbed" / "baseline_waf_modsec"))

from cascade_onnx_engine import CascadeSQLOnnxEngine
from feature_extractor_stage2 import check_dos_guard
from owasp_crs_baseline import OwaspCrsV4Baseline

TEST_CSV = ROOT / "01_datasets" / "processed" / "test.csv"
IMB_CSV = ROOT / "01_datasets" / "imbalanced_splits" / "test_imbalanced_100_1.csv"
ADV_CSV = ROOT / "01_datasets" / "adversarial_wafamole" / "test_adversarial_obfuscated.csv"
ROUTER_CFG = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"

LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "rq1_evaluation_audit.json"
PREDS_NPZ = ROOT / "05_results_and_paper" / "raw_logs" / "rq1_real_test_predictions.npz"
TAB2_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab2_rq1_performance.tex"


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


def compute_metrics_safe(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n = len(y_true)
    acc = float((y_pred == y_true).mean()) if n > 0 else 0.0
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    fnr = fn / max(1, tp + fn)

    has_both_classes = (tp + fn > 0) and (fp + tn > 0)
    fpr = round(fp / (fp + tn), 6) if (fp + tn > 0) else None
    roc_auc = round(float(roc_auc_score(y_true, scores)), 6) if (has_both_classes and scores is not None) else None
    pr_auc = round(float(average_precision_score(y_true, scores)), 6) if (has_both_classes and scores is not None) else None

    return {
        "samples": int(n),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "false_positive_rate": fpr,
        "false_negative_rate": round(fnr, 6),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def evaluate_cascade_on_df(engine: CascadeSQLOnnxEngine, df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    queries = df["query"].fillna("").astype(str).tolist()
    dialects = df["dialect"].fillna("mysql").astype(str).tolist()
    y_true = df["label"].astype(np.int64).values
    n = len(y_true)

    dos_mask = np.array([check_dos_guard(q) for q in queries], dtype=bool)
    p1 = engine.predict_stage1_probs_batch(queries)

    esc_mask = (~dos_mask) & (p1 > engine.tau_low) & (p1 < engine.tau_high)
    fast_pos = (~dos_mask) & (p1 >= engine.tau_high)

    p2_full = np.zeros(n, dtype=np.float32)
    if esc_mask.any():
        esc_idx = np.where(esc_mask)[0]
        esc_q = [queries[i] for i in esc_idx]
        esc_d = [dialects[i] for i in esc_idx]
        p2_esc = engine.predict_stage2_probs_batch(esc_q, esc_d)
        p2_full[esc_idx] = p2_esc

    preds = np.zeros(n, dtype=np.int64)
    preds[dos_mask] = 1
    preds[fast_pos] = 1
    preds[esc_mask] = (p2_full[esc_mask] >= engine.tau_stage2).astype(np.int64)

    # Unified continuous score for ROC-AUC / PR-AUC
    scores = p1.copy()
    scores[dos_mask] = 1.0
    scores[esc_mask] = p2_full[esc_mask]

    res = compute_metrics_safe(y_true, preds, scores)
    res["escalation_rate_pct"] = round(float(esc_mask.mean() * 100.0), 4)
    res["escalated_queries"] = int(esc_mask.sum())
    res["dos_blocked_queries"] = int(dos_mask.sum())
    return res, preds, p1, scores


def evaluate_crs_on_df(crs: OwaspCrsV4Baseline, df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    queries = df["query"].fillna("").astype(str).tolist()
    y_true = df["label"].astype(np.int64).values
    preds = np.zeros(len(y_true), dtype=np.int64)
    rule_scores = np.zeros(len(y_true), dtype=np.float32)
    for i, q in enumerate(queries):
        pred, _, hits = crs.inspect_query(q)
        preds[i] = pred
        rule_scores[i] = min(1.0, hits / 3.0)
    return compute_metrics_safe(y_true, preds, rule_scores), preds


def fmt_fpr(val) -> str:
    return "N/A" if val is None else f"{val:.4f}"


def main() -> None:
    t0 = time.perf_counter_ns()
    cfg = json.loads(ROUTER_CFG.read_text(encoding="utf-8"))
    engine = CascadeSQLOnnxEngine(
        tau_low=cfg["tau_low"],
        tau_high=cfg["tau_high"],
        tau_stage2=cfg["tau_stage2"],
    )
    crs = OwaspCrsV4Baseline(anomaly_threshold=1)

    df_test = pd.read_csv(TEST_CSV, low_memory=False)
    df_imb = pd.read_csv(IMB_CSV, low_memory=False)
    df_adv = pd.read_csv(ADV_CSV, low_memory=False)

    cas_test, pred_cas_test, p1_test, score_test = evaluate_cascade_on_df(engine, df_test)
    cas_imb, _, _, _ = evaluate_cascade_on_df(engine, df_imb)
    cas_adv, pred_cas_adv, _, _ = evaluate_cascade_on_df(engine, df_adv)

    crs_test, pred_crs_test = evaluate_crs_on_df(crs, df_test)
    crs_imb, _ = evaluate_crs_on_df(crs, df_imb)
    crs_adv, pred_crs_adv = evaluate_crs_on_df(crs, df_adv)

    # Breakdown per adversarial obfuscation strategy & dialect
    df_adv_eval = df_adv.copy()
    df_adv_eval["cas_pred"] = pred_cas_adv
    df_adv_eval["crs_pred"] = pred_crs_adv
    strategy_breakdown = {}
    for strat, grp in df_adv_eval.groupby("mutation_strategy"):
        strategy_breakdown[str(strat)] = {
            "count": int(len(grp)),
            "cascade_recall": round(float(grp["cas_pred"].mean()), 6),
            "crs_v4_recall": round(float(grp["crs_pred"].mean()), 6),
        }

    dialect_breakdown = {}
    df_test_eval = df_test.copy()
    df_test_eval["cas_pred"] = pred_cas_test
    for d_name, grp in df_test_eval.groupby("dialect"):
        m_d = compute_metrics_safe(grp["label"].values.astype(np.int64), grp["cas_pred"].values.astype(np.int64))
        dialect_breakdown[str(d_name)] = {
            "samples": m_d["samples"],
            "f1_score": m_d["f1_score"],
            "fpr": m_d["false_positive_rate"],
            "fnr": m_d["false_negative_rate"],
        }

    # Save exact row-level predictions on held-out test set for exact McNemar's test in RQ4
    np.savez_compressed(
        PREDS_NPZ,
        y_true=df_test["label"].astype(np.int64).values,
        pred_cascade=pred_cas_test,
        pred_crs=pred_crs_test,
        p1_test=p1_test,
        score_test=score_test,
    )

    tab2_tex = rf"""% Auto-generated from 100% Real ONNX INT8 RQ1 Execution Audit
\begin{{table*}}[htbp]
\centering
\caption{{Detection performance comparison between OWASP ModSecurity CRS v4 and CASCADE-SQL (INT8 ONNX, $\tau_{{\text{{low}}}}={cfg['tau_low']:.2f}, \tau_{{\text{{high}}}}={cfg['tau_high']:.2f}$) across held-out test, 100:1 realistic imbalanced, and adversarial obfuscated suites.}}
\label{{tab:rq1_performance}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrr}}
\toprule
\textbf{{Evaluation Suite}} & \textbf{{Defense Model}} & \textbf{{Samples}} & \textbf{{Accuracy}} & \textbf{{Precision}} & \textbf{{Recall}} & \textbf{{F1-Score}} & \textbf{{FPR}} & \textbf{{FNR}} \\
\midrule
Standard Test Split & ModSecurity CRS v4 & {crs_test['samples']:,} & {crs_test['accuracy']:.4f} & {crs_test['precision']:.4f} & {crs_test['recall']:.4f} & {crs_test['f1_score']:.4f} & {fmt_fpr(crs_test['false_positive_rate'])} & {crs_test['false_negative_rate']:.4f} \\
Standard Test Split & \textbf{{CASCADE-SQL (Ours)}} & {cas_test['samples']:,} & \textbf{{{cas_test['accuracy']:.4f}}} & \textbf{{{cas_test['precision']:.4f}}} & \textbf{{{cas_test['recall']:.4f}}} & \textbf{{{cas_test['f1_score']:.4f}}} & \textbf{{{fmt_fpr(cas_test['false_positive_rate'])}}} & \textbf{{{cas_test['false_negative_rate']:.4f}}} \\
\midrule
Imbalanced (100:1) & ModSecurity CRS v4 & {crs_imb['samples']:,} & {crs_imb['accuracy']:.4f} & {crs_imb['precision']:.4f} & {crs_imb['recall']:.4f} & {crs_imb['f1_score']:.4f} & {fmt_fpr(crs_imb['false_positive_rate'])} & {crs_imb['false_negative_rate']:.4f} \\
Imbalanced (100:1) & \textbf{{CASCADE-SQL (Ours)}} & {cas_imb['samples']:,} & \textbf{{{cas_imb['accuracy']:.4f}}} & \textbf{{{cas_imb['precision']:.4f}}} & \textbf{{{cas_imb['recall']:.4f}}} & \textbf{{{cas_imb['f1_score']:.4f}}} & \textbf{{{fmt_fpr(cas_imb['false_positive_rate'])}}} & \textbf{{{cas_imb['false_negative_rate']:.4f}}} \\
\midrule
Adversarial Obfuscated & ModSecurity CRS v4 & {crs_adv['samples']:,} & {crs_adv['accuracy']:.4f} & {crs_adv['precision']:.4f} & {crs_adv['recall']:.4f} & {crs_adv['f1_score']:.4f} & N/A & {crs_adv['false_negative_rate']:.4f} \\
Adversarial Obfuscated & \textbf{{CASCADE-SQL (Ours)}} & {cas_adv['samples']:,} & \textbf{{{cas_adv['accuracy']:.4f}}} & \textbf{{{cas_adv['precision']:.4f}}} & \textbf{{{cas_adv['recall']:.4f}}} & \textbf{{{cas_adv['f1_score']:.4f}}} & N/A & \textbf{{{cas_adv['false_negative_rate']:.4f}}} \\
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    write_utf8_lf(TAB2_TEX, tab2_tex)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_ms": round((time.perf_counter_ns() - t0) / 1e6, 3),
        "thresholds_used": {
            "tau_low": cfg["tau_low"],
            "tau_high": cfg["tau_high"],
            "tau_stage2": cfg["tau_stage2"],
        },
        "cascade_sql_onnx_int8": {
            "held_out_test": cas_test,
            "imbalanced_100_1": cas_imb,
            "adversarial_obfuscated": cas_adv,
            "dialect_breakdown_test": dialect_breakdown,
        },
        "owasp_crs_v4_baseline": {
            "held_out_test": crs_test,
            "imbalanced_100_1": crs_imb,
            "adversarial_obfuscated": crs_adv,
        },
        "adversarial_strategy_recall_comparison": strategy_breakdown,
        "sha256_verification": {
            "test_csv": sha256_of(TEST_CSV),
            "imbalanced_csv": sha256_of(IMB_CSV),
            "adversarial_csv": sha256_of(ADV_CSV),
            "router_thresholds_json": sha256_of(ROUTER_CFG),
            "stage1_int8_onnx": sha256_of(engine.s1_path),
            "stage2_int8_onnx": sha256_of(engine.s2_path),
            "predictions_npz": sha256_of(PREDS_NPZ),
            "tab2_rq1_performance_tex": sha256_of(TAB2_TEX),
        },
    }

    write_utf8_lf(LOG_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()