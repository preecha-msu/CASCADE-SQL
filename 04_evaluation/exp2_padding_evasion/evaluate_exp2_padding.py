import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))

from cascade_onnx_engine import CascadeSQLOnnxEngine

TEST_CSV = ROOT / "01_datasets" / "processed" / "test.csv"
ROUTER_CFG = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "exp2_padding_bypass_audit.json"
TAB7_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab7_exp2_padding_bypass.tex"


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


def build_benign_sql_prefix(target_bytes: int, idx: int) -> str:
    """Construct realistic benign SQL WHERE clauses of exact target byte length before injecting tail SQLi."""
    if target_bytes <= 0:
        return ""
    base = (
        f"SELECT order_id, customer_ref, total_amount, status_text FROM enterprise_orders "
        f"WHERE tenant_code = 'ORG_{idx % 97:03d}' AND region_name = 'APAC_PROD' AND notes_text = '"
    )
    filler_unit = "verified_transaction_record_item_"
    rem = max(0, target_bytes - len(base) - 2)
    repeats = (rem // len(filler_unit)) + 1
    body = (filler_unit * repeats)[:rem]
    return base + body + "' "


def craft_tail_injected_query(benign_prefix: str, raw_sqli: str) -> str:
    if not benign_prefix:
        return raw_sqli
    s = raw_sqli.strip()
    if s.upper().startswith(("UNION", "AND", "OR", ";", "/*")):
        return benign_prefix + s
    if s.startswith(("'", '"', ")")):
        return benign_prefix + "AND product_tag = " + s
    return benign_prefix + "UNION ALL " + s


def main() -> None:
    t0 = time.perf_counter_ns()
    cfg = json.loads(ROUTER_CFG.read_text(encoding="utf-8"))

    eng_trunc = CascadeSQLOnnxEngine(
        cfg["tau_low"], cfg["tau_high"], cfg["tau_stage2"], window_mode="naive_truncated_256"
    )
    eng_strided = CascadeSQLOnnxEngine(
        cfg["tau_low"], cfg["tau_high"], cfg["tau_stage2"], window_mode="strided_multi_window"
    )

    df_test = pd.read_csv(TEST_CSV, low_memory=False)

    # Part A: Regression check on full 9,920 held-out test set (verify FPR & F1 do not degrade)
    q_all = df_test["query"].fillna("").astype(str).tolist()
    d_all = df_test["dialect"].fillna("mysql").astype(str).tolist()
    y_all = df_test["label"].astype(np.int64).values

    long_benign_count = sum(1 for q, y in zip(q_all, y_all) if y == 0 and len(q) > 256)
    preds_strided_test = np.zeros(len(q_all), dtype=np.int64)
    lat_strided_test = np.zeros(len(q_all), dtype=np.float64)

    # Warmup
    for i in range(25):
        eng_strided.inspect_single_query_timed(q_all[i], d_all[i])
        eng_trunc.inspect_single_query_timed(q_all[i], d_all[i])

    for i, (q, d) in enumerate(zip(q_all, d_all)):
        res = eng_strided.inspect_single_query_timed(q, d)
        preds_strided_test[i] = res["pred"]
        lat_strided_test[i] = res["total_ms"]

    tp = int(((preds_strided_test == 1) & (y_all == 1)).sum())
    fp = int(((preds_strided_test == 1) & (y_all == 0)).sum())
    fn = int(((preds_strided_test == 0) & (y_all == 1)).sum())
    tn = int(((preds_strided_test == 0) & (y_all == 0)).sum())
    reg_f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    reg_fpr = fp / max(1, fp + tn)

    # Part B: Prefix-Padding Tail-Injection Benchmark across 6 Padding Lengths (1,000 real test SQLi queries each)
    df_sqli = df_test[df_test["label"] == 1].iloc[:1000].reset_index(drop=True)
    padding_lengths = [0, 256, 512, 1024, 2048, 4096]
    padding_results = []

    for pad_b in padding_lengths:
        trunc_hits = 0
        strided_hits = 0
        trunc_ms_list = []
        strided_ms_list = []
        win_counts = []

        for idx, row in enumerate(df_sqli.itertuples(index=False)):
            prefix = build_benign_sql_prefix(pad_b, idx)
            adv_q = craft_tail_injected_query(prefix, str(row.query))
            dialect = str(row.dialect)

            r_tr = eng_trunc.inspect_single_query_timed(adv_q, dialect)
            r_st = eng_strided.inspect_single_query_timed(adv_q, dialect)

            trunc_hits += r_tr["pred"]
            strided_hits += r_st["pred"]
            trunc_ms_list.append(r_tr["total_ms"])
            strided_ms_list.append(r_st["total_ms"])
            win_counts.append(r_st["windows_evaluated"])

        n_eval = len(df_sqli)
        trunc_rec = trunc_hits / float(n_eval)
        strided_rec = strided_hits / float(n_eval)

        padding_results.append({
            "benign_prefix_bytes": pad_b,
            "samples": n_eval,
            "mean_windows_scanned": round(float(np.mean(win_counts)), 2),
            "naive_truncated_recall": round(trunc_rec, 6),
            "naive_truncated_escape_rate_fnr": round(1.0 - trunc_rec, 6),
            "naive_truncated_mean_ms": round(float(np.mean(trunc_ms_list)), 4),
            "strided_cascade_recall": round(strided_rec, 6),
            "strided_cascade_escape_rate_fnr": round(1.0 - strided_rec, 6),
            "strided_cascade_mean_ms": round(float(np.mean(strided_ms_list)), 4),
            "strided_cascade_p99_ms": round(float(np.percentile(strided_ms_list, 99)), 4),
        })

    rows_tex = []
    for r in padding_results:
        rows_tex.append(
            f"{r['benign_prefix_bytes']:,} B & {r['mean_windows_scanned']:.1f} & "
            f"{r['naive_truncated_recall']*100:.2f}\\% & {r['naive_truncated_escape_rate_fnr']*100:.2f}\\% & "
            f"\\textbf{{{r['strided_cascade_recall']*100:.2f}\\%}} & \\textbf{{{r['strided_cascade_escape_rate_fnr']*100:.2f}\\%}} & "
            f"{r['strided_cascade_mean_ms']:.3f} & {r['strided_cascade_p99_ms']:.3f} \\\\"
        )
    joined_rows = "\n".join(rows_tex)

    tab7_tex = rf"""% Auto-generated from 100% Real Prefix-Padding Tail-Injection Benchmark (Exp-2)
\begin{{table*}}[htbp]
\centering
\caption{{Resilience against Prefix-Padding Tail-Injection attacks ($N=1,000$ held-out SQLi payloads per padding tier) comparing Naive 256-Byte Input Truncation against CASCADE-SQL's Bounded Strided Multi-Window Pooling ($W=256, S=192$).}}
\label{{tab:exp2_padding}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{rrrrrrRR}}
\toprule
\textbf{{Benign Prefix}} & \textbf{{Windows ($K$)}} & \textbf{{Truncated Recall}} & \textbf{{Truncated Escape (FNR)}} & \textbf{{Strided Recall (Ours)}} & \textbf{{Strided Escape (FNR)}} & \textbf{{Mean (ms)}} & \textbf{{$p_{{99}}$ (ms)}} \\
\midrule
{joined_rows}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
""".replace("rrrrrrRR", "rrrrrrrr")
    write_utf8_lf(TAB7_TEX, tab7_tex)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_sec": round((time.perf_counter_ns() - t0) / 1e9, 3),
        "held_out_test_regression_check": {
            "total_samples": len(q_all),
            "benign_queries_over_256_bytes": long_benign_count,
            "strided_test_f1": round(reg_f1, 6),
            "strided_test_fpr": round(reg_fpr, 6),
            "strided_test_mean_ms": round(float(np.mean(lat_strided_test)), 4),
            "strided_test_p99_ms": round(float(np.percentile(lat_strided_test, 99)), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "prefix_padding_tail_injection_results": padding_results,
        "sha256_verification": {
            "upgraded_engine_py": sha256_of(ROOT / "02_models_colab/src/cascade_router/cascade_onnx_engine.py"),
            "tab7_exp2_padding_bypass_tex": sha256_of(TAB7_TEX),
        },
    }

    write_utf8_lf(LOG_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()