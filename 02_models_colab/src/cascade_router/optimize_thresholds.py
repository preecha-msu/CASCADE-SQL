import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))

from cascade_onnx_engine import CascadeSQLOnnxEngine, encode_single_stage1
from feature_extractor_stage2 import check_dos_guard, extract_single_stage2_features

VAL_CSV = ROOT / "01_datasets" / "processed" / "val.csv"
CONFIG_PATH = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "stage2_threshold_optimization_audit.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def val_obfuscate(query: str, dialect: str, idx: int) -> str:
    """Validation-only syntactic obfuscation probe to expose gray-zone uncertainty without touching Test."""
    q = query.strip()
    m = idx % 4
    if m == 0:
        return re.sub(r"\s+", "/**/", q)
    elif m == 1:
        if str(dialect).lower().startswith("mysql"):
            return re.sub(r"\b(UNION|SELECT|AND|OR)\b", r"/*!50000\1*/", q, flags=re.IGNORECASE)
        return re.sub(r"\b(UNION|SELECT|AND|OR)\b", r"/*pg*/\1/*pg*/", q, flags=re.IGNORECASE)
    elif m == 2:
        return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(q))
    return re.sub(r"1\s*=\s*1", "0x31=0x31", q)


def eval_cascade_arrays(
    y_true: np.ndarray,
    dos_mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    tau_low: float,
    tau_high: float,
    tau_s2: float = 0.5,
) -> dict:
    n = len(y_true)
    esc_mask = (~dos_mask) & (p1 > tau_low) & (p1 < tau_high)
    fast_pos = (~dos_mask) & (p1 >= tau_high)

    preds = np.zeros(n, dtype=np.int64)
    preds[dos_mask] = 1
    preds[fast_pos] = 1
    preds[esc_mask] = (p2[esc_mask] >= tau_s2).astype(np.int64)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    acc = float((preds == y_true).mean())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, tp + fn)
    rho = float(esc_mask.mean())

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "fpr": fpr,
        "fnr": fnr,
        "rho": rho,
        "escalated_count": int(esc_mask.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    t0 = time.perf_counter_ns()
    df_val = pd.read_csv(VAL_CSV, low_memory=False)

    val_q = df_val["query"].fillna("").astype(str).tolist()
    val_d = df_val["dialect"].fillna("mysql").astype(str).tolist()
    val_y = df_val["label"].astype(np.int64).values

    engine = CascadeSQLOnnxEngine()

    # 1. Measure real per-stage single-query latency on 500 validation samples (after warmup)
    for i in range(30):
        engine.sess_s1.run(["logits"], {"input_ids": encode_single_stage1(val_q[i])})
        tk, sf = extract_single_stage2_features(val_q[i], val_d[i])
        engine.sess_s2.run(["logits"], {"token_ids": tk[None, :], "struct_feats": sf[None, :]})

    t1_list, t2_list = [], []
    for i in range(min(500, len(val_q))):
        ts = time.perf_counter_ns()
        x1 = encode_single_stage1(val_q[i])
        engine.sess_s1.run(["logits"], {"input_ids": x1})
        te = time.perf_counter_ns()
        t1_list.append((te - ts) / 1e6)

        ts2 = time.perf_counter_ns()
        tk, sf = extract_single_stage2_features(val_q[i], val_d[i])
        engine.sess_s2.run(["logits"], {"token_ids": tk[None, :], "struct_feats": sf[None, :]})
        te2 = time.perf_counter_ns()
        t2_list.append((te2 - ts2) / 1e6)

    mean_t1_ms = float(np.mean(t1_list))
    mean_t2_ms = float(np.mean(t2_list))

    # 2. Compute real ONNX INT8 probabilities on clean Validation set + Validation obfuscation probes
    p1_clean = engine.predict_stage1_probs_batch(val_q)
    p2_clean = engine.predict_stage2_probs_batch(val_q, val_d)
    dos_clean = np.array([check_dos_guard(q) for q in val_q], dtype=bool)

    sqli_indices = [i for i, y in enumerate(val_y) if y == 1]
    obf_q = [val_obfuscate(val_q[i], val_d[i], k) for k, i in enumerate(sqli_indices)]
    obf_d = [val_d[i] for i in sqli_indices]
    obf_y = np.ones(len(obf_q), dtype=np.int64)
    p1_obf = engine.predict_stage1_probs_batch(obf_q)
    p2_obf = engine.predict_stage2_probs_batch(obf_q, obf_d)
    dos_obf = np.array([check_dos_guard(q) for q in obf_q], dtype=bool)

    y_calib = np.concatenate([val_y, obf_y])
    p1_calib = np.concatenate([p1_clean, p1_obf])
    p2_calib = np.concatenate([p2_clean, p2_obf])
    dos_calib = np.concatenate([dos_clean, dos_obf])

    # 3. Constrained Optimization Grid Search
    tau_low_grid = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    tau_high_grid = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995]

    eps_fpr = 0.0025   # Strict <= 0.25% FPR ceiling on Validation
    eps_fnr = 0.0050   # Strict <= 0.50% FNR ceiling on Combined Validation + Obfuscated Validation

    feasible_candidates = []
    all_candidates = []

    for tl in tau_low_grid:
        for th in tau_high_grid:
            if tl >= th:
                continue
            m_cal = eval_cascade_arrays(y_calib, dos_calib, p1_calib, p2_calib, tl, th)
            m_cln = eval_cascade_arrays(val_y, dos_clean, p1_clean, p2_clean, tl, th)
            exp_lat_ms = mean_t1_ms + m_cln["rho"] * mean_t2_ms

            entry = {
                "tau_low": tl,
                "tau_high": th,
                "val_clean": m_cln,
                "val_combined": m_cal,
                "expected_latency_ms": exp_lat_ms,
            }
            all_candidates.append(entry)

            if m_cln["fpr"] <= eps_fpr and m_cal["fnr"] <= eps_fnr and m_cln["rho"] <= 0.15:
                feasible_candidates.append(entry)

    constraint_status = "FEASIBLE_OPTIMAL"
    if feasible_candidates:
        # Maximize combined F1 while minimizing expected latency
        feasible_candidates.sort(
            key=lambda c: (-round(c["val_combined"]["f1_score"], 5), c["expected_latency_ms"])
        )
        best = feasible_candidates[0]
    else:
        constraint_status = "FALLBACK_PENALTY_USED"
        all_candidates.sort(
            key=lambda c: (
                c["expected_latency_ms"]
                + 500.0 * max(0.0, c["val_clean"]["fpr"] - eps_fpr)
                + 500.0 * max(0.0, c["val_combined"]["fnr"] - eps_fnr)
            )
        )
        best = all_candidates[0]

    best_tl = float(best["tau_low"])
    best_th = float(best["tau_high"])
    cln = best["val_clean"]
    cmb = best["val_combined"]

    router_cfg = {
        "tau_low": best_tl,
        "tau_high": best_th,
        "tau_stage2": 0.50,
        "measured_mean_stage1_ms": round(mean_t1_ms, 4),
        "measured_mean_stage2_ms": round(mean_t2_ms, 4),
        "constraint_status": constraint_status,
        "stage1_onnx_sha256": sha256_of(engine.s1_path),
        "stage2_onnx_sha256": sha256_of(engine.s2_path),
    }
    with CONFIG_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(router_cfg, f, indent=2)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_ms": round((time.perf_counter_ns() - t0) / 1e6, 3),
        "constraint_status": constraint_status,
        "constraints": {"eps_fpr": eps_fpr, "eps_fnr": eps_fnr},
        "measured_stage_latencies_ms": {
            "stage1_int8_mean_ms": round(mean_t1_ms, 4),
            "stage2_ast_int8_mean_ms": round(mean_t2_ms, 4),
        },
        "validation_samples_clean": int(len(val_y)),
        "validation_obfuscated_probes": int(len(obf_y)),
        "optimal_thresholds": {"tau_low": best_tl, "tau_high": best_th, "tau_stage2": 0.50},
        "performance_on_clean_val": {
            "escalation_rate_pct": round(cln["rho"] * 100.0, 3),
            "escalated_queries": cln["escalated_count"],
            "expected_latency_ms": round(best["expected_latency_ms"], 4),
            "accuracy": round(cln["accuracy"], 6),
            "precision": round(cln["precision"], 6),
            "recall": round(cln["recall"], 6),
            "f1_score": round(cln["f1_score"], 6),
            "fpr": round(cln["fpr"], 6),
            "fnr": round(cln["fnr"], 6),
            "tp": cln["tp"], "fp": cln["fp"], "fn": cln["fn"], "tn": cln["tn"],
        },
        "performance_on_combined_val_with_obfuscation": {
            "escalation_rate_pct": round(cmb["rho"] * 100.0, 3),
            "f1_score": round(cmb["f1_score"], 6),
            "fpr": round(cmb["fpr"], 6),
            "fnr": round(cmb["fnr"], 6),
        },
        "artifacts_sha256": {
            "val_csv": sha256_of(VAL_CSV),
            "stage1_int8_onnx": sha256_of(engine.s1_path),
            "stage2_int8_onnx": sha256_of(engine.s2_path),
            "router_thresholds_json": sha256_of(CONFIG_PATH),
        },
    }

    with LOG_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()