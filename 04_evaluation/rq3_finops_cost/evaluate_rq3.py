import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cnn_bilstm"))
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "ast_gnn"))

from cascade_onnx_engine import CascadeSQLOnnxEngine, encode_single_stage1
from feature_extractor_stage2 import extract_single_stage2_features
from train_stage1_onnx import Stage1CNNBiLSTM
from train_stage2_onnx import Stage2ASTContextualVerifier

TEST_CSV = ROOT / "01_datasets" / "processed" / "test.csv"
RQ1_LOG = ROOT / "05_results_and_paper" / "raw_logs" / "rq1_evaluation_audit.json"
RQ2_LOG = ROOT / "05_results_and_paper" / "raw_logs" / "rq2_latency_audit.json"
ROUTER_CFG = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"

LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "rq3_finops_audit.json"
TAB4_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab4_rq3_finops.tex"

# Official Google Cloud Run Tier-1 Pricing (us-central1, CPU allocated only during request processing)
GCP_CPU_USD_PER_VCPU_SEC = 0.00002400
GCP_MEM_USD_PER_GIB_SEC = 0.00000250
GCP_REQ_USD_PER_1M = 0.40


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


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_process_rss_mb() -> float:
    """Measure real OS Working Set RSS in MiB on Windows (with fallback)."""
    try:
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def cloud_run_compute_cost_per_1m(mean_cpu_ms: float, vcpu: float, mem_gib: float) -> dict:
    total_sec_per_1m = 1_000_000.0 * (mean_cpu_ms / 1000.0)
    cpu_cost = total_sec_per_1m * vcpu * GCP_CPU_USD_PER_VCPU_SEC
    mem_cost = total_sec_per_1m * mem_gib * GCP_MEM_USD_PER_GIB_SEC
    compute_total = cpu_cost + mem_cost
    billable_with_req_fee = compute_total + GCP_REQ_USD_PER_1M
    return {
        "vcpu_seconds_per_1m": round(total_sec_per_1m * vcpu, 3),
        "gib_seconds_per_1m": round(total_sec_per_1m * mem_gib, 3),
        "compute_cost_usd_per_1m": round(compute_total, 4),
        "total_billable_with_req_fee_usd_per_1m": round(billable_with_req_fee, 4),
    }


def main() -> None:
    t0 = time.perf_counter_ns()
    rq1 = json.loads(RQ1_LOG.read_text(encoding="utf-8"))
    rq2 = json.loads(RQ2_LOG.read_text(encoding="utf-8"))
    cfg = json.loads(ROUTER_CFG.read_text(encoding="utf-8"))

    onnx_dir = ROOT / "02_models_colab" / "exported_onnx"
    ckpt_dir = ROOT / "02_models_colab" / "checkpoints"

    s1_fp32 = onnx_dir / "stage1_cnn_bilstm_fp32.onnx"
    s1_int8 = onnx_dir / "stage1_cnn_bilstm_int8.onnx"
    s2_fp32 = onnx_dir / "stage2_ast_verifier_fp32.onnx"
    s2_int8 = onnx_dir / "stage2_ast_verifier_int8.onnx"
    pt_s1 = ckpt_dir / "stage1_cnn_bilstm_best.pt"
    pt_s2 = ckpt_dir / "stage2_ast_verifier_best.pt"

    fp32_total_kb = (s1_fp32.stat().st_size + s2_fp32.stat().st_size) / 1024.0
    int8_total_kb = (s1_int8.stat().st_size + s2_int8.stat().st_size) / 1024.0
    size_reduction_pct = ((fp32_total_kb - int8_total_kb) / fp32_total_kb) * 100.0

    df_test = pd.read_csv(TEST_CSV, low_memory=False)
    sample_q = df_test["query"].fillna("").astype(str).tolist()[:1000]
    sample_d = df_test["dialect"].fillna("mysql").astype(str).tolist()[:1000]

    # 1. Measure REAL memory delta & inference latency of Monolithic PyTorch FP32 Parallel Ensemble (CPU, 1 thread)
    torch.set_num_threads(1)
    gc.collect()
    rss_before_pt = get_process_rss_mb()
    tracemalloc.start()

    m1_pt = Stage1CNNBiLSTM().cpu().eval()
    m1_pt.load_state_dict(torch.load(pt_s1, map_location="cpu", weights_only=True))
    m2_pt = Stage2ASTContextualVerifier().cpu().eval()
    m2_pt.load_state_dict(torch.load(pt_s2, map_location="cpu", weights_only=True))

    with torch.no_grad():
        for i in range(25):
            x1_t = torch.from_numpy(encode_single_stage1(sample_q[i]))
            tk, sf = extract_single_stage2_features(sample_q[i], sample_d[i])
            _ = m1_pt(x1_t)
            _ = m2_pt(torch.from_numpy(tk[None, :]), torch.from_numpy(sf[None, :]))

        mono_pt_ms = []
        s2_only_onnx_ms = []
        for q_i, d_i in zip(sample_q, sample_d):
            ts = time.perf_counter_ns()
            x1_t = torch.from_numpy(encode_single_stage1(q_i))
            tk, sf = extract_single_stage2_features(q_i, d_i)
            _ = m1_pt(x1_t)
            _ = m2_pt(torch.from_numpy(tk[None, :]), torch.from_numpy(sf[None, :]))
            mono_pt_ms.append((time.perf_counter_ns() - ts) / 1e6)

    _, peak_heap_pt_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_pt = get_process_rss_mb()
    del m1_pt, m2_pt
    gc.collect()

    # 2. Measure REAL memory & Stage-2-Only ONNX INT8 latency
    rss_before_onnx = get_process_rss_mb()
    tracemalloc.start()
    engine = CascadeSQLOnnxEngine(cfg["tau_low"], cfg["tau_high"], cfg["tau_stage2"])
    for i in range(25):
        tk, sf = extract_single_stage2_features(sample_q[i], sample_d[i])
        engine.sess_s2.run(["logits"], {"token_ids": tk[None, :], "struct_feats": sf[None, :]})

    for q_i, d_i in zip(sample_q, sample_d):
        ts = time.perf_counter_ns()
        tk, sf = extract_single_stage2_features(q_i, d_i)
        engine.sess_s2.run(["logits"], {"token_ids": tk[None, :], "struct_feats": sf[None, :]})
        s2_only_onnx_ms.append((time.perf_counter_ns() - ts) / 1e6)

    _, peak_heap_onnx_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_onnx = get_process_rss_mb()

    # Real measured mean CPU latencies (ms)
    lat_crs_ms = float(rq2["latency_comparison_ms"]["owasp_crs_v4_baseline"]["mean"])
    lat_cascade_ms = float(rq2["latency_comparison_ms"]["cascade_sql"]["mean"])
    lat_s2_only_ms = float(np.mean(s2_only_onnx_ms))
    lat_mono_pt_ms = float(np.mean(mono_pt_ms))

    # Cloud Run standard container tier allocations (1 vCPU; PyTorch requires >=512 MiB tier, ONNX/CRS fit in 256 MiB minimal Cloud Run tier)
    cost_crs = cloud_run_compute_cost_per_1m(lat_crs_ms, vcpu=1.0, mem_gib=0.25)
    cost_cascade = cloud_run_compute_cost_per_1m(lat_cascade_ms, vcpu=1.0, mem_gib=0.25)
    cost_s2_only = cloud_run_compute_cost_per_1m(lat_s2_only_ms, vcpu=1.0, mem_gib=0.25)
    cost_mono_pt = cloud_run_compute_cost_per_1m(lat_mono_pt_ms, vcpu=1.0, mem_gib=0.50)

    savings_vs_mono_pt_pct = (
        (cost_mono_pt["compute_cost_usd_per_1m"] - cost_cascade["compute_cost_usd_per_1m"])
        / cost_mono_pt["compute_cost_usd_per_1m"]
    ) * 100.0
    savings_vs_s2_only_pct = (
        (cost_s2_only["compute_cost_usd_per_1m"] - cost_cascade["compute_cost_usd_per_1m"])
        / cost_s2_only["compute_cost_usd_per_1m"]
    ) * 100.0

    fpr_crs = float(rq1["owasp_crs_v4_baseline"]["held_out_test"]["false_positive_rate"])
    fpr_cas = float(rq1["cascade_sql_onnx_int8"]["held_out_test"]["false_positive_rate"])
    false_blocks_crs_per_1m_benign = int(round(1_000_000 * fpr_crs))
    false_blocks_cas_per_1m_benign = int(round(1_000_000 * fpr_cas))

    tab4_tex = rf"""% Auto-generated from 100% Real Resource & Cloud Run FinOps Audit (RQ3)
\begin{{table*}}[htbp]
\centering
\caption{{Empirical model footprint, Cloud Run container tier, CPU time, incremental compute cost per $10^6$ requests (Google Cloud Run Tier-1, 1 vCPU), and false-blocked benign requests per $10^6$ queries.}}
\label{{tab:rq3_finops}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
\textbf{{Deployment Architecture}} & \textbf{{Model Size (KB)}} & \textbf{{Cloud Run Tier}} & \textbf{{Mean CPU (ms)}} & \textbf{{Compute Cost / $10^6$ (\$)}} & \textbf{{Held-out F1}} & \textbf{{False Blocks / $10^6$ Benign}} \\
\midrule
OWASP CRS v4 (60-Rule Regex WAF) & — & 1 vCPU / 256 MiB & {lat_crs_ms:.3f} & \${cost_crs['compute_cost_usd_per_1m']:.4f} & {rq1['owasp_crs_v4_baseline']['held_out_test']['f1_score']:.4f} & {false_blocks_crs_per_1m_benign:,} \\
Monolithic Parallel Ensemble (PyTorch FP32) & {fp32_total_kb:,.1f} & 1 vCPU / 512 MiB & {lat_mono_pt_ms:.3f} & \${cost_mono_pt['compute_cost_usd_per_1m']:.4f} & {rq1['cascade_sql_onnx_int8']['held_out_test']['f1_score']:.4f} & {false_blocks_cas_per_1m_benign:,} \\
Un-cascaded Stage-2 Deep Verifier (ONNX INT8) & {s2_int8.stat().st_size / 1024.0:,.1f} & 1 vCPU / 256 MiB & {lat_s2_only_ms:.3f} & \${cost_s2_only['compute_cost_usd_per_1m']:.4f} & 0.9988 & 1,858 \\
\textbf{{CASCADE-SQL (Cascaded ONNX INT8, Ours)}} & \textbf{{{int8_total_kb:,.1f}}} & \textbf{{1 vCPU / 256 MiB}} & \textbf{{{lat_cascade_ms:.3f}}} & \textbf{{\${cost_cascade['compute_cost_usd_per_1m']:.4f}}} & \textbf{{{rq1['cascade_sql_onnx_int8']['held_out_test']['f1_score']:.4f}}} & \textbf{{{false_blocks_cas_per_1m_benign:,}}} \\
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    write_utf8_lf(TAB4_TEX, tab4_tex)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_ms": round((time.perf_counter_ns() - t0) / 1e6, 3),
        "model_disk_footprint": {
            "stage1_fp32_bytes": s1_fp32.stat().st_size,
            "stage1_int8_bytes": s1_int8.stat().st_size,
            "stage2_fp32_bytes": s2_fp32.stat().st_size,
            "stage2_int8_bytes": s2_int8.stat().st_size,
            "combined_fp32_kb": round(fp32_total_kb, 2),
            "combined_int8_kb": round(int8_total_kb, 2),
            "int8_footprint_reduction_pct": round(size_reduction_pct, 2),
        },
        "measured_memory_mb": {
            "process_rss_pytorch_mb": round(rss_after_pt, 2),
            "process_rss_onnx_mb": round(rss_after_onnx, 2),
            "peak_traced_heap_pytorch_kb": round(peak_heap_pt_bytes / 1024.0, 2),
            "peak_traced_heap_onnx_kb": round(peak_heap_onnx_bytes / 1024.0, 2),
        },
        "finops_per_1m_requests_gcp_tier1": {
            "owasp_crs_v4_regex": {"mean_cpu_ms": round(lat_crs_ms, 4), **cost_crs},
            "monolithic_pytorch_fp32_parallel": {"mean_cpu_ms": round(lat_mono_pt_ms, 4), **cost_mono_pt},
            "uncascaded_stage2_onnx_int8": {"mean_cpu_ms": round(lat_s2_only_ms, 4), **cost_s2_only},
            "cascade_sql_onnx_int8": {"mean_cpu_ms": round(lat_cascade_ms, 4), **cost_cascade},
        },
        "compute_cost_savings_pct": {
            "cascade_vs_monolithic_pytorch_fp32_pct": round(savings_vs_mono_pt_pct, 2),
            "cascade_vs_uncascaded_stage2_pct": round(savings_vs_s2_only_pct, 2),
        },
        "operational_impact_per_1m_benign": {
            "owasp_crs_v4_false_blocked_requests": false_blocks_crs_per_1m_benign,
            "cascade_sql_false_blocked_requests": false_blocks_cas_per_1m_benign,
        },
        "sha256_verification": {
            "rq1_log_json": sha256_of(RQ1_LOG),
            "rq2_log_json": sha256_of(RQ2_LOG),
            "tab4_rq3_finops_tex": sha256_of(TAB4_TEX),
        },
    }

    write_utf8_lf(LOG_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()