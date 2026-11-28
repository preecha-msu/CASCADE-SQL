import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "05_results_and_paper" / "raw_logs"
FIG_DIR = ROOT / "05_results_and_paper" / "figures_pdf"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rq1 = json.loads((LOGS_DIR / "rq1_evaluation_audit.json").read_text(encoding="utf-8"))
    rq2 = json.loads((LOGS_DIR / "rq2_latency_audit.json").read_text(encoding="utf-8"))
    rq3 = json.loads((LOGS_DIR / "rq3_finops_audit.json").read_text(encoding="utf-8"))
    rq4 = json.loads((LOGS_DIR / "rq4_ablation_audit.json").read_text(encoding="utf-8"))
    rq5 = json.loads((LOGS_DIR / "rq5_xai_audit.json").read_text(encoding="utf-8"))
    lat_npz = np.load(LOGS_DIR / "rq2_real_latencies.npz")

    # --- Figure 1: Empirical Per-Query Latency CDF & Stage Breakdown ---
    cas_ms = np.sort(lat_npz["cascade_total_ms"])
    waf_ms = np.sort(lat_npz["waf_crs_total_ms"])
    cdf_y = np.arange(1, len(cas_ms) + 1) / float(len(cas_ms))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.plot(cas_ms, cdf_y, lw=1.8, color="#0b559f", label=f"CASCADE-SQL INT8 (p50={rq2['latency_comparison_ms']['cascade_sql']['p50']:.3f} ms)")
    ax1.plot(waf_ms, cdf_y, lw=1.6, ls="--", color="#d95f02", label=f"OWASP CRS v4 60-Rule (p50={rq2['latency_comparison_ms']['owasp_crs_v4_baseline']['p50']:.3f} ms)")
    ax1.axhline(0.95, color="gray", ls=":", lw=0.9)
    ax1.axhline(0.99, color="black", ls=":", lw=0.9)
    ax1.set_xlim(0.0, 1.8)
    ax1.set_xlabel("Single-Query Processing Latency (ms)")
    ax1.set_ylabel("Empirical CDF (N = 9,920 Test Queries)")
    ax1.set_title("(a) Empirical Per-Query Latency Distribution")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower right")

    comp_labels = ["DoS Guard\n(All Queries)", "Stage-1 CNN-BiLSTM\n(INT8 Fast Path)", "Stage-2 AST+Attn\n(When Escalated)", "Full Cascade\n(Amortized Mean)"]
    cb = rq2["cascade_component_breakdown_mean_ms"]
    comp_vals = [
        cb["dos_guard_mean_ms"],
        cb["stage1_onnx_int8_mean_ms"],
        cb["stage2_when_escalated_mean_ms"],
        rq2["latency_comparison_ms"]["cascade_sql"]["mean"],
    ]
    bars = ax2.bar(comp_labels, comp_vals, color=["#2ca25f", "#3182bd", "#756bb1", "#08519c"], width=0.55)
    for b, v in zip(bars, comp_vals):
        ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.04, f"{v:.4f}" if v < 0.01 else f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylim(0, max(comp_vals) * 1.22)
    ax2.set_ylabel("Mean Execution Time (ms)")
    ax2.set_title("(b) Stage-Wise Latency Decomposition")
    ax2.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    f1_path = FIG_DIR / "fig1_latency_cdf.pdf"
    fig.savefig(f1_path, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Adversarial Obfuscation Detection Recall by Strategy ---
    strat_dict = rq1["adversarial_strategy_recall_comparison"]
    strat_names = sorted(strat_dict.keys())
    pretty_names = [s.replace("_", "\n") for s in strat_names]
    cas_rec = [strat_dict[s]["cascade_recall"] * 100.0 for s in strat_names]
    crs_rec = [strat_dict[s]["crs_v4_recall"] * 100.0 for s in strat_names]

    x = np.arange(len(strat_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    b1 = ax.bar(x - width / 2, crs_rec, width, label="OWASP ModSecurity CRS v4", color="#e66101")
    b2 = ax.bar(x + width / 2, cas_rec, width, label="CASCADE-SQL (ONNX INT8, Ours)", color="#1b7837")
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=7)
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_names)
    ax.set_ylim(85.0, 102.5)
    ax.set_ylabel("Detection Recall (%)")
    ax.set_title("Detection Recall Across 7 SQLMap / WAF-A-MoLE Obfuscation Strategies (N = 6,152)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    f2_path = FIG_DIR / "fig2_adversarial_evasion.pdf"
    fig.savefig(f2_path, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Cloud Run FinOps Cost vs. False Blocks Trade-off ---
    fin = rq3["finops_per_1m_requests_gcp_tier1"]
    arch_names = [
        "OWASP CRS v4\n(60-Rule Regex)",
        "Monolithic Ensemble\n(PyTorch FP32)",
        "Un-cascaded Stage-2\n(ONNX INT8)",
        "CASCADE-SQL\n(ONNX INT8, Ours)",
    ]
    costs = [
        fin["owasp_crs_v4_regex"]["compute_cost_usd_per_1m"],
        fin["monolithic_pytorch_fp32_parallel"]["compute_cost_usd_per_1m"],
        fin["uncascaded_stage2_onnx_int8"]["compute_cost_usd_per_1m"],
        fin["cascade_sql_onnx_int8"]["compute_cost_usd_per_1m"],
    ]
    false_blocks = [
        rq3["operational_impact_per_1m_benign"]["owasp_crs_v4_false_blocked_requests"],
        rq3["operational_impact_per_1m_benign"]["cascade_sql_false_blocked_requests"],
        rq3["uncascaded_stage2_exact_test_metrics"]["false_blocks_per_1m_benign"],
        rq3["operational_impact_per_1m_benign"]["cascade_sql_false_blocked_requests"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    c_bars = ax1.bar(arch_names, costs, color=["#d95f02", "#7570b3", "#66a61e", "#08519c"], width=0.52)
    for b, v in zip(c_bars, costs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.002, f"${v:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_ylim(0, max(costs) * 1.2)
    ax1.set_ylabel("Incremental Compute Cost per 1M Req ($USD)")
    ax1.set_title("(a) Cloud Run Compute Overhead per 1M Queries")
    ax1.grid(axis="y", alpha=0.25)

    fb_bars = ax2.bar(arch_names, false_blocks, color=["#cb181d", "#7570b3", "#66a61e", "#08519c"], width=0.52)
    ax2.set_yscale("log")
    for b, v in zip(fb_bars, false_blocks):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:,}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylim(100, max(false_blocks) * 4.0)
    ax2.set_ylabel("False-Blocked Legitimate Queries / 1M (Log Scale)")
    ax2.set_title("(b) Operational False-Positive Impact per 1M Benign")
    ax2.grid(axis="y", alpha=0.25, which="both")
    plt.tight_layout()
    f3_path = FIG_DIR / "fig3_finops_pareto.pdf"
    fig.savefig(f3_path, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: SHAP Top-10 AST Feature Attributions & DoS Resilience ---
    top10 = rq5["shap_verification"]["top_10_structural_features"][::-1]
    feat_labels = [item["feature_name"] for item in top10]
    feat_vals = [item["mean_abs_shap"] for item in top10]

    dos_list = rq5["algorithmic_dos_resilience"]
    dos_names = [d["attack_profile"].replace(" ", "\n", 1) for d in dos_list]
    ung_ms = [d["unguarded_ast_mean_ms"] for d in dos_list]
    grd_ms = [d["cascade_guarded_mean_ms"] for d in dos_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    ax1.barh(feat_labels, feat_vals, color="#2b8cbe")
    for idx, v in enumerate(feat_vals):
        ax1.text(v + 0.008, idx, f"{v:.4f}", va="center", fontsize=7.5)
    ax1.set_xlim(0, max(feat_vals) * 1.22)
    ax1.set_xlabel("Mean |SHAP Attribution| (Stage-2 ONNX INT8)")
    ax1.set_title("(a) Top-10 Structural AST Feature Attributions")
    ax1.grid(axis="x", alpha=0.25)

    xd = np.arange(len(dos_names))
    w = 0.35
    ax2.bar(xd - w / 2, ung_ms, w, label="Unguarded AST + Stage-2", color="#d7301f")
    ax2.bar(xd + w / 2, grd_ms, w, label="CASCADE-SQL Pre-Parser Guard", color="#238b45")
    ax2.set_yscale("log")
    ax2.set_xticks(xd)
    ax2.set_xticklabels(dos_names, fontsize=7.5)
    ax2.set_ylim(1e-4, 400.0)
    ax2.set_ylabel("Mean Processing Latency in ms (Log Scale)")
    ax2.set_title("(b) Algorithmic Complexity DoS Mitigation")
    ax2.grid(axis="y", alpha=0.25, which="both")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    f4_path = FIG_DIR / "fig4_shap_dos_resilience.pdf"
    fig.savefig(f4_path, bbox_inches="tight")
    plt.close(fig)

    out_summary = {
        p.name: {"size_bytes": p.stat().st_size, "sha256": sha256_of(p)}
        for p in sorted(FIG_DIR.glob("*.pdf"))
    }
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()