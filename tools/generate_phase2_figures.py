import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
})

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "05_results_and_paper" / "raw_logs"
FIGS = ROOT / "05_results_and_paper" / "figures_pdf"
FIGS.mkdir(parents=True, exist_ok=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_box(ax, xy, w, h, title, subtitle, fc, ec="#1f2937", lw=1.2):
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2.0, xy[1] + h / 2.0
    ax.text(cx, cy + h * 0.14, title, ha="center", va="center", fontweight="bold", fontsize=8.2, color="#111827")
    ax.text(cx, cy - h * 0.18, subtitle, ha="center", va="center", fontsize=7.1, color="#374151")


def main() -> None:
    exp1 = json.loads((LOGS / "exp1_sota_baselines_audit.json").read_text(encoding="utf-8"))
    exp2 = json.loads((LOGS / "exp2_padding_bypass_audit.json").read_text(encoding="utf-8"))
    exp3 = json.loads((LOGS / "exp3_adaptive_wafamole_audit.json").read_text(encoding="utf-8"))
    exp4 = json.loads((LOGS / "exp4_live_proxy_concurrency_audit.json").read_text(encoding="utf-8"))
    surf = np.load(LOGS / "exp6_threshold_surface.npz")

    # --- fig0_system_architecture.pdf: Vector System & Threat Boundary Diagram ---
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    trust_zone = mpatches.FancyBboxPatch(
        (0.19, 0.08), 0.61, 0.84, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#f8fafc", edgecolor="#0284c7", linestyle="--", linewidth=1.3
    )
    ax.add_patch(trust_zone)
    ax.text(0.495, 0.89, "Serverless Security Boundary: CASCADE-SQL Inline ONNX INT8 Proxy (1 vCPU / 256 MiB)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0369a1")

    draw_box(ax, (0.01, 0.36), 0.15, 0.28, "Untrusted Clients\n& Attackers", "T1: Dialect Obfusc.\nT2: WAF-A-MoLE\nT3: 64KB/Paren DoS", "#fee2e2", "#b91c1c")
    draw_box(ax, (0.21, 0.36), 0.14, 0.28, "Pre-Parser\nDoS Guard", "L_max <= 8192 B\nDepth <= 32\n0.0015 ms O(1)", "#dcfce7", "#15803d")
    draw_box(ax, (0.39, 0.36), 0.18, 0.28, "Stage-1 Strided\nCNN-BiLSTM (INT8)", "W=256, S=192 Pool\n145K Params\nFast Exit ~99.8%", "#dbeafe", "#1d4ed8")
    draw_box(ax, (0.61, 0.53), 0.17, 0.28, "Stage-2 Canonical\nAST + Self-Attn", "sqlglot De-obfusc.\n24 AST Feats + 2L Attn\nGray-Zone Verifier", "#ede9fe", "#6d28d9")
    draw_box(ax, (0.61, 0.14), 0.17, 0.24, "HTTP 403 Block\n& OTel Audit", "Blocked SQLi / DoS\nSHAP Attribution", "#fef3c7", "#b45309")
    draw_box(ax, (0.83, 0.36), 0.16, 0.28, "Cloud DBaaS\nBackends", "PostgreSQL 16\nMySQL 8.0\nVerified Benign SQL", "#ecfccb", "#3f6212")

    arrow_kw = dict(arrowstyle="->,head_width=0.25,head_length=0.35", lw=1.3, color="#1f2937")
    ax.annotate("", xy=(0.21, 0.50), xytext=(0.16, 0.50), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.39, 0.50), xytext=(0.35, 0.50), arrowprops=arrow_kw)
    ax.annotate("0.05 < p1 < 0.65", xy=(0.61, 0.67), xytext=(0.57, 0.56), fontsize=7.0, arrowprops=arrow_kw)
    ax.annotate("p1 >= 0.65", xy=(0.61, 0.26), xytext=(0.57, 0.43), fontsize=7.0, arrowprops=dict(arrowstyle="->", lw=1.2, color="#b91c1c"))
    ax.annotate("DoS", xy=(0.61, 0.20), xytext=(0.28, 0.36), fontsize=7.0, arrowprops=dict(arrowstyle="->", lw=1.1, ls=":", color="#b91c1c"))
    ax.annotate("p1 <= 0.05 or p2 < 0.5", xy=(0.83, 0.50), xytext=(0.57, 0.49), fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=1.3, color="#15803d"))
    ax.annotate("", xy=(0.83, 0.56), xytext=(0.78, 0.65), arrowprops=dict(arrowstyle="->", lw=1.2, color="#15803d"))
    ax.annotate("p2 >= 0.5", xy=(0.695, 0.38), xytext=(0.695, 0.53), fontsize=7.0, ha="center", arrowprops=dict(arrowstyle="->", lw=1.2, color="#b91c1c"))

    plt.tight_layout()
    fig.savefig(FIGS / "fig0_system_architecture.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- fig5_sota_and_padding.pdf: SOTA Latency-Recall Pareto + Prefix-Padding Defense ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    sota = exp1["sota_comparison_results"]
    markers = ["s", "D", "^", "v", "P", "*"]
    colors = ["#d95f02", "#e7298a", "#7570b3", "#66a61e", "#e6ab02", "#08519c"]
    for (name, d), mk, col in zip(sota.items(), markers, colors):
        lat = d["single_vcpu_latency_ms"]["mean"]
        adv_r = d["adversarial_obfuscated_recall"] * 100.0
        sz = 150 if "CASCADE" in name else 75
        short_lbl = name.split(" (")[0]
        ax1.scatter(lat, adv_r, s=sz, marker=mk, color=col, label=f"{short_lbl} (F1={d['held_out_test']['f1_score']:.4f})", zorder=4)
    ax1.set_xscale("log")
    ax1.set_xlabel("Single-vCPU Mean Latency in ms (Log Scale — Lower is Better)")
    ax1.set_ylabel("Adversarial Obfuscated Recall (%)")
    ax1.set_ylim(54, 102)
    ax1.set_title("(a) SOTA Accuracy-Latency Frontier (Exp-1)")
    ax1.grid(alpha=0.28, which="both")
    ax1.legend(loc="lower right", fontsize=6.8)

    pad_list = exp2["prefix_padding_tail_injection_results"]
    px = [r["benign_prefix_bytes"] for r in pad_list]
    tr_rec = [r["naive_truncated_recall"] * 100.0 for r in pad_list]
    st_rec = [r["strided_cascade_recall"] * 100.0 for r in pad_list]
    ax2.plot(px, tr_rec, "o--", color="#cb181d", lw=1.7, label="Naive 256-Byte Truncation")
    ax2.plot(px, st_rec, "s-", color="#08519c", lw=2.0, label="Strided Multi-Window CASCADE-SQL (Ours)")
    ax2.set_xlabel("Benign SQL Prefix Padding Length (Bytes)")
    ax2.set_ylabel("Tail-Injected SQLi Recall (%)")
    ax2.set_ylim(-5, 106)
    ax2.set_title("(b) Prefix-Padding Tail-Injection Defense (Exp-2)")
    ax2.grid(alpha=0.28)
    ax2.legend(loc="center right")
    plt.tight_layout()
    fig.savefig(FIGS / "fig5_sota_and_padding.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- fig6_wafamole_and_concurrency.pdf: Iterative WAF-A-MoLE + Live Concurrency Spans ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    waf_b = exp3["budget_progression"]
    bx = [r["query_budget_B"] for r in waf_b]
    s1_r = [r["stage1_only_recall"] * 100.0 for r in waf_b]
    cas_r = [r["cascade_recall"] * 100.0 for r in waf_b]
    esc_r = [r["cascade_escalation_rate_pct"] for r in waf_b]

    ax1.plot(bx, cas_r, "o-", color="#08519c", lw=2.0, label="Full CASCADE-SQL Recall (%)")
    ax1.plot(bx, s1_r, "s--", color="#d95f02", lw=1.7, label="Stage-1 Only Recall (%)")
    ax1.plot(bx, esc_r, "^:", color="#6a51a3", lw=1.5, label="Dynamic Stage-2 Escalation rho (%)")
    ax1.set_xlabel("Grey-Box WAF-A-MoLE Query Budget (B Queries / Payload)")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("(a) Adaptive WAF-A-MoLE Resilience (Exp-3, p=1.2e-12)")
    ax1.grid(alpha=0.28)
    ax1.legend(loc="center right", fontsize=7.2)

    tiers = exp4["concurrency_tiers"]
    clbls = [f"C={t['concurrency']}\n({t['sustained_rps']:.0f} RPS)" for t in tiers]
    ai_sp = [t["otel_span_breakdown_mean_ms"]["cascade_ai_inspection_ms"] for t in tiers]
    db_sp = [t["otel_span_breakdown_mean_ms"]["relational_db_execution_ms"] for t in tiers]
    net_sp = [t["otel_span_breakdown_mean_ms"]["network_and_http_serialization_ms"] for t in tiers]

    xc = np.arange(len(clbls))
    ax2.bar(xc, ai_sp, width=0.5, label="CASCADE-SQL AI Span (ms)", color="#08519c")
    ax2.bar(xc, db_sp, width=0.5, bottom=ai_sp, label="Relational SQL DB Span (ms)", color="#2ca25f")
    ax2.bar(xc, net_sp, width=0.5, bottom=np.array(ai_sp) + np.array(db_sp), label="HTTP/ASGI Queue + Net (ms)", color="#9ecae1")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(clbls)
    ax2.set_ylabel("Mean End-to-End Latency (ms)")
    ax2.set_title("(b) Live HTTP Proxy OpenTelemetry Spans (Exp-4)")
    ax2.grid(axis="y", alpha=0.28)
    ax2.legend(loc="upper left", fontsize=7.2)
    plt.tight_layout()
    fig.savefig(FIGS / "fig6_wafamole_and_concurrency.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- fig7_cv_sensitivity_heatmap.pdf: 5-Fold Out-of-Fold 2D Sensitivity Surface ---
    tl_g = surf["tau_low_grid"]
    th_g = surf["tau_high_grid"]
    f1_s = surf["f1_surface"]
    rho_s = surf["rho_pct_surface"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    im1 = ax1.imshow(f1_s, origin="lower", aspect="auto", cmap="YlGnBu",
                     extent=[th_g[0], th_g[-1], tl_g[0], tl_g[-1]])
    ax1.scatter([0.65], [0.05], color="#cb181d", marker="*", s=130, label="Selected (0.05, 0.65)")
    ax1.set_xlabel("Upper Threshold tau_high")
    ax1.set_ylabel("Lower Threshold tau_low")
    ax1.set_title("(a) 5-Fold OOF F1-Score Surface (N=50,460)")
    ax1.legend(loc="upper left", fontsize=7.2)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(rho_s, origin="lower", aspect="auto", cmap="magma_r",
                     extent=[th_g[0], th_g[-1], tl_g[0], tl_g[-1]])
    ax2.scatter([0.65], [0.05], color="#00ffcc", marker="*", s=130, label="Selected (rho=0.17%)")
    ax2.set_xlabel("Upper Threshold tau_high")
    ax2.set_ylabel("Lower Threshold tau_low")
    ax2.set_title("(b) Stage-2 Escalation Rate rho (%) Surface")
    ax2.legend(loc="upper left", fontsize=7.2)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(FIGS / "fig7_cv_sensitivity_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    out = {p.name: {"size_bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in sorted(FIGS.glob("*.pdf"))}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()