import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Unified Academic Publication Palette (matching reference architecture style)
PALETTE = {
    "slate_dark": "#1E293B",
    "slate_border": "#64748B",
    "slate_fill": "#F1F5F9",
    "slate_header": "#E2E8F0",
    "blue_border": "#2563EB",
    "blue_fill": "#EFF6FF",
    "teal_border": "#0D9488",
    "teal_fill": "#CCFBF1",
    "teal_bg": "#E6FFFA",
    "amber_border": "#D97706",
    "amber_fill": "#FFFBEB",
    "red_accent": "#DC2626",
}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.edgecolor": "#475569",
    "axes.linewidth": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
})

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "05_results_and_paper" / "raw_logs"
FIGS = ROOT / "05_results_and_paper" / "figures_pdf"
FIGS.mkdir(parents=True, exist_ok=True)


def rbox(ax, x, y, w, h, fc="#FFFFFF", ec="#64748B", lw=1.3, ls="-", pad=0.008, r=0.014, zorder=2):
    patch = mpatches.FancyBboxPatch(
        (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def build_fig0_reference_style() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # === LEFT COLUMN: PHASE LABELS ===
    phases = [
        (0.76, 0.21, "PHASE 1 & 2:\nZERO-LEAKAGE\nCORPUS &\nOFFLINE ONNX\nTRAINING\n(RTX 2050 GPU)"),
        (0.55, 0.18, "PRE-PARSER\nBOUNDED DoS\n& TRAFFIC\nISOLATION\nLAYER"),
        (0.28, 0.24, "PHASE 3:\nCASCADED\nONNX PROXY\nCOMPUTE LAYER\n(1 vCPU / 256 MiB)"),
        (0.03, 0.22, "PHASE 3:\nMANAGED CLOUD\nRELATIONAL\nDBaaS LAYER\n(Verified SQL)"),
    ]
    for py, ph, ptxt in phases:
        ax.text(0.062, py + ph / 2.0, ptxt, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=PALETTE["slate_dark"], linespacing=1.22)

    # === TIER 1 (TOP): OFFLINE TRAINING & QUANTIZATION ===
    rbox(ax, 0.125, 0.76, 0.675, 0.215, fc="#F8FAFC", ec=PALETTE["slate_border"], lw=1.5)
    ax.text(0.145, 0.945, "ZERO-LEAKAGE SQL CORPUS & OFFLINE INT8 QUANTIZATION PIPELINE",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=PALETTE["slate_dark"])

    t1_boxes = [
        (0.14, 0.145, "Canonical AST\nGroupSplit\n(0% Leakage)", PALETTE["slate_fill"], PALETTE["slate_border"]),
        (0.295, 0.175, "Stage-1 Fast Filter:\nChar CNN-BiLSTM\n(145,249 Params)", PALETTE["blue_fill"], PALETTE["blue_border"]),
        (0.48, 0.155, "Stage-2 Verifier:\nsqlglot AST + Attn\n(106,177 Params)", "#FFFFFF", PALETTE["slate_border"]),
        (0.645, 0.14, "Telemetry & Audit:\nOpenTelemetry,\nSHA-256 Logs", "#FFFFFF", PALETTE["slate_border"]),
    ]
    for bx, bw, btxt, bfc, bec in t1_boxes:
        rbox(ax, bx, 0.815, bw, 0.105, fc=bfc, ec=bec, lw=1.3)
        ax.text(bx + bw / 2.0, 0.867, btxt, ha="center", va="center", fontsize=7.8, fontweight="bold", color=PALETTE["slate_dark"])

    ax.text(0.462, 0.785,
            "Multi-Suite Scale: 50,460 Deduplicated Queries -> 42,667 Templates | Thresholds: tau_low* = 0.05, tau_high* = 0.65 (721.8 KB INT8)",
            ha="center", va="center", fontsize=7.9, fontweight="bold", color=PALETTE["slate_dark"])

    # === TIER 2: PRE-PARSER DoS GUARD & ISOLATION FABRIC ===
    rbox(ax, 0.125, 0.635, 0.675, 0.085, fc=PALETTE["slate_header"], ec=PALETTE["slate_dark"], lw=1.5)
    ax.text(0.462, 0.688, "Pre-Parser Bounded DoS Guard Fabric (L_max <= 8,192 Bytes | Depth <= 32)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=PALETTE["slate_dark"])
    ax.text(0.462, 0.655, "O(1) Byte-Length & O(L_max) Parenthesis Nesting Filter (0.0015 ms — 69,852x DoS Speedup)",
            ha="center", va="center", fontsize=7.8, color="#334155")

    ax.text(0.285, 0.585, "CLIENT / API SQL STREAM\n(Benign ORM, Obfuscated SQLi, WAF-A-MoLE)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=PALETTE["slate_dark"])
    ax.text(0.635, 0.585, "GRAY-ZONE ESCALATION PATH\n(Uncertain 0.05 < p1 < 0.65 or Multi-Window)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0F766E")

    # === TIER 3: 4 PROXY MODULES ===
    rbox(ax, 0.12, 0.275, 0.685, 0.265, fc="#F8FAFC", ec="#CBD5E1", lw=1.0, pad=0.003)
    m_specs = [
        (0.132, 0.158, "Module 1:\nStrided Window", "W=256, S=192\n100% Byte Cover", "Tail-Pad Safe", "#FFFFFF", PALETTE["slate_border"], "--"),
        (0.298, 0.158, "Module 2:\nStage-1 INT8", "CNN-BiLSTM\nFast Exit ~99.8%", "0.630 ms Mean", PALETTE["blue_fill"], PALETTE["blue_border"], "-"),
        (0.464, 0.158, "Module 3:\nCascade Router", "tau_low* = 0.05\ntau_high* = 0.65", "HTTP 403 Block", "#FFFFFF", PALETTE["slate_border"], "--"),
        (0.630, 0.158, "Module 4:\nStage-2 INT8", "Comment Unwrap\n24 AST + 4H Attn", "99.63% Adv. Hit", PALETTE["teal_fill"], PALETTE["teal_border"], "-"),
    ]
    for mx, mw, mtitle, mbody, mfoot, mfc, mec, mls in m_specs:
        rbox(ax, mx, 0.29, mw, 0.235, fc=mfc, ec=mec, lw=1.4, ls=mls)
        ax.text(mx + mw / 2.0, 0.478, mtitle, ha="center", va="center", fontsize=8.1, fontweight="bold", color=PALETTE["slate_dark"])
        ax.text(mx + mw / 2.0, 0.402, mbody, ha="center", va="center", fontsize=7.4, color="#334155")
        ax.plot([mx + 0.015, mx + mw - 0.015], [0.352, 0.352], color=mec, lw=0.9, ls=mls)
        ax.text(mx + mw / 2.0, 0.322, mfoot, ha="center", va="center", fontsize=7.6, fontweight="bold", color=PALETTE["slate_dark"])

    # === TIER 4 (BOTTOM): MANAGED CLOUD DBaaS FABRIC ===
    rbox(ax, 0.125, 0.03, 0.675, 0.215, fc=PALETTE["teal_bg"], ec=PALETTE["teal_border"], lw=1.5)
    ax.text(0.462, 0.215, "Managed Cloud Relational DBaaS Fabric (Verified Benign SQL Only)",
            ha="center", va="center", fontsize=9.3, fontweight="bold", color=PALETTE["slate_dark"])

    db_nodes = [
        (0.145, 0.135, "PostgreSQL 16\n(JSONB / CTE)"),
        (0.300, 0.135, "MySQL 8.0\n(InnoDB)"),
        (0.485, 0.145, "Live SQL Pool\n(10,000 Rows)"),
        (0.650, 0.135, "OTel Span\nTelemetry"),
    ]
    for dx, dw, dtxt in db_nodes:
        rbox(ax, dx, 0.105, dw, 0.082, fc=PALETTE["teal_fill"], ec=PALETTE["slate_dark"], lw=1.2)
        ax.text(dx + dw / 2.0, 0.146, dtxt, ha="center", va="center", fontsize=7.7, fontweight="bold", color=PALETTE["slate_dark"])

    ax.text(0.462, 0.062, "Zero False Blocks on 2,000 Hard-Benign Modern ORM Queries (0.00% FPR | 549.0 RPS @ 0% Error)",
            ha="center", va="center", fontsize=8.3, fontweight="bold", color=PALETTE["slate_dark"])

    # === RIGHT SIDEBAR: EXPERIMENTAL CAMPAIGN ===
    rbox(ax, 0.815, 0.03, 0.175, 0.945, fc="#F8FAFC", ec=PALETTE["slate_border"], lw=1.5)
    ax.text(0.9025, 0.915, "EXPERIMENTAL\nCAMPAIGN\n(PHASE 4 & 5)",
            ha="center", va="center", fontsize=9.3, fontweight="bold", color=PALETTE["slate_dark"])

    rbox(ax, 0.827, 0.165, 0.151, 0.685, fc=PALETTE["amber_fill"], ec=PALETTE["amber_border"], lw=1.5)
    ax.text(0.9025, 0.795, "ZERO-ESTIMATION\nEVALUATION\n(RQ1-RQ5 & EXP 1-6)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=PALETTE["slate_dark"])

    camp_items = [
        (0.685, "1. Detection & SOTA", "(0.9991 F1, 6 Baselines,\n99.63% Obfusc. Recall)"),
        (0.555, "2. Latency Profiles", "(0.636 ms Mean, p99 Tail,\nLive C=1..100 Load)"),
        (0.420, "3. Cloud Run FinOps", "($0.0157 / 1M Queries,\n-81.1% vs. PyTorch)"),
        (0.275, "4. Adversarial & XAI", "(40K WAF-A-MoLE,\n5-Fold CV, SHAP, DoS)"),
    ]
    for cy, chead, csub in camp_items:
        ax.text(0.838, cy, chead, ha="left", va="center", fontsize=7.7, fontweight="bold", color=PALETTE["slate_dark"])
        ax.text(0.838, cy - 0.048, csub, ha="left", va="center", fontsize=7.0, color="#334155")

    ax.text(0.9025, 0.095, "SHA-256 Logs ->\n13 Tables & 8 Figs ->\nComputers & Security",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=PALETTE["slate_dark"])

    # === ORTHOGONAL ARROWS ===
    akw = dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", lw=1.3, color="#334155")
    ax.annotate("", xy=(0.462, 0.720), xytext=(0.462, 0.760), arrowprops=akw)
    ax.annotate("", xy=(0.211, 0.525), xytext=(0.211, 0.635), arrowprops=akw)
    ax.annotate("", xy=(0.377, 0.525), xytext=(0.377, 0.635), arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", lw=1.3, color=PALETTE["blue_border"]))
    ax.annotate("", xy=(0.543, 0.525), xytext=(0.543, 0.635), arrowprops=akw)
    ax.annotate("", xy=(0.709, 0.525), xytext=(0.709, 0.635), arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", lw=1.3, color=PALETTE["teal_border"]))

    ax.annotate("", xy=(0.377, 0.245), xytext=(0.377, 0.290), arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", lw=1.3, color=PALETTE["blue_border"]))
    ax.annotate("", xy=(0.709, 0.245), xytext=(0.709, 0.290), arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", lw=1.3, color=PALETTE["teal_border"]))
    ax.annotate("", xy=(0.815, 0.675), xytext=(0.800, 0.675), arrowprops=akw)
    ax.annotate("", xy=(0.815, 0.140), xytext=(0.800, 0.140), arrowprops=akw)

    out_pdf = FIGS / "fig0_system_architecture.pdf"
    plt.tight_layout(pad=0.2)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    p = build_fig0_reference_style()
    print(json.dumps({"updated_figure": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_of(p)}, indent=2))