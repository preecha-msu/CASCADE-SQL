import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))

from cascade_onnx_engine import CascadeSQLOnnxEngine

SEED = 42
TEST_CSV = ROOT / "01_datasets" / "processed" / "test.csv"
ROUTER_CFG = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "exp3_adaptive_wafamole_audit.json"
TAB9_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab9_exp3_adaptive_wafamole.tex"


def log_step(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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


def mcnemar_test(y_true: np.ndarray, pred_cascade: np.ndarray, pred_s1_only: np.ndarray) -> dict:
    ca = (pred_cascade == y_true)
    cb = (pred_s1_only == y_true)
    b = int((ca & (~cb)).sum())  # Cascade blocked, Stage-1 bypassed
    c = int(((~ca) & cb).sum())  # Cascade bypassed, Stage-1 blocked
    if b + c == 0:
        return {"b_cascade_saved": b, "c_s1_saved": c, "chi2": 0.0, "p_value": 1.0}
    if b + c < 25:
        pv = float(min(1.0, stats.binomtest(min(b, c), b + c, 0.5).pvalue))
        return {"b_cascade_saved": b, "c_s1_saved": c, "chi2": None, "p_value": pv}
    chi2 = float(((abs(b - c) - 1.0) ** 2) / float(b + c))
    pv = float(stats.chi2.sf(chi2, df=1))
    return {"b_cascade_saved": b, "c_s1_saved": c, "chi2": round(chi2, 4), "p_value": pv}


_KEYWORDS_RE = re.compile(r"\b(SELECT|UNION|ALL|FROM|WHERE|AND|OR|SLEEP|BENCHMARK|INFORMATION_SCHEMA)\b", re.IGNORECASE)


def apply_wafamole_mutation(sql: str, dialect: str, rng: random.Random) -> str:
    op = rng.randint(0, 8)
    out = sql
    if op == 0:
        matches = list(re.finditer(r"\s+", out))
        if matches:
            m = rng.choice(matches)
            fill = rng.choice(["/**/", "/*_*/", "/*x*/", "\t", "\n"])
            out = out[:m.start()] + fill + out[m.end():]
    elif op == 1:
        matches = list(_KEYWORDS_RE.finditer(out))
        if matches:
            m = rng.choice(matches)
            kw = m.group(0)
            mixed = "".join(ch.upper() if rng.random() > 0.5 else ch.lower() for ch in kw)
            if dialect == "mysql":
                rep = f"/*!50000{mixed}*/"
            else:
                rep = f"/*pg*/{mixed}/*pg*/"
            out = out[:m.start()] + rep + out[m.end():]
    elif op == 2:
        tag = rng.choice(["/*cache_ok*/", "/*idx_scan*/", "/*read_only*/", "/*v2_api*/"])
        pos = rng.randint(0, max(0, min(len(out), 96)))
        space_pos = out.find(" ", pos)
        if space_pos != -1:
            out = out[:space_pos] + f" {tag} " + out[space_pos + 1:]
        else:
            out = f"{tag} " + out
    elif op == 3:
        num_matches = list(re.finditer(r"\b(\d+)\b", out))
        if num_matches:
            m = rng.choice(num_matches)
            val = m.group(1)
            rep = f"0{val}" if rng.random() < 0.6 else f"{val}e0"
            out = out[:m.start()] + rep + out[m.end():]
    elif op == 4:
        eq_matches = list(re.finditer(r"=", out))
        if eq_matches:
            m = rng.choice(eq_matches)
            out = out[:m.start()] + " LIKE " + out[m.end():]
    elif op == 5:
        chars = list(out)
        alpha_idx = [i for i, ch in enumerate(chars) if ch.isalpha()]
        for i in rng.sample(alpha_idx, k=min(len(alpha_idx), 6)):
            chars[i] = chars[i].swapcase()
        out = "".join(chars)
    elif op == 6:
        if len(out) < 700:
            hint = rng.choice([
                "/* route=checkout_service status=healthy */ ",
                "/* orm=sqlalchemy pool=read_replica */ ",
                "/* trace_id=98a7f12b_prod_select */ ",
            ])
            out = hint + out
    elif op == 7:
        # MySQL versioned inline keyword split or logical neutral wrapper
        matches = list(_KEYWORDS_RE.finditer(out))
        if matches and dialect == "mysql":
            m = rng.choice(matches)
            kw = m.group(0)
            if len(kw) >= 4:
                mid = len(kw) // 2
                rep = f"/*!50000{kw[:mid]}*//*!50000{kw[mid:]}*/"
                out = out[:m.start()] + rep + out[m.end():]
    else:
        out = out.replace(" ", "\t")

    return out[:1024]


def main() -> None:
    t0 = time.perf_counter_ns()
    rng = random.Random(SEED)

    cfg = json.loads(ROUTER_CFG.read_text(encoding="utf-8"))
    engine = CascadeSQLOnnxEngine(
        cfg["tau_low"], cfg["tau_high"], cfg["tau_stage2"], window_mode="strided_multi_window"
    )

    df_test = pd.read_csv(TEST_CSV, low_memory=False)
    df_sqli = df_test[df_test["label"] == 1].iloc[:400].reset_index(drop=True)
    n_samples = len(df_sqli)

    # Exact multiples of children_per_round=4 -> zero KeyError risk
    children_per_round = 4
    budgets = [0, 12, 24, 48, 100]
    max_budget = max(budgets)
    assert all(b % children_per_round == 0 for b in budgets), "All budgets must be exact multiples of children_per_round!"

    current_best_q = df_sqli["query"].astype(str).tolist()
    dialects = df_sqli["dialect"].astype(str).tolist()

    current_best_p1 = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        res0 = engine.inspect_single_query_timed(current_best_q[i], dialects[i])
        current_best_p1[i] = res0["p1"]

    snapshots: dict[int, list[str]] = {0: list(current_best_q)}

    log_step(f"[1/2] Executing Guided WAF-A-MoLE evolutionary search on {n_samples} SQLi payloads (budgets={budgets})...")
    queries_used = 0
    total_rounds = max_budget // children_per_round

    for _ in range(1, total_rounds + 1):
        cand_queries = []
        cand_map = []
        for s_idx in range(n_samples):
            base_q = current_best_q[s_idx]
            d_str = dialects[s_idx]
            for _ in range(children_per_round):
                cand_queries.append(apply_wafamole_mutation(base_q, d_str, rng))
                cand_map.append(s_idx)

        cand_p1 = engine.predict_stage1_probs_batch(cand_queries, batch_size=512)
        for c_q, c_p1, s_idx in zip(cand_queries, cand_p1, cand_map):
            if float(c_p1) < current_best_p1[s_idx]:
                current_best_p1[s_idx] = float(c_p1)
                current_best_q[s_idx] = c_q

        queries_used += children_per_round
        if queries_used in budgets:
            snapshots[queries_used] = list(current_best_q)
            log_step(f"   -> Reached query budget B={queries_used:3d}: Mean Stage-1 score depressed to {np.mean(current_best_p1):.4f}")

    assert set(budgets).issubset(set(snapshots.keys())), "Missing budget snapshot!"

    log_step("[2/2] Evaluating Stage-1-Only vs. Full Hardened CASCADE-SQL across all attack budgets...")
    y_ones = np.ones(n_samples, dtype=np.int64)
    budget_reports = []

    for B in budgets:
        evolved_q = snapshots[B]
        pred_s1_only = np.zeros(n_samples, dtype=np.int64)
        pred_cascade = np.zeros(n_samples, dtype=np.int64)
        p1_vals = np.zeros(n_samples, dtype=np.float64)
        escalated_cnt = 0
        lats = []

        for i in range(n_samples):
            out = engine.inspect_single_query_timed(evolved_q[i], dialects[i])
            p1_vals[i] = out["p1"]
            pred_s1_only[i] = 1 if out["p1"] >= 0.50 else 0
            pred_cascade[i] = out["pred"]
            if out["route"] == "escalated_stage2":
                escalated_cnt += 1
            lats.append(out["total_ms"])

        s1_recall = float(np.mean(pred_s1_only))
        s1_asr = 1.0 - s1_recall
        cas_recall = float(np.mean(pred_cascade))
        cas_asr = 1.0 - cas_recall
        esc_rate = (escalated_cnt / float(n_samples)) * 100.0
        mc = mcnemar_test(y_ones, pred_cascade, pred_s1_only)

        budget_reports.append({
            "query_budget_B": B,
            "samples": n_samples,
            "mean_depressed_p1_score": round(float(np.mean(p1_vals)), 4),
            "stage1_only_recall": round(s1_recall, 6),
            "stage1_only_asr_pct": round(s1_asr * 100.0, 2),
            "cascade_escalation_rate_pct": round(esc_rate, 2),
            "cascade_recall": round(cas_recall, 6),
            "cascade_asr_pct": round(cas_asr * 100.0, 2),
            "cascade_mean_latency_ms": round(float(np.mean(lats)), 4),
            "mcnemar_vs_stage1_only": mc,
        })

    def fmt_pv(p: float) -> str:
        return "< 0.0001" if p < 1e-4 else f"{p:.4f}"

    tex_rows = []
    for r in budget_reports:
        pv_str = fmt_pv(r["mcnemar_vs_stage1_only"]["p_value"])
        tex_rows.append(
            f"$B = {r['query_budget_B']}$ & {r['mean_depressed_p1_score']:.4f} & "
            f"{r['stage1_only_recall']*100:.2f}\\% & {r['stage1_only_asr_pct']:.2f}\\% & "
            f"{r['cascade_escalation_rate_pct']:.2f}\\% & \\textbf{{{r['cascade_recall']*100:.2f}\\%}} & "
            f"\\textbf{{{r['cascade_asr_pct']:.2f}\\%}} & {pv_str} \\\\"
        )
    joined_rows = "\n".join(tex_rows)

    tab9_tex = rf"""% Auto-generated from 100% Real Iterative Guided WAF-A-MoLE Attack Benchmark (Exp-3)
\begin{{table*}}[htbp]
\centering
\caption{{Robustness against iterative grey-box Guided WAF-A-MoLE evolutionary attacks ($N={n_samples}$ SQLi payloads, up to $B=100$ queries per payload) comparing Stage-1 CNN-BiLSTM alone against Full Hardened CASCADE-SQL. ASR denotes Attack Success Rate (evasion rate).}}
\label{{tab:exp3_adaptive_wafamole}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
\textbf{{Attack Budget}} & \textbf{{Mean $p_1(x^{{(B)}})$}} & \textbf{{Stage-1 Recall}} & \textbf{{Stage-1 ASR}} & \textbf{{Escalated ($\rho$)}} & \textbf{{CASCADE Recall}} & \textbf{{CASCADE ASR}} & \textbf{{McNemar $p$-value}} \\
\midrule
{joined_rows}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    write_utf8_lf(TAB9_TEX, tab9_tex)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_sec": round((time.perf_counter_ns() - t0) / 1e9, 3),
        "total_adversarial_queries_executed": int(n_samples * max_budget),
        "budget_progression": budget_reports,
        "sha256_verification": {
            "test_csv": sha256_of(TEST_CSV),
            "tab9_exp3_adaptive_wafamole_tex": sha256_of(TAB9_TEX),
        },
    }

    write_utf8_lf(LOG_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()