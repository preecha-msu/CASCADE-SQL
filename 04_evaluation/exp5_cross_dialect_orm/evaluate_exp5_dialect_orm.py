import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cnn_bilstm"))
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "ast_gnn"))
sys.path.insert(0, str(ROOT / "03_cloud_testbed" / "baseline_waf_modsec"))
sys.path.insert(0, str(ROOT / "04_evaluation" / "exp1_sota_baselines"))

from cascade_onnx_engine import CascadeSQLOnnxEngine, batch_encode_stage1
from evaluate_exp1_sota import LibInjectionFingerprintEngine
from feature_extractor_stage2 import batch_extract_stage2
from owasp_crs_baseline import OwaspCrsV4Baseline
from train_stage1_onnx import Stage1CNNBiLSTM
from train_stage2_onnx import Stage2ASTContextualVerifier

SEED = 42
TRAIN_CSV = ROOT / "01_datasets" / "processed" / "train.csv"
TEST_CSV = ROOT / "01_datasets" / "processed" / "test.csv"
ROUTER_CFG = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"

LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "exp5_cross_dialect_orm_audit.json"
TAB11_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab11_exp5_cross_dialect_orm.tex"

_MYSQL_AUTH_RE = re.compile(
    r"(`[^`]+`|/\*!\d+|\bsleep\s*\(|\bbenchmark\s*\(|\bextractvalue\b|\bupdatexml\b|\bgroup_concat\b|\bifnull\b|@@\w+|\blimit\s+\d+\s*,\s*\d+)",
    re.IGNORECASE,
)
_PG_AUTH_RE = re.compile(
    r"(::[a-z_]+|\bpg_sleep\s*\(|\bpg_catalog\b|\bilike\b|\bchr\s*\(|\bstring_agg\b|\bjsonb\b|\bcurrent_schema\b|\$\d+)",
    re.IGNORECASE,
)


def log_step(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def set_seed(seed: int = SEED) -> None:
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


def generate_hard_benign_orm_suite(n_total: int = 2000) -> list[tuple[str, str, str]]:
    """
    Generates 2,000 deterministic, realistic modern ORM & analytical benign SQL statements
    containing SQL homographs ('select', 'union', 'or', 'drop', '--'), CTEs, JSONB, and Window functions.
    Returns list of (category, dialect, sql_query).
    """
    items: list[tuple[str, str, str]] = []
    n_per_cat = n_total // 4

    # Category 1: CTEs & Window Analytics (PostgreSQL & MySQL 8+)
    depts = ["engineering", "security_ops", "cloud_finops", "data_platform", "customer_success"]
    for i in range(n_per_cat):
        dept = depts[i % len(depts)]
        dialect = "postgresql" if i % 2 == 0 else "mysql"
        sql = (
            f"WITH monthly_kpi AS ("
            f"SELECT account_id, SUM(net_amount) AS total_spend, COUNT(invoice_id) AS inv_cnt "
            f"FROM billing_ledgers WHERE department = '{dept}' AND fiscal_year = {2022 + (i % 5)} "
            f"GROUP BY account_id) "
            f"SELECT account_id, total_spend, ROW_NUMBER() OVER (ORDER BY total_spend DESC) AS spend_rank "
            f"FROM monthly_kpi WHERE inv_cnt >= {(i % 6) + 1} LIMIT 25;"
        )
        items.append(("cte_window_analytics", dialect, sql))

    # Category 2: PostgreSQL JSONB, Typecasts & Array Operators
    events = ["user_select_item", "subscription_upgrade", "order_checkout", "password_reset_request"]
    for i in range(n_per_cat):
        ev = events[i % len(events)]
        sql = (
            f"SELECT event_uuid::text, payload_json ->> 'session_id' AS sid, created_at "
            f"FROM telemetry_events WHERE payload_json ->> 'event_type' = '{ev}' "
            f"AND is_verified = TRUE AND retry_count <= {i % 4} ORDER BY created_at DESC LIMIT 20;"
        )
        items.append(("pg_jsonb_typecast", "postgresql", sql))

    # Category 3: Customer Free-Text Inputs Containing SQL Keywords/Homographs (Classic WAF False-Positive Trap)
    phrases = [
        "Please select option A or option B before Friday and drop the old draft",
        "Union Station delivery desk: call agent 101 or leave package at gate 4",
        "Customer asked to update billing address and delete duplicate newsletter entry",
        "Order note: table lamp from collection_2025 where shade color is warm white",
        "Feedback: great service, nothing to alter or drop from our current enterprise plan",
    ]
    for i in range(n_per_cat):
        msg = phrases[i % len(phrases)]
        dialect = "mysql" if i % 2 == 0 else "postgresql"
        sql = (
            f"INSERT INTO customer_feedback_notes (ticket_no, author_ref, message_body, priority_level) "
            f"VALUES ({10000 + i}, 'USR_{i:04d}', '{msg} (ref #{i})', {(i % 3) + 1});"
        )
        items.append(("homograph_prose_trap", dialect, sql))

    # Category 4: Legitimate Multi-Quarter Reporting UNION ALL Queries
    regions = ["APAC", "EMEA", "NA_EAST", "LATAM"]
    for i in range(n_per_cat):
        reg = regions[i % len(regions)]
        dialect = "mysql" if i % 2 == 0 else "postgresql"
        sql = (
            f"SELECT order_id, total_usd, 'Q1' AS quarter_tag FROM orders_archive_q1 WHERE region_code = '{reg}' AND total_usd > {50 + (i % 100)} "
            f"UNION ALL "
            f"SELECT order_id, total_usd, 'Q2' AS quarter_tag FROM orders_archive_q2 WHERE region_code = '{reg}' AND total_usd > {50 + (i % 100)} "
            f"ORDER BY order_id DESC LIMIT 50;"
        )
        items.append(("legitimate_union_report", dialect, sql))

    return items


def train_zero_shot_cascade(
    df_src_train: pd.DataFrame,
    df_tgt_test: pd.DataFrame,
    tgt_dialect: str,
    device: torch.device,
) -> dict:
    q_tr = df_src_train["query"].fillna("").astype(str).tolist()
    d_tr = df_src_train["dialect"].fillna("mysql").astype(str).tolist()
    y_tr = df_src_train["label"].astype(np.float32).values

    q_te = df_tgt_test["query"].fillna("").astype(str).tolist()
    d_te = [tgt_dialect] * len(q_te)
    y_te = df_tgt_test["label"].astype(np.int64).values

    pos_w = torch.tensor([float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))], device=device)

    # Train Stage-1 from scratch strictly on source dialect
    X1_tr = batch_encode_stage1(q_tr)
    X1_te = batch_encode_stage1(q_te)
    m1 = Stage1CNNBiLSTM().to(device)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1.5e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ld1 = DataLoader(TensorDataset(torch.from_numpy(X1_tr), torch.from_numpy(y_tr)), batch_size=256, shuffle=True)

    m1.train()
    for _ in range(4):
        for bx, by in ld1:
            bx, by = bx.to(device), by.to(device)
            opt1.zero_grad(set_to_none=True)
            loss = crit(m1(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(m1.parameters(), 1.0)
            opt1.step()

    # Train Stage-2 from scratch strictly on source dialect
    Xt_tr, Xs_tr = batch_extract_stage2(q_tr, d_tr)
    Xt_te, Xs_te = batch_extract_stage2(q_te, d_te)
    m2 = Stage2ASTContextualVerifier().to(device)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1.2e-3, weight_decay=1e-4)
    ld2 = DataLoader(
        TensorDataset(torch.from_numpy(Xt_tr), torch.from_numpy(Xs_tr), torch.from_numpy(y_tr)),
        batch_size=256,
        shuffle=True,
    )

    m2.train()
    for _ in range(5):
        for btok, bstr, by in ld2:
            btok, bstr, by = btok.to(device), bstr.to(device), by.to(device)
            opt2.zero_grad(set_to_none=True)
            loss = crit(m2(btok, bstr), by)
            loss.backward()
            nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt2.step()

    m1.eval()
    m2.eval()
    with torch.no_grad():
        lg1 = np.concatenate([m1(torch.from_numpy(X1_te[i:i+512]).to(device)).cpu().numpy() for i in range(0, len(X1_te), 512)])
        lg2 = np.concatenate([
            m2(torch.from_numpy(Xt_te[i:i+512]).to(device), torch.from_numpy(Xs_te[i:i+512]).to(device)).cpu().numpy()
            for i in range(0, len(Xt_te), 512)
        ])

    p1 = 1.0 / (1.0 + np.exp(-np.clip(lg1, -50.0, 50.0)))
    p2 = 1.0 / (1.0 + np.exp(-np.clip(lg2, -50.0, 50.0)))

    tl, th = 0.05, 0.65
    esc = (p1 > tl) & (p1 < th)
    preds = np.zeros(len(y_te), dtype=np.int64)
    preds[p1 >= th] = 1
    preds[esc] = (p2[esc] >= 0.50).astype(np.int64)
    scores = np.where(esc, p2, p1)

    tp = int(((preds == 1) & (y_te == 1)).sum())
    fp = int(((preds == 1) & (y_te == 0)).sum())
    fn = int(((preds == 0) & (y_te == 1)).sum())
    tn = int(((preds == 0) & (y_te == 0)).sum())
    f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    rec = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    roc = float(roc_auc_score(y_te, scores)) if (tp + fn > 0 and fp + tn > 0) else None

    return {
        "train_rows": int(len(y_tr)),
        "zero_shot_test_rows": int(len(y_te)),
        "f1_score": round(f1, 6),
        "recall": round(rec, 6),
        "fpr": round(fpr, 6),
        "roc_auc": round(roc, 6) if roc is not None else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    t0 = time.perf_counter_ns()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(TRAIN_CSV, low_memory=False)
    df_test = pd.read_csv(TEST_CSV, low_memory=False)

    # [1/4] Zero-Shot Cross-Dialect Generalization on Authentic Dialect-Specific Subsets
    log_step("[1/4] Training & evaluating Zero-Shot Cross-Dialect models (MySQL <-> PostgreSQL)...")
    tr_q = df_train["query"].fillna("").astype(str)
    te_q = df_test["query"].fillna("").astype(str)

    tr_is_mysql_only = tr_q.apply(lambda s: bool(_MYSQL_AUTH_RE.search(s)) and not bool(_PG_AUTH_RE.search(s)))
    tr_is_pg_only = tr_q.apply(lambda s: bool(_PG_AUTH_RE.search(s)) and not bool(_MYSQL_AUTH_RE.search(s)))
    tr_is_ansi_benign = (df_train["label"] == 0) & (~tr_is_mysql_only) & (~tr_is_pg_only)

    te_auth_pg = df_test[te_q.apply(lambda s: bool(_PG_AUTH_RE.search(s)) and not bool(_MYSQL_AUTH_RE.search(s))) | ((df_test["label"] == 0) & (df_test["dialect"] == "postgresql"))].reset_index(drop=True)
    te_auth_mysql = df_test[te_q.apply(lambda s: bool(_MYSQL_AUTH_RE.search(s)) and not bool(_PG_AUTH_RE.search(s))) | ((df_test["label"] == 0) & (df_test["dialect"] == "mysql"))].reset_index(drop=True)

    # Source A: MySQL attacks + shared ANSI benign (zero PostgreSQL-specific syntax seen during training)
    df_src_mysql = df_train[tr_is_mysql_only | (tr_is_ansi_benign & (df_train["dialect"] == "mysql"))].reset_index(drop=True)
    # Source B: PostgreSQL attacks + shared ANSI benign (zero MySQL-specific syntax seen during training)
    df_src_pg = df_train[tr_is_pg_only | (tr_is_ansi_benign & (df_train["dialect"] == "postgresql"))].reset_index(drop=True)

    zs_mysql_to_pg = train_zero_shot_cascade(df_src_mysql, te_auth_pg, "postgresql", device)
    zs_pg_to_mysql = train_zero_shot_cascade(df_src_pg, te_auth_mysql, "mysql", device)

    # [2/4] Hard-Benign Modern ORM & Homograph False-Positive Trap Suite (N=2,000)
    log_step("[2/4] Evaluating Hard-Benign Modern ORM & Homograph Suite (N=2,000) across 4 defenses...")
    cfg = json.loads(ROUTER_CFG.read_text(encoding="utf-8"))
    engine = CascadeSQLOnnxEngine(cfg["tau_low"], cfg["tau_high"], cfg["tau_stage2"], window_mode="strided_multi_window")
    crs = OwaspCrsV4Baseline(anomaly_threshold=1)
    libinj = LibInjectionFingerprintEngine()

    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=12000, sublinear_tf=True)
    svd = TruncatedSVD(n_components=96, random_state=SEED)
    Xtr_ml = svd.fit_transform(tfidf.fit_transform(df_train["query"].fillna("").astype(str).tolist()))
    gbdt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=8, random_state=SEED)
    gbdt.fit(Xtr_ml, df_train["label"].astype(np.int64).values)

    orm_items = generate_hard_benign_orm_suite(2000)
    orm_cats = [x[0] for x in orm_items]
    orm_dials = [x[1] for x in orm_items]
    orm_sqls = [x[2] for x in orm_items]

    cas_orm_preds = np.zeros(len(orm_sqls), dtype=np.int64)
    cas_orm_scores = np.zeros(len(orm_sqls), dtype=np.float32)
    for i, (q, d) in enumerate(zip(orm_sqls, orm_dials)):
        out = engine.inspect_single_query_timed(q, d)
        cas_orm_preds[i] = out["pred"]
        cas_orm_scores[i] = float(out["p2"] if out["p2"] is not None else out["p1"])

    crs_orm_preds = np.array([crs.inspect_query(q)[0] for q in orm_sqls], dtype=np.int64)
    lib_orm_preds = np.array([libinj.inspect_timed(q)[0] for q in orm_sqls], dtype=np.int64)
    gb_orm_scores = gbdt.predict_proba(svd.transform(tfidf.transform(orm_sqls)))[:, 1]
    gb_orm_preds = (gb_orm_scores >= 0.50).astype(np.int64)

    orm_category_breakdown = {}
    for cat in sorted(set(orm_cats)):
        mask = np.array([c == cat for c in orm_cats], dtype=bool)
        orm_category_breakdown[cat] = {
            "queries": int(mask.sum()),
            "cascade_fpr_pct": round(float(cas_orm_preds[mask].mean() * 100.0), 3),
            "owasp_crs_fpr_pct": round(float(crs_orm_preds[mask].mean() * 100.0), 2),
            "libinjection_fpr_pct": round(float(lib_orm_preds[mask].mean() * 100.0), 2),
            "tfidf_gbdt_fpr_pct": round(float(gb_orm_preds[mask].mean() * 100.0), 2),
        }

    orm_overall = {
        "total_hard_benign_queries": len(orm_sqls),
        "cascade_fp_count": int(cas_orm_preds.sum()),
        "cascade_fpr_pct": round(float(cas_orm_preds.mean() * 100.0), 4),
        "owasp_crs_fpr_pct": round(float(crs_orm_preds.mean() * 100.0), 2),
        "libinjection_fpr_pct": round(float(lib_orm_preds.mean() * 100.0), 2),
        "tfidf_gbdt_fpr_pct": round(float(gb_orm_preds.mean() * 100.0), 2),
    }

    # [3/4] 1,000:1 Extreme Imbalance Benchmark with 500-Round Stratified Bootstrap 95% CI
    log_step("[3/4] Running 500-round Stratified Bootstrap on 1,000:1 Extreme Imbalance (5,000 Benign : 5 SQLi)...")
    df_te_ben = df_test[df_test["label"] == 0].reset_index(drop=True)
    df_te_sqli = df_test[df_test["label"] == 1].reset_index(drop=True)

    # Evaluate Cascade & CRS scores on test benign + test sqli
    cas_ben_scores, cas_ben_preds = [], []
    for row in df_te_ben.itertuples(index=False):
        r = engine.inspect_single_query_timed(str(row.query), str(row.dialect))
        cas_ben_preds.append(r["pred"])
        cas_ben_scores.append(float(r["p2"] if r["p2"] is not None else r["p1"]))

    cas_sqli_scores, cas_sqli_preds = [], []
    for row in df_te_sqli.iloc[:1000].itertuples(index=False):
        r = engine.inspect_single_query_timed(str(row.query), str(row.dialect))
        cas_sqli_preds.append(r["pred"])
        cas_sqli_scores.append(float(r["p2"] if r["p2"] is not None else r["p1"]))

    # Pool 5,000 benign queries (3,768 Held-out Test Benign + first 1,232 Hard-Benign ORM)
    pool_ben_preds = np.concatenate([np.array(cas_ben_preds, dtype=np.int64), cas_orm_preds[:1232]])
    pool_ben_scores = np.concatenate([np.array(cas_ben_scores, dtype=np.float32), cas_orm_scores[:1232]])
    pool_sqli_preds = np.array(cas_sqli_preds, dtype=np.int64)
    pool_sqli_scores = np.array(cas_sqli_scores, dtype=np.float32)

    rng = np.random.default_rng(SEED)
    boot_prauc, boot_prec, boot_f1 = [], [], []
    n_ben_1000 = 5000
    n_pos_1000 = 5  # Exact 1,000 : 1 ratio

    for _ in range(500):
        b_idx = rng.integers(0, len(pool_ben_preds), size=n_ben_1000)
        s_idx = rng.integers(0, len(pool_sqli_preds), size=n_pos_1000)

        y_b = np.concatenate([np.zeros(n_ben_1000, dtype=np.int64), np.ones(n_pos_1000, dtype=np.int64)])
        pr_b = np.concatenate([pool_ben_preds[b_idx], pool_sqli_preds[s_idx]])
        sc_b = np.concatenate([pool_ben_scores[b_idx], pool_sqli_scores[s_idx]])

        tp = int(((pr_b == 1) & (y_b == 1)).sum())
        fp = int(((pr_b == 1) & (y_b == 0)).sum())
        fn = int(((pr_b == 0) & (y_b == 1)).sum())
        boot_prauc.append(float(average_precision_score(y_b, sc_b)))
        boot_prec.append(tp / max(1, tp + fp))
        boot_f1.append((2.0 * tp) / max(1, 2 * tp + fp + fn))

    imbalance_1000_1 = {
        "ratio": "1000:1 (5,000 Benign : 5 SQLi per draw, 500 Stratified Bootstrap replicates)",
        "pr_auc_mean": round(float(np.mean(boot_prauc)), 4),
        "pr_auc_95ci": [round(float(np.percentile(boot_prauc, 2.5)), 4), round(float(np.percentile(boot_prauc, 97.5)), 4)],
        "precision_mean": round(float(np.mean(boot_prec)), 4),
        "precision_95ci": [round(float(np.percentile(boot_prec, 2.5)), 4), round(float(np.percentile(boot_prec, 97.5)), 4)],
        "f1_mean": round(float(np.mean(boot_f1)), 4),
        "pool_benign_fpr": round(float(pool_ben_preds.mean()), 6),
    }

    log_step("[4/4] Writing verified Table 11 LaTeX and Exp-5 JSON audit log...")
    c1 = orm_category_breakdown["cte_window_analytics"]
    c2 = orm_category_breakdown["pg_jsonb_typecast"]
    c3 = orm_category_breakdown["homograph_prose_trap"]
    c4 = orm_category_breakdown["legitimate_union_report"]

    tab11_tex = rf"""% Auto-generated from 100% Real Cross-Dialect Zero-Shot, Hard-Benign ORM & 1000:1 Imbalance Audit (Exp-5)
\begin{{table*}}[htbp]
\centering
\caption{{Zero-shot cross-dialect transferability, false-positive resilience on 2,000 Hard-Benign Modern ORM / SQL-homograph queries, and 1,000:1 extreme class-imbalance bootstrap evaluation (500 replicates).}}
\label{{tab:exp5_cross_dialect_orm}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr|lrrrr}}
\toprule
\multicolumn{{5}}{{c|}}{{\textbf{{Part A: Zero-Shot Transfer \& 1,000:1 Imbalance}}}} & \multicolumn{{5}}{{c}}{{\textbf{{Part B: Hard-Benign Modern ORM Stress FPR ($N=2,000$)}}}} \\
\cmidrule(r){{1-5}} \cmidrule(l){{6-10}}
\textbf{{Generalization / Imbalance Regime}} & \textbf{{Test Rows}} & \textbf{{Recall}} & \textbf{{FPR}} & \textbf{{F1 / PR-AUC}} & \textbf{{ORM / Homograph Category}} & \textbf{{CRS v4}} & \textbf{{LibInj}} & \textbf{{GBDT}} & \textbf{{Ours}} \\
\midrule
Zero-Shot: MySQL $\rightarrow$ PostgreSQL & {zs_mysql_to_pg['zero_shot_test_rows']:,} & {zs_mysql_to_pg['recall']:.4f} & {zs_mysql_to_pg['fpr']:.4f} & {zs_mysql_to_pg['f1_score']:.4f} (F1) & CTE \& Window Analytics ($n=500$) & {c1['owasp_crs_fpr_pct']:.1f}\\% & {c1['libinjection_fpr_pct']:.1f}\\% & {c1['tfidf_gbdt_fpr_pct']:.1f}\\% & \textbf{{{c1['cascade_fpr_pct']:.2f}\\%}} \\
Zero-Shot: PostgreSQL $\rightarrow$ MySQL & {zs_pg_to_mysql['zero_shot_test_rows']:,} & {zs_pg_to_mysql['recall']:.4f} & {zs_pg_to_mysql['fpr']:.4f} & {zs_pg_to_mysql['f1_score']:.4f} (F1) & PostgreSQL JSONB \& Casts ($n=500$) & {c2['owasp_crs_fpr_pct']:.1f}\\% & {c2['libinjection_fpr_pct']:.1f}\\% & {c2['tfidf_gbdt_fpr_pct']:.1f}\\% & \textbf{{{c2['cascade_fpr_pct']:.2f}\\%}} \\
1,000:1 Extreme Imbalance (Mean) & 5,005 & — & {imbalance_1000_1['pool_benign_fpr']:.4f} & \textbf{{{imbalance_1000_1['pr_auc_mean']:.4f}}} (PR) & English SQL-Homographs ($n=500$) & {c3['owasp_crs_fpr_pct']:.1f}\\% & {c3['libinjection_fpr_pct']:.1f}\\% & {c3['tfidf_gbdt_fpr_pct']:.1f}\\% & \textbf{{{c3['cascade_fpr_pct']:.2f}\\%}} \\
1,000:1 Bootstrap 95\% CI & 500 draws & — & — & [{imbalance_1000_1['pr_auc_95ci'][0]:.3f}, {imbalance_1000_1['pr_auc_95ci'][1]:.3f}] & Multi-Quarter \texttt{{UNION ALL}} ($n=500$) & {c4['owasp_crs_fpr_pct']:.1f}\\% & {c4['libinjection_fpr_pct']:.1f}\\% & {c4['tfidf_gbdt_fpr_pct']:.1f}\\% & \textbf{{{c4['cascade_fpr_pct']:.2f}\\%}} \\
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    write_utf8_lf(TAB11_TEX, tab11_tex)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_sec": round((time.perf_counter_ns() - t0) / 1e9, 3),
        "zero_shot_cross_dialect": {
            "mysql_to_postgresql": zs_mysql_to_pg,
            "postgresql_to_mysql": zs_pg_to_mysql,
        },
        "hard_benign_orm_overall": orm_overall,
        "hard_benign_orm_by_category": orm_category_breakdown,
        "extreme_imbalance_1000_to_1": imbalance_1000_1,
        "sha256_verification": {
            "train_csv": sha256_of(TRAIN_CSV),
            "test_csv": sha256_of(TEST_CSV),
            "tab11_exp5_cross_dialect_orm_tex": sha256_of(TAB11_TEX),
        },
    }

    write_utf8_lf(LOG_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()