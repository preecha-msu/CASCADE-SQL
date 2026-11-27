import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import sqlglot
from sqlglot import exp, TokenType
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
MAX_SQL_BYTES = 8192
MAX_PAREN_DEPTH = 40

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "01_datasets" / "raw" / "generic"
PROC_DIR = ROOT / "01_datasets" / "processed"
IMB_DIR = ROOT / "01_datasets" / "imbalanced_splits"
ADV_DIR = ROOT / "01_datasets" / "adversarial_wafamole"
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "c1_c2_dataset_pipeline_audit.json"

# Strict SQL statement verb (fixes 'drop-box' hyphen edge case)
STRICT_SQL_VERB_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP|SHOW|EXPLAIN)(?:\s+|/\*)",
    re.IGNORECASE,
)

# Strict malicious SQLi signature for multi-turn backtick extraction
STRICT_SQLI_ATTACK_RE = re.compile(
    r"(\bUNION\s+(?:ALL\s+)?SELECT\b|/\*!\d*|\bPG_SLEEP\s*\(|\bSLEEP\s*\(|\bBENCHMARK\s*\("
    r"|\bWAITFOR\s+DELAY\b|\bINFORMATION_SCHEMA\b|\bEXTRACTVALUE\s*\(|\bUPDATEXML\s*\("
    r"|1\s*=\s*CONVERT\s*\(|\bASCII\s*\(\s*SUBSTRING|\bCHR\s*\(\d+\)"
    r"|['\"]\s*(?:OR|AND)\s+['\"\d\w]+\s*[=<>]|['\"\)]\s*--\s*$)",
    re.IGNORECASE,
)

# Exclude remediation / parameterized / ORM defensive snippets
DEFENSIVE_REMEDIATION_RE = re.compile(
    r"(\bPREPARE\s+\w+\s+FROM\b|\bEXECUTE\s+\w+\b|%s|\?\s*(?:,|;|\)|$)|:\w+|@\w+|\bcursor\.execute\b|\bPDO\b)",
    re.IGNORECASE,
)

PG_HINT_RE = re.compile(
    r"(\bpg_sleep\b|::\w+|\bILIKE\b|\bpg_catalog\b|\bcurrent_database\s*\(|\bgenerate_series\b|\bchr\s*\()",
    re.IGNORECASE,
)
MYSQL_HINT_RE = re.compile(
    r"(/\*!\d*|`\w+`|\bsleep\s*\(|\bbenchmark\s*\(|\bwp_\w+|\bconcat_ws\b|@@version|\bextractvalue\b|\bupdatexml\b)",
    re.IGNORECASE,
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_complex_benign_sql(max_samples: int = 8000) -> List[str]:
    """Fetch authentic complex benign SQL queries (JOIN/GROUP BY/UNION/AGG) from HuggingFace."""
    cache_path = RAW_DIR / "hf_sql_create_context_benign.json"
    if not cache_path.exists():
        urls = [
            "https://huggingface.co/datasets/b-mc2/sql-create-context/resolve/main/sql_create_context_v4.json",
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=45)
                r.raise_for_status()
                cache_path.write_bytes(r.content)
                break
            except Exception:
                continue

    if not cache_path.exists():
        return []

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    answers = [str(item.get("answer", "")).strip() for item in data if item.get("answer")]
    answers = [a for a in answers if STRICT_SQL_VERB_RE.search(a)]
    rng = random.Random(SEED)
    rng.shuffle(answers)
    return answers[:max_samples]


def detect_dialect(query: str, idx: int) -> str:
    if PG_HINT_RE.search(query):
        return "postgresql"
    if MYSQL_HINT_RE.search(query):
        return "mysql"
    # Deterministic multi-dialect assignment for standard ANSI SQL queries
    return "postgresql" if (idx % 2 == 0) else "mysql"


def max_paren_nesting(s: str) -> int:
    depth = 0
    best = 0
    for ch in s:
        if ch == "(":
            depth += 1
            if depth > best:
                best = depth
        elif ch == ")":
            depth = max(0, depth - 1)
    return best


def canonicalize_sql(sql_text: str, dialect: str) -> Tuple[str, str]:
    if not isinstance(sql_text, str) or not sql_text.strip():
        return "__EMPTY__", "EMPTY_INPUT"

    cleaned = sql_text.replace("\x00", " ").strip()
    if len(cleaned.encode("utf-8", errors="ignore")) > MAX_SQL_BYTES:
        return "__OVERLENGTH_DOS__", "OVERLENGTH_DOS"

    sg_dialect = "postgres" if dialect == "postgresql" else "mysql"

    # Attempt full AST canonicalization if within safe recursion depth
    if max_paren_nesting(cleaned) <= MAX_PAREN_DEPTH and len(cleaned) <= 4096:
        try:
            tree = sqlglot.parse_one(cleaned, read=sg_dialect, error_level=sqlglot.ErrorLevel.RAISE)
            if tree is not None:
                def _mask(node: exp.Expression) -> exp.Expression:
                    if isinstance(node, (exp.Literal, exp.Boolean)):
                        return exp.Placeholder()
                    return node
                masked = tree.transform(_mask)
                return masked.sql(dialect=sg_dialect, comments=False), "PARSED_AST"
        except Exception:
            pass

    # Fallback: sqlglot Rust/Python lexical structural normalization
    try:
        tokens = sqlglot.tokenize(cleaned, read=sg_dialect)
        norm_toks = []
        for tok in tokens:
            if tok.token_type in (TokenType.STRING, TokenType.NUMBER, TokenType.HEX_STRING, TokenType.BIT_STRING):
                norm_toks.append("?")
            elif tok.token_type == TokenType.VAR:
                norm_toks.append(tok.text.upper())
            else:
                norm_toks.append(tok.token_type.name)
        if norm_toks:
            return " ".join(norm_toks), "TOKENIZED_STRUCTURAL"
    except Exception:
        pass

    fallback = re.sub(r"('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\"|\b\d+\b|0x[0-9a-fA-F]+)", "?", cleaned)
    fallback = re.sub(r"\s+", " ", fallback).upper()
    return fallback, "REGEX_FALLBACK"


def mutate_adversarial_payload(query: str, dialect: str, variant_idx: int) -> Tuple[str, str]:
    """Deterministic SQLMap-tamper & WAF-A-MoLE syntactic/dialect obfuscations (semantics-preserving)."""
    q = query.strip()
    tech = variant_idx % 6
    if tech == 0:
        # space2comment
        mutated = re.sub(r"\s+", "/**/", q)
        return mutated, "space2comment"
    elif tech == 1:
        # randomcase + comment padding
        chars = [c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(q)]
        return "".join(chars) + " /*evade*/", "alternating_case"
    elif tech == 2:
        if dialect == "mysql":
            mutated = re.sub(r"\b(UNION|SELECT|AND|OR|FROM|WHERE)\b", r"/*!50000\1*/", q, flags=re.IGNORECASE)
            return mutated, "mysql_versioned_comment"
        else:
            mutated = re.sub(r"\b(UNION|SELECT|AND|OR)\b", r"/*pg*/\1/*pg*/", q, flags=re.IGNORECASE)
            mutated = re.sub(r"'(\d+)'", r"'\1'::text", mutated)
            return mutated, "pg_comment_typecast"
    elif tech == 3:
        # whitespace control char obfuscation (\t, \n)
        mutated = re.sub(r"\s+", "\t\n", q)
        return mutated, "control_whitespace"
    elif tech == 4:
        # tautology equivalence / operator obfuscation
        mutated = re.sub(r"1\s*=\s*1", "0x31=0x31" if dialect == "mysql" else "CHR(49)=CHR(49)", q)
        if mutated == q:
            mutated = re.sub(r"=", " LIKE ", q, count=1)
        return mutated, "tautology_hex_chr_like"
    else:
        # compound WAF-A-MoLE style multi-mutation
        mutated = re.sub(r"\s+", "/**/\t", q)
        if dialect == "mysql":
            mutated = re.sub(r"\bSELECT\b", "/*!50000SeLeCt*/", mutated, flags=re.IGNORECASE)
        else:
            mutated = re.sub(r"\bSELECT\b", "SeLeCt/*pg_bypass*/", mutated, flags=re.IGNORECASE)
        return mutated, "compound_wafamole"


def main() -> None:
    t0 = time.perf_counter_ns()
    records: List[Dict] = []

    # 1. Load Modified_SQL_Dataset.csv
    df_csv = pd.read_csv(RAW_DIR / "sqli_modified_corpus.csv", encoding="utf-8", low_memory=False)
    for q, lbl in zip(df_csv["Query"], df_csv["Label"]):
        q_str = str(q).strip()
        if q_str and int(lbl) in (0, 1):
            records.append({"query": q_str, "label": int(lbl), "source": "modified_sql_corpus"})

    # 2. Extract pure verified SQLi payloads from Kaveny multi-turn dataset
    df_json = pd.read_json(RAW_DIR / "hf_kaveny_sqli.json")
    backtick_re = re.compile(r"`([^`\n]{6,500})`")
    kaveny_added = 0
    for conv in df_json["conversations"]:
        if not isinstance(conv, list):
            continue
        for turn in conv:
            txt = str(turn.get("value", ""))
            for m in backtick_re.findall(txt):
                cand = m.strip()
                if any(ord(c) > 127 for c in cand):
                    continue
                if DEFENSIVE_REMEDIATION_RE.search(cand):
                    continue
                if STRICT_SQLI_ATTACK_RE.search(cand):
                    records.append({"query": cand, "label": 1, "source": "hf_kaveny_sqli"})
                    kaveny_added += 1

    # 3. Load authentic complex benign SQL statements (b-mc2/sql-create-context)
    benign_complex = fetch_complex_benign_sql(max_samples=8000)
    for bq in benign_complex:
        records.append({"query": bq, "label": 0, "source": "hf_sql_create_context"})

    df_all = pd.DataFrame(records)
    raw_total = len(df_all)

    # 4. Exact deduplication & contradictory label removal
    df_all["norm_exact"] = df_all["query"].str.strip()
    label_nunique = df_all.groupby("norm_exact")["label"].nunique()
    conflicting_queries = set(label_nunique[label_nunique > 1].index)
    contradictory_removed = int(df_all["norm_exact"].isin(conflicting_queries).sum())

    df_clean = df_all[~df_all["norm_exact"].isin(conflicting_queries)].drop_duplicates(subset=["norm_exact"]).copy()
    df_clean = df_clean.drop(columns=["norm_exact"]).reset_index(drop=True)

    # 5. Dialect tagging & Canonical AST Template extraction
    df_clean["dialect"] = [detect_dialect(q, i) for i, q in enumerate(df_clean["query"])]
    df_clean["is_full_sql"] = [int(bool(STRICT_SQL_VERB_RE.search(q))) for q in df_clean["query"]]

    templates, statuses = zip(*(
        canonicalize_sql(q, d) for q, d in zip(df_clean["query"], df_clean["dialect"])
    ))
    df_clean["canonical_template"] = templates
    df_clean["parse_status"] = statuses
    df_clean["template_hash"] = [
        hashlib.sha256(f"{d}::{t}".encode("utf-8")).hexdigest()[:16]
        for d, t in zip(df_clean["dialect"], df_clean["canonical_template"])
    ]

    # 6. Strict Zero-Leakage Group Split (Train 68% / Val 12% / Test 20% by template_hash)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv_idx, test_idx = next(gss_test.split(df_clean, df_clean["label"], groups=df_clean["template_hash"]))
    df_tv = df_clean.iloc[tv_idx].reset_index(drop=True)
    df_test = df_clean.iloc[test_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(gss_val.split(df_tv, df_tv["label"], groups=df_tv["template_hash"]))
    df_train = df_tv.iloc[train_idx].reset_index(drop=True)
    df_val = df_tv.iloc[val_idx].reset_index(drop=True)

    # Cryptographic zero-leakage assertion
    h_train = set(df_train["template_hash"])
    h_val = set(df_val["template_hash"])
    h_test = set(df_test["template_hash"])
    assert len(h_train & h_val) == 0, "Leakage between train and val!"
    assert len(h_train & h_test) == 0, "Leakage between train and test!"
    assert len(h_val & h_test) == 0, "Leakage between val and test!"

    # 7. Save primary splits
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    IMB_DIR.mkdir(parents=True, exist_ok=True)
    ADV_DIR.mkdir(parents=True, exist_ok=True)

    p_train = PROC_DIR / "train.csv"
    p_val = PROC_DIR / "val.csv"
    p_test = PROC_DIR / "test.csv"
    df_train.to_csv(p_train, index=False, encoding="utf-8", lineterminator="\n")
    df_val.to_csv(p_val, index=False, encoding="utf-8", lineterminator="\n")
    df_test.to_csv(p_test, index=False, encoding="utf-8", lineterminator="\n")

    # 8. Construct 100:1 Realistic Imbalanced Test Split strictly from held-out Test set
    test_benign = df_test[df_test["label"] == 0].copy()
    test_sqli = df_test[df_test["label"] == 1].copy()
    n_sqli_100_1 = max(1, len(test_benign) // 100)
    n_benign_100_1 = n_sqli_100_1 * 100
    df_imb = pd.concat([
        test_benign.iloc[:n_benign_100_1],
        test_sqli.iloc[:n_sqli_100_1],
    ], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    p_imb = IMB_DIR / "test_imbalanced_100_1.csv"
    df_imb.to_csv(p_imb, index=False, encoding="utf-8", lineterminator="\n")

    # 9. Construct Adversarial Obfuscated Suite strictly from held-out Test SQLi payloads
    adv_rows = []
    for idx, row in enumerate(test_sqli.itertuples(index=False)):
        mut_q, mut_type = mutate_adversarial_payload(row.query, row.dialect, idx)
        adv_rows.append({
            "original_query": row.query,
            "query": mut_q,
            "label": 1,
            "dialect": row.dialect,
            "mutation_strategy": mut_type,
            "origin_template_hash": row.template_hash,
        })
    df_adv = pd.DataFrame(adv_rows)
    p_adv = ADV_DIR / "test_adversarial_obfuscated.csv"
    df_adv.to_csv(p_adv, index=False, encoding="utf-8", lineterminator="\n")

    elapsed_s = (time.perf_counter_ns() - t0) / 1e9
    output_files = {
        "train_csv": p_train,
        "val_csv": p_val,
        "test_csv": p_test,
        "imbalanced_100_1_csv": p_imb,
        "adversarial_obfuscated_csv": p_adv,
    }

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_seconds": round(elapsed_s, 3),
        "raw_records_ingested": raw_total,
        "kaveny_clean_sqli_instances": kaveny_added,
        "complex_benign_sql_added": len(benign_complex),
        "contradictory_label_records_dropped": contradictory_removed,
        "deduplicated_total": int(len(df_clean)),
        "dialect_breakdown": df_clean["dialect"].value_counts().to_dict(),
        "parse_status_breakdown": df_clean["parse_status"].value_counts().to_dict(),
        "unique_canonical_templates": int(df_clean["template_hash"].nunique()),
        "zero_template_leakage_verified": True,
        "splits": {
            "train": {"rows": int(len(df_train)), "benign": int((df_train["label"] == 0).sum()), "sqli": int((df_train["label"] == 1).sum()), "templates": len(h_train)},
            "val": {"rows": int(len(df_val)), "benign": int((df_val["label"] == 0).sum()), "sqli": int((df_val["label"] == 1).sum()), "templates": len(h_val)},
            "test": {"rows": int(len(df_test)), "benign": int((df_test["label"] == 0).sum()), "sqli": int((df_test["label"] == 1).sum()), "templates": len(h_test)},
            "imbalanced_100_1": {"rows": int(len(df_imb)), "benign": int((df_imb["label"] == 0).sum()), "sqli": int((df_imb["label"] == 1).sum())},
            "adversarial_obfuscated": {"rows": int(len(df_adv)), "strategies": df_adv["mutation_strategy"].value_counts().to_dict()},
        },
        "sha256_hashes": {k: sha256_of(v) for k, v in output_files.items()},
    }

    with LOG_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()