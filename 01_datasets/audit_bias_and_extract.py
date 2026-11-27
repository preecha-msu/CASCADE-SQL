import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "01_datasets" / "raw" / "generic"

df_csv = pd.read_csv(RAW_DIR / "sqli_modified_corpus.csv", encoding="utf-8", low_memory=False)
df_json = pd.read_json(RAW_DIR / "hf_kaveny_sqli.json")

SQL_VERB_RE = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP|SHOW|EXPLAIN)\b", re.IGNORECASE)
SQLI_TOKEN_RE = re.compile(
    r"(\bUNION\b|\bSELECT\b|\bFROM\b|\bWHERE\b|\bSLEEP\s*\(|\bPG_SLEEP\s*\(|\bWAITFOR\b|\bBENCHMARK\s*\(|\bINFORMATION_SCHEMA\b|/\*!|'[\s]*OR[\s]+|'[\s]*AND[\s]+|--)",
    re.IGNORECASE,
)

benign_df = df_csv[df_csv["Label"] == 0]
sqli_df = df_csv[df_csv["Label"] == 1]

benign_full_sql = int(benign_df["Query"].astype(str).apply(lambda s: bool(SQL_VERB_RE.search(s))).sum())
sqli_full_sql = int(sqli_df["Query"].astype(str).apply(lambda s: bool(SQL_VERB_RE.search(s))).sum())

# Extract pure SQLi payloads inside backticks from Kaveny multi-turn JSON (strictly filtering non-SQL words)
extracted_payloads = set()
BACKTICK_RE = re.compile(r"`([^`\n]{6,500})`")
NON_SQL_CODE_RE = re.compile(r"\b(import\s+requests|pip\s+install|http://|https://|sqlmap\s+-u)\b", re.IGNORECASE)

for conv in df_json["conversations"]:
    if not isinstance(conv, list):
        continue
    for turn in conv:
        text = str(turn.get("value", ""))
        for match in BACKTICK_RE.findall(text):
            cand = match.strip()
            # Exclude Chinese/non-ASCII prose and non-SQL CLI/Python snippets
            if any(ord(ch) > 127 for ch in cand):
                continue
            if NON_SQL_CODE_RE.search(cand):
                continue
            if SQLI_TOKEN_RE.search(cand):
                extracted_payloads.add(cand)

extracted_list = sorted(extracted_payloads)

audit = {
    "csv_benign_total": int(len(benign_df)),
    "csv_benign_full_sql_statements": benign_full_sql,
    "csv_benign_raw_parameter_values": int(len(benign_df) - benign_full_sql),
    "csv_sqli_total": int(len(sqli_df)),
    "csv_sqli_starting_with_sql_verb": sqli_full_sql,
    "kaveny_unique_ascii_sqli_payloads_extracted": len(extracted_list),
    "kaveny_extracted_samples": extracted_list[:8],
    "csv_benign_full_sql_samples": benign_df[benign_df["Query"].astype(str).apply(lambda s: bool(SQL_VERB_RE.search(s)))]["Query"].iloc[:5].tolist(),
}

print(json.dumps(audit, indent=2, ensure_ascii=False))