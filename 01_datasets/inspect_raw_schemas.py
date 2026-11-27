import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "01_datasets" / "raw" / "generic"

csv_path = RAW_DIR / "sqli_modified_corpus.csv"
json_path = RAW_DIR / "hf_kaveny_sqli.json"

df_csv = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
df_json = pd.read_json(json_path)

csv_summary = {
    "total_rows": int(len(df_csv)),
    "null_queries": int(df_csv["Query"].isna().sum()),
    "null_labels": int(df_csv["Label"].isna().sum()),
    "label_distribution": {str(k): int(v) for k, v in df_csv["Label"].value_counts(dropna=False).to_dict().items()},
    "sample_benign": str(df_csv[df_csv["Label"] == 0]["Query"].iloc[0])[:120],
    "sample_sqli": str(df_csv[df_csv["Label"] == 1]["Query"].iloc[0])[:120],
}

json_samples = df_json["conversations"].iloc[:2].tolist()

report = {
    "sqli_modified_corpus_csv": csv_summary,
    "hf_kaveny_sqli_json": {
        "total_rows": int(len(df_json)),
        "first_two_conversations": json_samples,
    },
}

print(json.dumps(report, indent=2, ensure_ascii=False))