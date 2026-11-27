import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "01_datasets" / "raw" / "generic"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "raw_datasets_download_audit.json"

# Multi-mirror verified public SQLi & Benign SQL repositories (GitHub Raw + HF Fallback)
SOURCES = [
    {
        "name": "sqli_modified_corpus.csv",
        "urls": [
            "https://raw.githubusercontent.com/taruntiwarihp/SQL_Injection_Detection/master/sqli.csv",
            "https://raw.githubusercontent.com/ebrasha/sql-injection-dataset/main/Modified_SQL_Dataset.csv",
            "https://huggingface.co/datasets/zrmarine/sql_injection/resolve/main/Modified_SQL_Dataset.csv",
        ],
    },
    {
        "name": "sqli_payloads_owasp.txt",
        "urls": [
            "https://raw.githubusercontent.com/payloadbox/sql-injection-payload-list/master/Intruder/detect/Generic_SQLI.txt",
            "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/SQLi/Generic-SQLi.txt",
        ],
    },
    {
        "name": "sqli_error_time_union.txt",
        "urls": [
            "https://raw.githubusercontent.com/payloadbox/sql-injection-payload-list/master/Intruder/detect/Generic_UnionSelect.txt",
            "https://raw.githubusercontent.com/payloadbox/sql-injection-payload-list/master/Intruder/detect/Generic_TimeBased.txt",
        ],
    },
    {
        "name": "hf_kaveny_sqli.json",
        "urls": [
            "https://huggingface.co/datasets/Kaveny/sql-injection/resolve/main/sqli1.json",
            "https://huggingface.co/api/datasets/zrmarine/sql_injection/parquet/default/train/0.parquet",
        ],
    },
]


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "CASCADE-SQL-ZeroEstimation-Auditor/1.0"})
    return session


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                return {"rows": int(len(df)), "columns": [str(c) for c in df.columns.tolist()], "encoding": enc}
            except Exception:
                continue
    elif suffix == ".json":
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
        return {"rows": int(len(df)), "columns": [str(c) for c in df.columns.tolist()], "encoding": "utf-8"}
    elif suffix == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        return {"rows": len(lines), "columns": ["raw_payload_line"], "encoding": "utf-8"}
    return {"rows": -1, "columns": [], "encoding": "binary"}


def main() -> None:
    session = make_session()
    t0 = time.perf_counter_ns()
    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_files": {},
        "failed_urls": [],
    }

    for item in SOURCES:
        fname = item["name"]
        dest = RAW_DIR / fname
        downloaded = False
        for url in item["urls"]:
            try:
                resp = session.get(url, timeout=45, stream=True)
                resp.raise_for_status()
                with dest.open("wb") as out:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            out.write(chunk)
                if dest.stat().st_size > 128:
                    meta = inspect_file(dest)
                    audit["downloaded_files"][fname] = {
                        "source_url": url,
                        "size_bytes": dest.stat().st_size,
                        "sha256": sha256_of(dest),
                        **meta,
                    }
                    downloaded = True
                    break
            except Exception as exc:
                audit["failed_urls"].append({"file": fname, "url": url, "error": str(exc)})

        if not downloaded and dest.exists():
            dest.unlink()

    audit["elapsed_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
    if not audit["downloaded_files"]:
        raise RuntimeError("No datasets could be downloaded; check network/DNS connectivity.")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()