import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
AUDIT_JSON = ROOT / "05_results_and_paper" / "raw_logs" / "c1_c2_dataset_pipeline_audit.json"
TAB1_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab1_dataset_summary.tex"
MAIN_TEX = ROOT / "05_results_and_paper" / "manuscript" / "main.tex"
ZIP_PATH = ROOT / "05_results_and_paper" / "CASCADE-SQL_manuscript.zip"
MANIFEST_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "sha256_manifest.json"


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


def main() -> None:
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))

    # Cryptographically verify every split CSV on disk matches the recorded SHA-256
    split_files = {
        "train_csv": ROOT / "01_datasets/processed/train.csv",
        "val_csv": ROOT / "01_datasets/processed/val.csv",
        "test_csv": ROOT / "01_datasets/processed/test.csv",
        "imbalanced_100_1_csv": ROOT / "01_datasets/imbalanced_splits/test_imbalanced_100_1.csv",
        "adversarial_obfuscated_csv": ROOT / "01_datasets/adversarial_wafamole/test_adversarial_obfuscated.csv",
    }
    for key, fp in split_files.items():
        actual_hash = sha256_of(fp)
        expected_hash = audit["sha256_hashes"][key]
        assert actual_hash == expected_hash, f"SHA-256 mismatch on {key}: {actual_hash} != {expected_hash}"

    sp = audit["splits"]
    tr, va, te, im, ad = sp["train"], sp["val"], sp["test"], sp["imbalanced_100_1"], sp["adversarial_obfuscated"]

    # Build Table 1 strictly from verified audit numbers
    tab1_content = rf"""% Auto-generated from SHA-256 verified c1_c2_dataset_pipeline_audit.json
% Audit SHA-256: {sha256_of(AUDIT_JSON)}
\begin{{table}}[htbp]
\centering
\caption{{Summary of template-isolated multi-dialect SQL datasets (0\% canonical AST template overlap across splits).}}
\label{{tab:dataset_splits}}
\begin{{tabular}}{{lrrrrl}}
\toprule
\textbf{{Dataset Split}} & \textbf{{Total}} & \textbf{{Benign (0)}} & \textbf{{SQLi (1)}} & \textbf{{Templates}} & \textbf{{SHA-256 Prefix}} \\
\midrule
Train Split & {tr['rows']:,} & {tr['benign']:,} & {tr['sqli']:,} & {tr['templates']:,} & \texttt{{{audit['sha256_hashes']['train_csv'][:12]}}} \\
Validation Split & {va['rows']:,} & {va['benign']:,} & {va['sqli']:,} & {va['templates']:,} & \texttt{{{audit['sha256_hashes']['val_csv'][:12]}}} \\
Held-out Test Split & {te['rows']:,} & {te['benign']:,} & {te['sqli']:,} & {te['templates']:,} & \texttt{{{audit['sha256_hashes']['test_csv'][:12]}}} \\
\midrule
Imbalanced Test (100:1) & {im['rows']:,} & {im['benign']:,} & {im['sqli']:,} & — & \texttt{{{audit['sha256_hashes']['imbalanced_100_1_csv'][:12]}}} \\
Adversarial Obfuscated & {ad['rows']:,} & 0 & {ad['rows']:,} & — & \texttt{{{audit['sha256_hashes']['adversarial_obfuscated_csv'][:12]}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write_utf8_lf(TAB1_TEX, tab1_content)

    # Update main.tex to include verified Section 5 dataset summary & Table 1
    main_tex_updated = rf"""\documentclass[preprint,12pt]{{elsarticle}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}

\journal{{Computers \& Security}}

\begin{{document}}
\begin{{frontmatter}}

\title{{CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases}}

\author[msu]{{Research Team}}
\affiliation[msu]{{organization={{Mahasarakham University}}, country={{Thailand}}}}

\begin{{abstract}}
% Strict Zero-Estimation Policy: Populated exclusively after all RQ1--RQ5 real executions complete.
Empirical evaluation in progress under 100\% real execution and cryptographic SHA-256 verification.
\end{{abstract}}

\begin{{keyword}}
SQL Injection \sep Serverless Computing \sep Cloud DBaaS \sep Cascaded Deep Learning \sep ONNX Runtime
\end{{keyword}}

\end{{frontmatter}}

\section{{Introduction}}
\section{{Related Work}}
\section{{System and Threat Model}}
\section{{CASCADE-SQL Methodology}}

\section{{Cloud Testbed and Zero-Leakage Dataset Protocol}}
To eliminate template memorization and cross-split data leakage, {audit['deduplicated_total']:,} deduplicated queries ({audit['dialect_breakdown']['mysql']:,} MySQL and {audit['dialect_breakdown']['postgresql']:,} PostgreSQL) were normalized into {audit['unique_canonical_templates']:,} unique canonical structural templates via \texttt{{sqlglot}} prior to group-isolated partitioning (verified 0\% template overlap across Train, Validation, and Held-out Test splits), as detailed in Table~\ref{{tab:dataset_splits}}.

\input{{../tables_latex/tab1_dataset_summary.tex}}

\section{{Experimental Evaluation}}
\section{{Discussion and Conclusion}}

\end{{document}}
"""
    write_utf8_lf(MAIN_TEX, main_tex_updated)

    # Overwrite CASCADE-SQL_manuscript.zip completely (zero backup)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    tracked_hashes = {k: audit["sha256_hashes"][k] for k in split_files}
    tracked_hashes["05_results_and_paper/raw_logs/c1_c2_dataset_pipeline_audit.json"] = sha256_of(AUDIT_JSON)

    pack_dirs = [
        ROOT / "05_results_and_paper/manuscript",
        ROOT / "05_results_and_paper/tables_latex",
        ROOT / "05_results_and_paper/figures_pdf",
    ]
    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdir in pack_dirs:
            for fp in sorted(pdir.rglob("*")):
                if fp.is_file() and fp.name != ".gitkeep" and not fp.name.endswith(".bak"):
                    rel_arc = fp.relative_to(ROOT / "05_results_and_paper").as_posix()
                    zf.write(fp, arcname=rel_arc)
                    tracked_hashes[fp.relative_to(ROOT).as_posix()] = sha256_of(fp)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Zero Estimation / 100% Real Execution / No-Backup ZIP Overwrite",
        "manuscript_zip_sha256": sha256_of(ZIP_PATH),
        "artifacts_sha256": tracked_hashes,
        "hardware_probe": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cpu_threads": torch.get_num_threads(),
        },
    }
    write_utf8_lf(MANIFEST_PATH, json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()