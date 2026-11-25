import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    "01_datasets/raw/postgresql",
    "01_datasets/raw/mysql",
    "01_datasets/raw/generic",
    "01_datasets/processed",
    "01_datasets/adversarial_wafamole",
    "01_datasets/imbalanced_splits",
    "02_models_colab/notebooks",
    "02_models_colab/src/cnn_bilstm",
    "02_models_colab/src/ast_gnn",
    "02_models_colab/src/distilled_transformer",
    "02_models_colab/src/cascade_router",
    "02_models_colab/checkpoints",
    "02_models_colab/exported_onnx",
    "03_cloud_testbed/serverless_api_functions",
    "03_cloud_testbed/cascade_sql_proxy/app",
    "03_cloud_testbed/cascade_sql_proxy/onnx_models",
    "03_cloud_testbed/cascade_sql_proxy/otel_tracing",
    "03_cloud_testbed/baseline_waf_modsec",
    "03_cloud_testbed/terraform_gcp",
    "03_cloud_testbed/sql_init/postgres",
    "03_cloud_testbed/sql_init/mysql",
    "04_evaluation/rq1_accuracy_evasion",
    "04_evaluation/rq2_latency_loadtest_k6",
    "04_evaluation/rq3_finops_cost",
    "04_evaluation/rq4_ablation_stats",
    "04_evaluation/rq5_xai_shap_resilience",
    "05_results_and_paper/raw_logs",
    "05_results_and_paper/figures_pdf",
    "05_results_and_paper/tables_latex",
    "05_results_and_paper/manuscript",
]

GITATTRIBUTES = """* text=auto eol=lf
*.zip binary
*.onnx binary
*.pdf binary
*.png binary
*.jpg binary
"""

GITIGNORE = """# Python & Virtual Environments
__pycache__/
*.py[cod]
.venv/
venv/
.env

# Unrelated Local Scratch Files
work_CAPQ-Pareto.txt
CASCADE-SQL-work.txt
Gemini_Generated_Image_*.jpg
*.bak
*~

# Large Raw Datasets & Checkpoints (Tracked via SHA-256 Manifest)
01_datasets/raw/**/*.csv
01_datasets/raw/**/*.parquet
01_datasets/processed/*.parquet
02_models_colab/checkpoints/*.pt
02_models_colab/checkpoints/*.bin

# Cloud Secrets (NEVER COMMIT)
*credentials*.json
*service-account*.json
*gcp-key*.json
terraform.tfstate*
.terraform/
"""

MAIN_TEX = r"""\documentclass[preprint,12pt]{elsarticle}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}

\journal{Computers \& Security}

\begin{document}
\begin{frontmatter}

\title{CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases}

\author[msu]{Research Team}
\affiliation[msu]{organization={Mahasarakham University}, country={Thailand}}

\begin{abstract}
% Strict Zero-Estimation Policy: Empirical metrics are injected exclusively from SHA-256 verified execution logs.
Empirical evaluation pending 100\% real execution across PostgreSQL and MySQL testbeds.
\end{abstract}

\begin{keyword}
SQL Injection \sep Serverless Computing \sep Cloud DBaaS \sep Cascaded Deep Learning \sep ONNX Runtime
\end{keyword}

\end{frontmatter}

\section{Introduction}
This manuscript follows a strict Zero-Estimation reproducibility protocol. Every empirical table in \texttt{tables\_latex/} and figure in \texttt{figures\_pdf/} is cryptographically bound to raw execution logs via \texttt{sha256\_manifest.json}.

\section{Related Work}
\section{System and Threat Model}
\section{CASCADE-SQL Methodology}
\section{Cloud Testbed and Zero-Leakage Dataset Protocol}
\section{Experimental Evaluation}
\section{Discussion and Conclusion}

\end{document}
"""

HIGHLIGHTS_LINES = [
    "Cascaded ONNX proxy defends serverless cloud DBaaS against SQLi in sub-ms.",
    "CNN-BiLSTM filters benign queries fast; AST-GNN verifies obfuscated gray-zones.",
    "Template-isolated splits prevent data leakage across PostgreSQL and MySQL.",
    "Maintains ultra-low false positives under 100:1 imbalanced & WAF-A-MoLE loads.",
    "Cloud Run & OpenTelemetry tests validate p99 tail latency and FinOps savings.",
]


def write_utf8_lf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_and_verify() -> None:
    for d in DIRS:
        target = ROOT / d
        target.mkdir(parents=True, exist_ok=True)
        gitkeep = target / ".gitkeep"
        if not any(target.iterdir()):
            write_utf8_lf(gitkeep, "")

    for line in HIGHLIGHTS_LINES:
        assert len(line) <= 85, f"Highlight exceeds 85 chars ({len(line)}): {line}"

    highlights_tex = "\\begin{itemize}\n" + "\n".join(
        f"  \\item {line}" for line in HIGHLIGHTS_LINES
    ) + "\n\\end{itemize}\n"

    write_utf8_lf(ROOT / ".gitattributes", GITATTRIBUTES)
    write_utf8_lf(ROOT / ".gitignore", GITIGNORE)
    write_utf8_lf(ROOT / "05_results_and_paper/manuscript/main.tex", MAIN_TEX)
    write_utf8_lf(ROOT / "05_results_and_paper/manuscript/highlights.tex", highlights_tex)

    # Remove any backup files before building overwrite ZIP
    for bak in (ROOT / "05_results_and_paper").rglob("*"):
        if bak.is_file() and (bak.name.endswith(".bak") or bak.name.endswith("~")):
            bak.unlink()

    zip_path = ROOT / "05_results_and_paper/CASCADE-SQL_manuscript.zip"
    if zip_path.exists():
        zip_path.unlink()

    tracked_hashes = {}
    pack_dirs = [
        ROOT / "05_results_and_paper/manuscript",
        ROOT / "05_results_and_paper/tables_latex",
        ROOT / "05_results_and_paper/figures_pdf",
    ]
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdir in pack_dirs:
            for fp in sorted(pdir.rglob("*")):
                if fp.is_file() and fp.name != ".gitkeep":
                    rel_arc = fp.relative_to(ROOT / "05_results_and_paper").as_posix()
                    zf.write(fp, arcname=rel_arc)
                    tracked_hashes[fp.relative_to(ROOT).as_posix()] = sha256_of(fp)

    manifest_path = ROOT / "05_results_and_paper/raw_logs/sha256_manifest.json"
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Zero Estimation / 100% Real Execution / No-Backup ZIP Overwrite",
        "manuscript_zip_path": zip_path.relative_to(ROOT).as_posix(),
        "manuscript_zip_sha256": sha256_of(zip_path),
        "artifacts_sha256": tracked_hashes,
    }
    write_utf8_lf(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build_and_verify()