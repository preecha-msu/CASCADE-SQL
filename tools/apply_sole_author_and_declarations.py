import hashlib
import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "05_results_and_paper" / "raw_logs"
TABS = ROOT / "05_results_and_paper" / "tables_latex"
FIGS = ROOT / "05_results_and_paper" / "figures_pdf"
MANU = ROOT / "05_results_and_paper" / "manuscript"
MANU_ZIP = ROOT / "05_results_and_paper" / "CASCADE-SQL_manuscript.zip"
FULL_V1_ZIP = ROOT / "CASCADE-SQL_V1.zip"
MANIFEST_PATH = LOGS / "sha256_manifest.json"
MANIFEST_V1_PATH = LOGS / "cascade_sql_v1_manifest.json"


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


def verify_zip_read_mode(zpath: Path) -> int:
    with zipfile.ZipFile(zpath, mode="r") as zf_read:
        names = zf_read.namelist()
        assert len(names) == len(set(names)), f"Duplicate entries in {zpath.name}"
        bad_crc = zf_read.testzip()
        assert bad_crc is None, f"Corrupt CRC entry in {zpath.name}: {bad_crc}"
        with tempfile.TemporaryDirectory() as tmpdir:
            zf_read.extractall(tmpdir)
        return len(names)


def main() -> None:
    main_tex_path = MANU / "main.tex"
    tex = main_tex_path.read_text(encoding="utf-8")

    # 1. Replace author & affiliation block with exact Mahasarakham University details (Sole Author)
    old_author_block = r"""\author[msu]{Preecha Research Group}
\affiliation[msu]{organization={Department of Computer Science, Mahasarakham University}, country={Thailand}}"""

    new_author_block = r"""\author[msu]{Preecha Noiumkar\corref{cor1}}
\ead{preecha.n@msu.ac.th}
\cortext[cor1]{Corresponding author.}
\affiliation[msu]{
  organization={Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University},
  city={Maha Sarakham},
  postcode={44150},
  country={Thailand}
}"""

    if old_author_block in tex:
        tex = tex.replace(old_author_block, new_author_block, 1)
    else:
        tex = re.sub(
            r"\\author\[msu\]\{.*?\}\s*(?:\\ead\{.*?\}\s*\\cortext\[cor1\]\{.*?\}\s*)?\\affiliation\[msu\]\{.*?\}\n",
            new_author_block + "\n",
            tex,
            count=1,
            flags=re.DOTALL,
        )

    # Harmonize plural first-person pronouns in prose to crisp sole-author academic voice
    pronoun_map = [
        ("We present ", "This study presents "),
        ("we design, harden, and cryptographically audit ", "this study designs, hardens, and cryptographically audits "),
        ("Our contributions are:", "The primary contributions of this work are:"),
        ("We assume an adaptive attacker", "The threat model considers an adaptive attacker"),
        ("We curate ", "This study curates "),
        ("We solve:", "The constrained threshold optimization problem is formulated as:"),
    ]
    for src_p, dst_p in pronoun_map:
        tex = tex.replace(src_p, dst_p)

    # 2. Inject complete Elsevier Mandatory Declarations before \bibliographystyle
    declarations_tex = r"""
\section*{Data and Code Availability}
All processed zero-leakage multi-dialect SQL splits (\texttt{01\_datasets/processed/train.csv}, \texttt{val.csv}, \texttt{test.csv}), 100:1 realistic imbalanced splits (\texttt{01\_datasets/imbalanced\_splits/test\_imbalanced\_100\_1.csv}), 7-strategy adversarial obfuscated suites (\texttt{01\_datasets/adversarial\_wafamole/test\_adversarial\_obfuscated.csv}), exported FP32 and dynamically quantized UINT8 ONNX models (\texttt{02\_models\_colab/exported\_onnx/stage1\_cnn\_bilstm\_int8.onnx}, \texttt{stage2\_ast\_verifier\_int8.onnx}, \texttt{router\_thresholds.json}), live FastAPI/Uvicorn OpenTelemetry proxy services (\texttt{03\_cloud\_testbed/cascade\_sql\_proxy/app/live\_proxy\_server.py}), benchmark evaluation scripts across RQ1--RQ5 and Experiments 1--6 (\texttt{04\_evaluation/}), raw JSON execution logs, and the cryptographic SHA-256 verification registry (\texttt{05\_results\_and\_paper/raw\_logs/sha256\_manifest.json}) reproducing all 13 tables and 8 figures are publicly archived at \url{https://github.com/preecha-msu/CASCADE-SQL} (Release tag \texttt{v1.0.0}).

\section*{CRediT Authorship Contribution Statement}
\textbf{Preecha Noiumkar:} Conceptualization, Methodology, Software, Data Curation, Formal Analysis, Investigation, Resources, Visualization, Writing -- Original Draft, Writing -- Review \& Editing, Project Administration, Funding Acquisition.

\section*{Declaration of Competing Interest}
The author declares the following financial interests/personal relationships which may be considered as potential competing interests: Preecha Noiumkar reports financial support was provided by Mahasarakham University. Preecha Noiumkar reports a relationship with Mahasarakham University that includes: employment and funding grants. Apart from this institutional support, the author declares no other competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Declaration of Generative AI and AI-Assisted Technologies in the Writing Process}
During the preparation of this work, the author used a generative AI language assistant to assist with \LaTeX{} formatting, manuscript copy-editing, and verification script structuring. After using this tool, the author reviewed, verified, and edited the content and experimental execution logs as needed and takes full responsibility for the content of the publication.

\section*{Ethics Statement}
This study analyzes publicly available SQL benchmark corpora and controlled synthetic/adversarial query generators evaluated strictly within isolated testbed environments. No human participants, personally identifiable information (PII), live production third-party systems, or biological specimens were involved; institutional review board (IRB) approval was not required.

\section*{Funding}
This research project was financially supported by Mahasarakham University.

\section*{Acknowledgments}
This research project was financially supported by Mahasarakham University. The author thanks the Faculty of Informatics and the Research Center of Information Technology for the Future, Mahasarakham University, for providing computational infrastructure.

"""

    # Strip any previous declaration sections if re-run so script is 100% idempotent
    tex = re.sub(
        r"\\section\*\{Data and Code Availability\}.*?(?=\\bibliographystyle)",
        "",
        tex,
        flags=re.DOTALL,
    )
    assert "\\bibliographystyle" in tex, "Missing bibliographystyle anchor in main.tex"
    tex = tex.replace("\\bibliographystyle", declarations_tex.lstrip("\n") + "\\bibliographystyle", 1)
    write_utf8_lf(main_tex_path, tex)

    # 3. Create Standalone Elsevier Cover Letter (cover_letter.tex) matching ARRAY-D-26-07818 style
    cover_letter_tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{parskip}

\begin{document}

\noindent
\textbf{Preecha Noiumkar}\\
Research Center of Information Technology for the Future, Department of Information Technology\\
Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand\\
Email: \texttt{preecha.n@msu.ac.th}

\vspace{0.8em}
\noindent November 28, 2026

\vspace{0.8em}
\noindent
\textbf{Editor-in-Chief}\\
\textit{Computers \& Security} (Elsevier)

\vspace{0.8em}
\noindent
\textbf{Subject:} Submission of manuscript for consideration in \textit{Computers \& Security}

\vspace{0.8em}
\noindent Dear Editor-in-Chief,

I am pleased to submit my original research manuscript entitled \textbf{``CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases''} for consideration for publication as a Regular Research Paper in \textit{Computers \& Security}.

Protecting stateless serverless compute instances (e.g., Google Cloud Run, AWS Lambda) connected to managed relational databases (PostgreSQL and MySQL) presents an acute security-systems trade-off: deterministic rule-based Web Application Firewalls (WAFs) suffer from severe false-positive rates on modern Object-Relational Mapping (ORM) queries and syntactic evasion, whereas monolithic deep neural detectors introduce multi-millisecond CPU latency, cold-start overhead, input truncation blind spots, and Algorithmic Complexity Denial-of-Service (DoS) risks. This manuscript directly addresses \textit{Computers \& Security}'s core scope in practical systems security, intrusion detection, and adversarial machine learning resilience through three primary contributions:

\begin{enumerate}
  \item \textbf{Hardened Multi-Window Cascaded Proxy \& Bounded Pre-Parser DoS Guard:} CASCADE-SQL pairs an $O(1)$ byte-length and $O(L_{\max})$ parenthesis-depth pre-parser guard (blocking 64~KB floods $69,852\times$ faster than unguarded AST parsing and eliminating 200-depth recursion faults) with a strided multi-window character CNN-BiLSTM fast filter ($W=256, S=192$; 145,249 parameters) and a hotspot-focused canonical \texttt{sqlglot} Abstract Syntax Tree (AST) + 4-head Self-Attention verifier (106,177 parameters) compiled into a 721.8~KB dynamically quantized UINT8 ONNX package.
  \item \textbf{Elimination of Truncation Bypasses, ORM False Alarms, and Grey-Box Evasion:} Strided multi-window pooling eliminates the 256-byte prefix-padding blind spot (restoring tail-injected SQLi recall from 0.00\% to 99.10\%--100.00\% across 256~B--4~KB benign prefixes). On 2,000 unseen Hard-Benign Modern ORM queries (CTEs, Window functions, PostgreSQL \texttt{JSONB}, and English SQL-homograph prose), CASCADE-SQL achieves \textbf{0.00\% FPR} (0/2,000 false blocks vs.\ 80.0\% FPR for OWASP CRS v4 and 55.0\% for LibInjection), while dynamically expanding Stage-2 escalation ($\rho: 9.50\% \rightarrow 34.25\%$) to defeat 40,000-query iterative grey-box WAF-A-MoLE evolutionary attacks (McNemar $p = 1.24 \times 10^{-12}$).
  \item \textbf{Zero-Leakage Multi-Benchmark Rigor \& Live Serverless FinOps Verification:} Evaluated across 50,460 deduplicated PostgreSQL and MySQL queries partitioned into 42,667 isolated canonical AST templates (0\% cross-split template overlap), 5-fold group cross-validation achieves $0.9989 \pm 0.0002$ F1 and $0.9998 \pm 0.0001$ ROC-AUC, matching a 1.38M-parameter 4-layer Subword SQL Transformer on adversarial obfuscations (99.63\% vs.\ 99.76\%, $p=0.1338$) while running $4.74\times$ faster. In live HTTP concurrency tests (\texttt{FastAPI}/\texttt{Uvicorn} + \texttt{OpenTelemetry} fronting a 10,000-row indexed SQL backend), CASCADE-SQL sustains 549.0~RPS at 0.00\% error and reduces Google Cloud Run compute costs by 81.13\% (\$0.0157 per $10^6$ requests). All code, ONNX models, and SHA-256 verified execution logs reproducing Tables 1--13 and Figures 1--8 are publicly archived at \url{https://github.com/preecha-msu/CASCADE-SQL} (Release tag \texttt{v1.0.0}).
\end{enumerate}

I confirm that this manuscript is original, has not been published previously, and is not currently under consideration for publication elsewhere. As the sole author, I have approved the final manuscript and disclosed institutional research funding from Mahasarakham University in the Declaration of Interests.

Thank you very much for your time and consideration.

\vspace{1.2em}
\noindent Sincerely,

\vspace{1.5em}
\noindent
\textbf{Preecha Noiumkar} (Sole \& Corresponding Author)\\
Research Center of Information Technology for the Future, Department of Information Technology\\
Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand\\
Email: \texttt{preecha.n@msu.ac.th}

\end{document}
"""
    write_utf8_lf(MANU / "cover_letter.tex", cover_letter_tex)

    # 4. Create Standalone Elsevier Declaration of Interests file (declaration_of_interests.tex)
    doi_tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{parskip}

\begin{document}

\section*{Declaration of Competing Interests}

The author declares the following financial interests/personal relationships which may be considered as potential competing interests:

\textbf{Preecha Noiumkar} reports financial support was provided by \textbf{Mahasarakham University}. \textbf{Preecha Noiumkar} reports a relationship with \textbf{Mahasarakham University} that includes: employment and funding grants.

Apart from this institutional support, the author declares that he has no other known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\vspace{1.5em}
\noindent
\textbf{Preecha Noiumkar} (Sole \& Corresponding Author)\\
Research Center of Information Technology for the Future, Department of Information Technology\\
Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand\\
Email: \texttt{preecha.n@msu.ac.th}

\end{document}
"""
    write_utf8_lf(MANU / "declaration_of_interests.tex", doi_tex)

    # 5. Update README.md author & citation metadata
    readme_path = ROOT / "README.md"
    readme_txt = readme_path.read_text(encoding="utf-8")
    author_line = (
        "**Author:** Preecha Noiumkar (Research Center of Information Technology for the Future, "
        "Department of Information Technology, Faculty of Informatics, Mahasarakham University, "
        "Maha Sarakham 44150, Thailand; `preecha.n@msu.ac.th`)  \n"
        "**Funding:** Financially supported by Mahasarakham University.  \n"
    )
    if "**Author:**" not in readme_txt:
        readme_txt = readme_txt.replace(
            "**Target Journal:**",
            author_line + "**Target Journal:**",
            1,
        )
        write_utf8_lf(readme_path, readme_txt)

    # 6. Overwrite CASCADE-SQL_manuscript.zip & CASCADE-SQL_V1.zip cleanly (Zero Backup)
    for bak in ROOT.rglob("*"):
        if bak.is_file() and (bak.name.endswith(".bak") or bak.name.endswith("~")):
            bak.unlink()

    if MANU_ZIP.exists():
        MANU_ZIP.unlink()

    tracked_hashes = {}
    with zipfile.ZipFile(MANU_ZIP, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as zf:
        for fp in sorted(MANU.rglob("*")):
            if fp.is_file() and fp.name != ".gitkeep":
                zf.write(fp, arcname=fp.name)
                tracked_hashes[fp.relative_to(ROOT).as_posix()] = sha256_of(fp)
        for fp in sorted(TABS.rglob("*.tex")):
            zf.write(fp, arcname=f"tables_latex/{fp.name}")
            tracked_hashes[fp.relative_to(ROOT).as_posix()] = sha256_of(fp)
        for fp in sorted(FIGS.rglob("*.pdf")):
            zf.write(fp, arcname=f"figures_pdf/{fp.name}")
            tracked_hashes[fp.relative_to(ROOT).as_posix()] = sha256_of(fp)

    manu_entries = verify_zip_read_mode(MANU_ZIP)

    for log_fp in sorted(LOGS.glob("*.json")):
        if log_fp.name not in ("sha256_manifest.json", "cascade_sql_v1_manifest.json"):
            tracked_hashes[log_fp.relative_to(ROOT).as_posix()] = sha256_of(log_fp)
    for onnx_fp in sorted((ROOT / "02_models_colab" / "exported_onnx").glob("*")):
        if onnx_fp.is_file() and onnx_fp.name != ".gitkeep":
            tracked_hashes[onnx_fp.relative_to(ROOT).as_posix()] = sha256_of(onnx_fp)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sole_author": "Preecha Noiumkar (preecha.n@msu.ac.th)",
        "affiliation": "Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand",
        "funding": "Mahasarakham University",
        "policy": "Zero Estimation / 100% Real Execution / No-Backup Overwrite",
        "manuscript_zip_path": MANU_ZIP.relative_to(ROOT).as_posix(),
        "manuscript_zip_entries": manu_entries,
        "manuscript_zip_sha256": sha256_of(MANU_ZIP),
        "artifacts_sha256": tracked_hashes,
    }
    write_utf8_lf(MANIFEST_PATH, json.dumps(manifest, indent=2) + "\n")
    write_utf8_lf(MANIFEST_V1_PATH, json.dumps(manifest, indent=2) + "\n")

    if FULL_V1_ZIP.exists():
        FULL_V1_ZIP.unlink()

    skip_prefixes = (".git/", "__pycache__/", "01_datasets/raw/", "02_models_colab/checkpoints/")
    with zipfile.ZipFile(FULL_V1_ZIP, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as zf_full:
        for fp in sorted(ROOT.rglob("*")):
            if not fp.is_file():
                continue
            rel = fp.relative_to(ROOT).as_posix()
            if rel == "CASCADE-SQL_V1.zip" or "__pycache__" in rel or rel.endswith(".db"):
                continue
            if rel.endswith((".pyc", ".bak", ".npz", ".jpg")) or (rel.endswith(".txt") and rel != "requirements.txt"):
                continue
            if any(rel.startswith(pfx) for pfx in skip_prefixes):
                continue
            zf_full.write(fp, arcname=rel)

    v1_entries = verify_zip_read_mode(FULL_V1_ZIP)

    print(json.dumps({
        "sole_author_verified": "Preecha Noiumkar (preecha.n@msu.ac.th)",
        "manuscript_files": {
            "main.tex": sha256_of(MANU / "main.tex"),
            "highlights.tex": sha256_of(MANU / "highlights.tex"),
            "cover_letter.tex": sha256_of(MANU / "cover_letter.tex"),
            "declaration_of_interests.tex": sha256_of(MANU / "declaration_of_interests.tex"),
            "references.bib": sha256_of(MANU / "references.bib"),
        },
        "manuscript_zip": {
            "entries": manu_entries,
            "size_bytes": MANU_ZIP.stat().st_size,
            "sha256": sha256_of(MANU_ZIP),
            "crc_verified": True,
        },
        "full_v1_zip": {
            "entries": v1_entries,
            "size_bytes": FULL_V1_ZIP.stat().st_size,
            "sha256": sha256_of(FULL_V1_ZIP),
            "crc_verified": True,
        },
    }, indent=2))


if __name__ == "__main__":
    main()