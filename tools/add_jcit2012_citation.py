import hashlib
import json
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
    # 1. Add cleaned BibTeX entry to references.bib (41 total references)
    bib_path = MANU / "references.bib"
    bib_txt = bib_path.read_text(encoding="utf-8")

    jcit_entry = """@article{noiumkar2012iptables,
  author = {Noiumkar, Preecha},
  title = {The {SQL} Injection Attacking Prevention Applying the Design Techniques of {IPTABLES}},
  journal = {Journal of Convergence Information Technology (JCIT)},
  publisher = {Advanced Institute of Convergence Information Technology (AICIT)},
  volume = {7},
  number = {22},
  issn = {1975-9320},
  year = {2012}
}
"""
    if "noiumkar2012iptables" not in bib_txt:
        bib_txt = bib_txt.rstrip() + "\n\n" + jcit_entry
        write_utf8_lf(bib_path, bib_txt)

    # 2. Cite naturally in Section 2 (Related Work) of main.tex
    main_path = MANU / "main.tex"
    main_txt = main_path.read_text(encoding="utf-8")

    target_rw = "Early SQLi countermeasures relied on static parse-tree comparison \\cite{buehrer2005parse}, instruction randomization \\cite{boyd2004sqlrand}, and formal command-token boundaries \\cite{su2006essence},"
    updated_rw = (
        "Early SQLi countermeasures relied on static parse-tree comparison \\cite{buehrer2005parse}, "
        "instruction randomization \\cite{boyd2004sqlrand}, formal command-token boundaries \\cite{su2006essence}, "
        "and network-layer packet-filtering rule chains modeled on \\texttt{iptables} \\cite{noiumkar2012iptables},"
    )
    if "noiumkar2012iptables" not in main_txt:
        assert target_rw in main_txt, "Target Related Work sentence not found in main.tex!"
        main_txt = main_txt.replace(target_rw, updated_rw, 1)
        write_utf8_lf(main_path, main_txt)

    # 3. Also include in Table 0 (Related Work Paradigm Comparison)
    tab0_path = TABS / "tab0_related_work.tex"
    tab0_txt = tab0_path.read_text(encoding="utf-8")
    old_t0 = r"Regex Signature WAFs \cite{owaspcrs2024,appelt2018mlwaf}"
    new_t0 = r"Rule Chains \& Regex WAFs \cite{noiumkar2012iptables,owaspcrs2024,appelt2018mlwaf}"
    if "noiumkar2012iptables" not in tab0_txt and old_t0 in tab0_txt:
        tab0_txt = tab0_txt.replace(old_t0, new_t0, 1)
        write_utf8_lf(tab0_path, tab0_txt)

    # 4. Overwrite CASCADE-SQL_manuscript.zip & CASCADE-SQL_V1.zip cleanly (No Backup)
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

    total_bib = bib_path.read_text(encoding="utf-8").count("@")
    print(json.dumps({
        "added_citation_key": "noiumkar2012iptables",
        "total_bib_references": total_bib,
        "updated_sha256": {
            "references.bib": sha256_of(bib_path),
            "main.tex": sha256_of(main_path),
            "tab0_related_work.tex": sha256_of(tab0_path),
            "CASCADE-SQL_manuscript.zip": sha256_of(MANU_ZIP),
            "CASCADE-SQL_V1.zip": sha256_of(FULL_V1_ZIP),
        },
        "manuscript_zip_entries": manu_entries,
        "full_v1_zip_entries": v1_entries,
        "crc_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()