import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_FILES = [
    "01_datasets/processed/train.csv",
    "01_datasets/processed/val.csv",
    "01_datasets/processed/test.csv",
    "01_datasets/imbalanced_splits/test_imbalanced_100_1.csv",
    "01_datasets/adversarial_wafamole/test_adversarial_obfuscated.csv",
]

SIMULATION_PATTERNS = re.compile(
    r"(np\.random\.(gamma|normal|uniform|rand|binomial)|random\.(random|uniform)|simulated|hardcoded)",
    re.IGNORECASE,
)

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

report = {
    "datasets_status": {},
    "onnx_models_found": [],
    "scripts_simulation_scan": {},
    "latex_tab3_extra_ampersand_bug": False,
}

for rel in DATASET_FILES:
    fp = ROOT / rel
    if fp.exists():
        report["datasets_status"][rel] = {
            "exists": True,
            "size_bytes": fp.stat().st_size,
            "sha256": sha256_of(fp),
        }
    else:
        report["datasets_status"][rel] = {"exists": False}

onnx_dir = ROOT / "02_models_colab" / "exported_onnx"
if onnx_dir.exists():
    report["onnx_models_found"] = [
        {"file": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_of(p)}
        for p in sorted(onnx_dir.glob("*.onnx"))
    ]

for py_fp in sorted((ROOT / "02_models_colab").rglob("*.py")) + sorted((ROOT / "04_evaluation").rglob("*.py")):
    rel = py_fp.relative_to(ROOT).as_posix()
    lines = py_fp.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [
        {"line": idx + 1, "code": line.strip()[:100]}
        for idx, line in enumerate(lines)
        if SIMULATION_PATTERNS.search(line)
    ]
    report["scripts_simulation_scan"][rel] = {
        "sha256": sha256_of(py_fp),
        "simulation_hits": hits,
    }

tab3_fp = ROOT / "05_results_and_paper" / "tables_latex" / "tab3_rq2_latency.tex"
if tab3_fp.exists():
    txt = tab3_fp.read_text(encoding="utf-8", errors="replace")
    report["latex_tab3_extra_ampersand_bug"] = bool(re.search(r"&\s*\\\\", txt))

print(json.dumps(report, indent=2))