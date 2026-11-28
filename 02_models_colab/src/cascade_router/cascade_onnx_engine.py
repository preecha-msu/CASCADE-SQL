import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "ast_gnn"))

from feature_extractor_stage2 import (
    MAX_PAREN_DEPTH,
    MAX_SQL_BYTES,
    batch_extract_stage2,
    check_dos_guard,
    extract_single_stage2_features,
)

MAX_LEN_S1 = 256
CTRL_IDX = 96
UNK_IDX = 97


def encode_single_stage1(q: str) -> np.ndarray:
    arr = np.zeros((1, MAX_LEN_S1), dtype=np.int64)
    if not isinstance(q, str) or not q:
        return arr
    cleaned = q.replace("\x00", " ").strip()
    limit = min(len(cleaned), MAX_LEN_S1)
    for i in range(limit):
        code = ord(cleaned[i])
        if 32 <= code <= 126:
            arr[0, i] = code - 31
        elif code in (9, 10, 13):
            arr[0, i] = CTRL_IDX
        else:
            arr[0, i] = UNK_IDX
    return arr


def batch_encode_stage1(queries: list[str]) -> np.ndarray:
    mat = np.zeros((len(queries), MAX_LEN_S1), dtype=np.int64)
    for r, q in enumerate(queries):
        mat[r] = encode_single_stage1(q)[0]
    return mat


class CascadeSQLOnnxEngine:
    """100% Real ONNX Runtime INT8 Cascaded SQLi Defense Engine."""
    def __init__(self, tau_low: float = 0.05, tau_high: float = 0.95, tau_stage2: float = 0.50):
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.tau_stage2 = float(tau_stage2)

        onnx_dir = ROOT / "02_models_colab" / "exported_onnx"
        self.s1_path = onnx_dir / "stage1_cnn_bilstm_int8.onnx"
        self.s2_path = onnx_dir / "stage2_ast_verifier_int8.onnx"
        if not self.s1_path.exists() or not self.s2_path.exists():
            raise FileNotFoundError(f"Missing real ONNX INT8 models in {onnx_dir}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess_s1 = ort.InferenceSession(str(self.s1_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess_s2 = ort.InferenceSession(str(self.s2_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def predict_stage1_probs_batch(self, queries: list[str], batch_size: int = 512) -> np.ndarray:
        X1 = batch_encode_stage1(queries)
        logits = np.concatenate(
            [self.sess_s1.run(["logits"], {"input_ids": X1[i:i + batch_size]})[0] for i in range(0, len(X1), batch_size)],
            axis=0,
        )
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    def predict_stage2_probs_batch(self, queries: list[str], dialects: list[str], batch_size: int = 512) -> np.ndarray:
        X_tok, X_str = batch_extract_stage2(queries, dialects)
        logits = np.concatenate(
            [
                self.sess_s2.run(
                    ["logits"],
                    {"token_ids": X_tok[i:i + batch_size], "struct_feats": X_str[i:i + batch_size]},
                )[0]
                for i in range(0, len(X_tok), batch_size)
            ],
            axis=0,
        )
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    def inspect_single_query_timed(self, query: str, dialect: str = "mysql") -> Dict:
        t0 = time.perf_counter_ns()
        if check_dos_guard(query):
            t1 = time.perf_counter_ns()
            return {
                "pred": 1,
                "route": "dos_guard_block",
                "p1": 1.0,
                "p2": None,
                "guard_ms": (t1 - t0) / 1e6,
                "stage1_ms": 0.0,
                "stage2_ms": 0.0,
                "total_ms": (t1 - t0) / 1e6,
            }
        t_guard = time.perf_counter_ns()

        x1 = encode_single_stage1(query)
        logit1 = float(self.sess_s1.run(["logits"], {"input_ids": x1})[0][0])
        p1 = float(1.0 / (1.0 + np.exp(-np.clip(logit1, -50.0, 50.0))))
        t_s1 = time.perf_counter_ns()

        if p1 <= self.tau_low:
            return {
                "pred": 0,
                "route": "fast_benign",
                "p1": p1,
                "p2": None,
                "guard_ms": (t_guard - t0) / 1e6,
                "stage1_ms": (t_s1 - t_guard) / 1e6,
                "stage2_ms": 0.0,
                "total_ms": (t_s1 - t0) / 1e6,
            }
        if p1 >= self.tau_high:
            return {
                "pred": 1,
                "route": "fast_sqli",
                "p1": p1,
                "p2": None,
                "guard_ms": (t_guard - t0) / 1e6,
                "stage1_ms": (t_s1 - t_guard) / 1e6,
                "stage2_ms": 0.0,
                "total_ms": (t_s1 - t0) / 1e6,
            }

        tok, sfeat = extract_single_stage2_features(query, dialect)
        logit2 = float(
            self.sess_s2.run(
                ["logits"],
                {"token_ids": tok[None, :], "struct_feats": sfeat[None, :]},
            )[0][0]
        )
        p2 = float(1.0 / (1.0 + np.exp(-np.clip(logit2, -50.0, 50.0))))
        t_s2 = time.perf_counter_ns()

        return {
            "pred": int(p2 >= self.tau_stage2),
            "route": "escalated_stage2",
            "p1": p1,
            "p2": p2,
            "guard_ms": (t_guard - t0) / 1e6,
            "stage1_ms": (t_s1 - t_guard) / 1e6,
            "stage2_ms": (t_s2 - t_s1) / 1e6,
            "total_ms": (t_s2 - t0) / 1e6,
        }