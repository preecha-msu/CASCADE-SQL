import math
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "ast_gnn"))

from feature_extractor_stage2 import (
    MAX_SQL_BYTES,
    batch_extract_stage2,
    check_dos_guard,
    extract_single_stage2_features,
)

MAX_LEN_S1 = 256
STRIDE_S1 = 192
CTRL_IDX = 96
UNK_IDX = 97


def _encode_slice_to_row(s: str, out_row: np.ndarray) -> None:
    limit = min(len(s), MAX_LEN_S1)
    for i in range(limit):
        code = ord(s[i])
        if 32 <= code <= 126:
            out_row[i] = code - 31
        elif code in (9, 10, 13):
            out_row[i] = CTRL_IDX
        else:
            out_row[i] = UNK_IDX


def encode_single_stage1(q: str) -> np.ndarray:
    arr = np.zeros((1, MAX_LEN_S1), dtype=np.int64)
    if not isinstance(q, str) or not q:
        return arr
    cleaned = q.replace("\x00", " ").strip()
    _encode_slice_to_row(cleaned, arr[0])
    return arr


def encode_strided_windows_stage1(q: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Bounded overlapping sliding-window encoder covering 100% of bytes up to MAX_SQL_BYTES.
    Guarantees exact right-edge coverage: last window always spans [L - W, L) when L > W.
    """
    if not isinstance(q, str) or not q:
        return np.zeros((1, MAX_LEN_S1), dtype=np.int64), [(0, 0)]
    cleaned = q.replace("\x00", " ").strip()[:MAX_SQL_BYTES]
    L = len(cleaned)
    if L <= MAX_LEN_S1:
        arr = np.zeros((1, MAX_LEN_S1), dtype=np.int64)
        _encode_slice_to_row(cleaned, arr[0])
        return arr, [(0, L)]

    n_win = 1 + math.ceil((L - MAX_LEN_S1) / float(STRIDE_S1))
    arr = np.zeros((n_win, MAX_LEN_S1), dtype=np.int64)
    spans: list[tuple[int, int]] = []
    for k in range(n_win):
        start = min(k * STRIDE_S1, L - MAX_LEN_S1)
        end = start + MAX_LEN_S1
        spans.append((start, end))
        _encode_slice_to_row(cleaned[start:end], arr[k])
    return arr, spans


def batch_encode_stage1(queries: list[str]) -> np.ndarray:
    mat = np.zeros((len(queries), MAX_LEN_S1), dtype=np.int64)
    for r, q in enumerate(queries):
        mat[r] = encode_single_stage1(q)[0]
    return mat


class CascadeSQLOnnxEngine:
    """
    100% Real ONNX Runtime INT8 Cascaded SQLi Defense Engine.
    Supports both 'strided_multi_window' (hardened default) and 'naive_truncated_256' (for ablation/Exp-2).
    """
    def __init__(
        self,
        tau_low: float = 0.05,
        tau_high: float = 0.65,
        tau_stage2: float = 0.50,
        window_mode: str = "strided_multi_window",
    ):
        assert window_mode in ("strided_multi_window", "naive_truncated_256")
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.tau_stage2 = float(tau_stage2)
        self.window_mode = window_mode

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
        if self.window_mode == "naive_truncated_256":
            X1 = batch_encode_stage1(queries)
            logits = np.concatenate(
                [self.sess_s1.run(["logits"], {"input_ids": X1[i:i + batch_size]})[0] for i in range(0, len(X1), batch_size)],
                axis=0,
            )
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

        # Strided multi-window batch evaluation: fast vectorized path for <=256B, exact max-pool for >256B
        n = len(queries)
        probs = np.zeros(n, dtype=np.float32)
        short_idx = []
        short_q = []
        long_idx = []

        for idx, q in enumerate(queries):
            cleaned = (q or "").replace("\x00", " ").strip()[:MAX_SQL_BYTES]
            if len(cleaned) <= MAX_LEN_S1:
                short_idx.append(idx)
                short_q.append(cleaned)
            else:
                long_idx.append(idx)

        if short_q:
            X_short = batch_encode_stage1(short_q)
            lg_short = np.concatenate(
                [self.sess_s1.run(["logits"], {"input_ids": X_short[i:i + batch_size]})[0] for i in range(0, len(X_short), batch_size)],
                axis=0,
            )
            p_short = 1.0 / (1.0 + np.exp(-np.clip(lg_short, -50.0, 50.0)))
            probs[np.array(short_idx, dtype=np.int64)] = p_short

        for idx in long_idx:
            win_mat, _ = encode_strided_windows_stage1(queries[idx])
            lg_win = self.sess_s1.run(["logits"], {"input_ids": win_mat})[0]
            p_win = 1.0 / (1.0 + np.exp(-np.clip(lg_win, -50.0, 50.0)))
            probs[idx] = float(np.max(p_win))

        return probs

    def _extract_focused_stage2(self, query: str, dialect: str, hotspot_span: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
        """If query exceeds 512 chars and Stage-1 localized a hotspot window, preserve prefix verb + hotspot context for Stage-2."""
        cleaned = (query or "").replace("\x00", " ").strip()[:MAX_SQL_BYTES]
        if hotspot_span is None or len(cleaned) <= 512:
            return extract_single_stage2_features(cleaned, dialect)
        st, en = hotspot_span
        ctx_start = max(0, st - 64)
        ctx_end = min(len(cleaned), en + 128)
        focused_sql = cleaned[:96] + " " + cleaned[ctx_start:ctx_end]
        tok_f, str_f = extract_single_stage2_features(focused_sql, dialect)
        _, str_full = extract_single_stage2_features(cleaned, dialect)
        return tok_f, np.maximum(str_f, str_full)

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
                "windows_evaluated": 0,
                "guard_ms": (t1 - t0) / 1e6,
                "stage1_ms": 0.0,
                "stage2_ms": 0.0,
                "total_ms": (t1 - t0) / 1e6,
            }
        t_guard = time.perf_counter_ns()

        hotspot_span = None
        if self.window_mode == "naive_truncated_256":
            x1 = encode_single_stage1(query)
            logit1 = float(self.sess_s1.run(["logits"], {"input_ids": x1})[0][0])
            p1 = float(1.0 / (1.0 + np.exp(-np.clip(logit1, -50.0, 50.0))))
            n_win = 1
        else:
            win_mat, spans = encode_strided_windows_stage1(query)
            logits_w = self.sess_s1.run(["logits"], {"input_ids": win_mat})[0].reshape(-1)
            probs_w = 1.0 / (1.0 + np.exp(-np.clip(logits_w, -50.0, 50.0)))
            best_k = int(np.argmax(probs_w))
            p1 = float(probs_w[best_k])
            hotspot_span = spans[best_k]
            n_win = len(spans)

        t_s1 = time.perf_counter_ns()

        if p1 <= self.tau_low:
            return {
                "pred": 0,
                "route": "fast_benign",
                "p1": p1,
                "p2": None,
                "windows_evaluated": n_win,
                "guard_ms": (t_guard - t0) / 1e6,
                "stage1_ms": (t_s1 - t_guard) / 1e6,
                "stage2_ms": 0.0,
                "total_ms": (t_s1 - t0) / 1e6,
            }
        if p1 >= self.tau_high and n_win == 1:
            return {
                "pred": 1,
                "route": "fast_sqli",
                "p1": p1,
                "p2": None,
                "windows_evaluated": n_win,
                "guard_ms": (t_guard - t0) / 1e6,
                "stage1_ms": (t_s1 - t_guard) / 1e6,
                "stage2_ms": 0.0,
                "total_ms": (t_s1 - t0) / 1e6,
            }

        tok, sfeat = self._extract_focused_stage2(query, dialect, hotspot_span)
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
            "windows_evaluated": n_win,
            "guard_ms": (t_guard - t0) / 1e6,
            "stage1_ms": (t_s1 - t_guard) / 1e6,
            "stage2_ms": (t_s2 - t_s1) / 1e6,
            "total_ms": (t_s2 - t0) / 1e6,
        }