import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
MAX_LEN = 256
PAD_IDX = 0
CTRL_IDX = 96
UNK_IDX = 97
VOCAB_SIZE = 98  # 0=PAD, 1..95=printable ASCII (32..126), 96=whitespace ctrl, 97=UNK

ROOT = Path(__file__).resolve().parents[3]
TRAIN_CSV = ROOT / "01_datasets" / "processed" / "train.csv"
VAL_CSV = ROOT / "01_datasets" / "processed" / "val.csv"
CKPT_DIR = ROOT / "02_models_colab" / "checkpoints"
ONNX_DIR = ROOT / "02_models_colab" / "exported_onnx"
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "stage1_train_onnx_audit.json"


def set_deterministic(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_queries_to_matrix(queries: list[str], max_len: int = MAX_LEN) -> np.ndarray:
    arr = np.zeros((len(queries), max_len), dtype=np.int64)
    for row_idx, q in enumerate(queries):
        if not isinstance(q, str) or not q:
            continue
        cleaned = q.replace("\x00", " ").strip()
        limit = min(len(cleaned), max_len)
        for col_idx in range(limit):
            code = ord(cleaned[col_idx])
            if 32 <= code <= 126:
                arr[row_idx, col_idx] = code - 31
            elif code in (9, 10, 13):
                arr[row_idx, col_idx] = CTRL_IDX
            else:
                arr[row_idx, col_idx] = UNK_IDX
    return arr


class Stage1CNNBiLSTM(nn.Module):
    """Multi-scale character CNN + BiLSTM fast lexical/syntactic filter."""
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 48, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids).transpose(1, 2)  # [B, E, L]
        c3 = self.act(self.conv3(emb))
        c5 = self.act(self.conv5(emb))
        feats = self.pool(torch.cat([c3, c5], dim=1)).transpose(1, 2)  # [B, L/2, 128]
        seq_out, (hn, _) = self.bilstm(feats)
        h_last = torch.cat([hn[0], hn[1]], dim=1)  # [B, 2*H]
        h_max, _ = torch.max(seq_out, dim=1)       # [B, 2*H]
        logits = self.head(torch.cat([h_last, h_max], dim=1)).squeeze(-1)
        return logits


def compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    acc = float((preds == y_true).mean())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, tp + fn)
    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    t0 = time.perf_counter_ns()
    set_deterministic(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(TRAIN_CSV, low_memory=False)
    df_val = pd.read_csv(VAL_CSV, low_memory=False)

    X_train = encode_queries_to_matrix(df_train["query"].fillna("").astype(str).tolist())
    y_train = df_train["label"].astype(np.float32).values
    X_val = encode_queries_to_matrix(df_val["query"].fillna("").astype(str).tolist())
    y_val = df_val["label"].astype(np.int64).values

    n_pos = float((y_train == 1.0).sum())
    n_neg = float((y_train == 0.0).sum())
    pos_weight_val = n_neg / max(1.0, n_pos)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=512,
        shuffle=False,
    )

    model = Stage1CNNBiLSTM().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state = None
    epoch_history = []

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter_ns()
        model.train()
        running_loss = 0.0
        seen = 0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(by)
            seen += len(by)
        scheduler.step()

        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                val_logits_list.append(model(bx).cpu().numpy())
        val_logits = np.concatenate(val_logits_list, axis=0)
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -50.0, 50.0)))
        val_m = compute_binary_metrics(y_val, val_probs, threshold=0.5)
        ep_sec = (time.perf_counter_ns() - ep_t0) / 1e9

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, seen), 6),
            "val_f1": val_m["f1_score"],
            "val_fpr": val_m["fpr"],
            "val_fnr": val_m["fnr"],
            "epoch_sec": round(ep_sec, 3),
        })

        if val_m["f1_score"] > best_val_f1:
            best_val_f1 = val_m["f1_score"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    ckpt_path = CKPT_DIR / "stage1_cnn_bilstm_best.pt"
    torch.save(best_state, ckpt_path)

    model_cpu = Stage1CNNBiLSTM().cpu()
    model_cpu.load_state_dict(best_state)
    model_cpu.eval()

    # Compute reference PyTorch CPU logits on full Validation set
    pt_logits_list = []
    with torch.no_grad():
        for (bx,) in val_loader:
            pt_logits_list.append(model_cpu(bx).numpy())
    pt_logits = np.concatenate(pt_logits_list, axis=0)
    pt_probs = 1.0 / (1.0 + np.exp(-np.clip(pt_logits, -50.0, 50.0)))

    onnx_fp32_path = ONNX_DIR / "stage1_cnn_bilstm_fp32.onnx"
    onnx_int8_path = ONNX_DIR / "stage1_cnn_bilstm_int8.onnx"
    dummy_in = torch.zeros((1, MAX_LEN), dtype=torch.long)

    export_kwargs = dict(
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model_cpu, dummy_in, str(onnx_fp32_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy_in, str(onnx_fp32_path), **export_kwargs)

    onnx.checker.check_model(onnx.load(str(onnx_fp32_path)))

    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp32 = ort.InferenceSession(str(onnx_fp32_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(onnx_int8_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    ort_fp32_logits = np.concatenate(
        [sess_fp32.run(["logits"], {"input_ids": X_val[i:i+512]})[0] for i in range(0, len(X_val), 512)],
        axis=0,
    )
    ort_int8_logits = np.concatenate(
        [sess_int8.run(["logits"], {"input_ids": X_val[i:i+512]})[0] for i in range(0, len(X_val), 512)],
        axis=0,
    )

    ort_fp32_probs = 1.0 / (1.0 + np.exp(-np.clip(ort_fp32_logits, -50.0, 50.0)))
    ort_int8_probs = 1.0 / (1.0 + np.exp(-np.clip(ort_int8_logits, -50.0, 50.0)))

    # Measure real single-query CPU latency (warmed up, 1000 real validation queries)
    for i in range(50):
        sess_int8.run(["logits"], {"input_ids": X_val[i:i+1]})
    single_lat_ms = []
    for i in range(min(1000, len(X_val))):
        t_s = time.perf_counter_ns()
        sess_int8.run(["logits"], {"input_ids": X_val[i:i+1]})
        single_lat_ms.append((time.perf_counter_ns() - t_s) / 1e6)
    single_lat_arr = np.array(single_lat_ms, dtype=np.float64)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_execution_sec": round((time.perf_counter_ns() - t0) / 1e9, 3),
        "training_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "parameter_count": int(param_count),
        "pos_weight_used": round(pos_weight_val, 6),
        "dataset_sha256": {
            "train_csv": sha256_of(TRAIN_CSV),
            "val_csv": sha256_of(VAL_CSV),
        },
        "epoch_history": epoch_history,
        "val_metrics_pytorch_fp32": compute_binary_metrics(y_val, pt_probs, 0.5),
        "val_metrics_onnx_fp32": compute_binary_metrics(y_val, ort_fp32_probs, 0.5),
        "val_metrics_onnx_int8": compute_binary_metrics(y_val, ort_int8_probs, 0.5),
        "parity_verification": {
            "max_abs_logit_diff_pt_vs_onnx_fp32": float(np.max(np.abs(pt_logits - ort_fp32_logits))),
            "max_abs_prob_diff_pt_vs_onnx_fp32": float(np.max(np.abs(pt_probs - ort_fp32_probs))),
            "max_abs_prob_diff_fp32_vs_int8": float(np.max(np.abs(ort_fp32_probs - ort_int8_probs))),
            "mean_abs_prob_diff_fp32_vs_int8": float(np.mean(np.abs(ort_fp32_probs - ort_int8_probs))),
        },
        "single_query_cpu_latency_int8_ms": {
            "mean": round(float(np.mean(single_lat_arr)), 4),
            "p50": round(float(np.percentile(single_lat_arr, 50)), 4),
            "p95": round(float(np.percentile(single_lat_arr, 95)), 4),
            "p99": round(float(np.percentile(single_lat_arr, 99)), 4),
        },
        "artifacts": {
            "ckpt_pt": {"size_bytes": ckpt_path.stat().st_size, "sha256": sha256_of(ckpt_path)},
            "onnx_fp32": {"size_bytes": onnx_fp32_path.stat().st_size, "sha256": sha256_of(onnx_fp32_path)},
            "onnx_int8": {"size_bytes": onnx_int8_path.stat().st_size, "sha256": sha256_of(onnx_int8_path)},
        },
    }

    with LOG_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()