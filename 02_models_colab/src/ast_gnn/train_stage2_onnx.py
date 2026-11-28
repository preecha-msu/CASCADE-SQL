import hashlib
import json
import sys
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "ast_gnn"))

from feature_extractor_stage2 import (
    MAX_TOK_LEN,
    NUM_STRUCT_FEATS,
    PAD_TOK_ID,
    STAGE2_VOCAB_SIZE,
    STRUCTURAL_FEATURE_NAMES,
    batch_extract_stage2,
)

SEED = 42
TRAIN_CSV = ROOT / "01_datasets" / "processed" / "train.csv"
VAL_CSV = ROOT / "01_datasets" / "processed" / "val.csv"
CKPT_DIR = ROOT / "02_models_colab" / "checkpoints"
ONNX_DIR = ROOT / "02_models_colab" / "exported_onnx"
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "stage2_train_onnx_audit.json"


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


class SelfAttentionBlock(nn.Module):
    """ONNX-clean Multi-Head Self-Attention + Feed-Forward Transformer block."""
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = float(self.d_head ** -0.5)

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(bsz, seq_len, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Dh]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class Stage2ASTContextualVerifier(nn.Module):
    """Dual-branch SQL AST structural + contextual Transformer verifier."""
    def __init__(
        self,
        vocab_size: int = STAGE2_VOCAB_SIZE,
        max_len: int = MAX_TOK_LEN,
        struct_dim: int = NUM_STRUCT_FEATS,
        d_model: int = 64,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOK_ID)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.block1 = SelfAttentionBlock(d_model=d_model, n_heads=4, dropout=0.1)
        self.block2 = SelfAttentionBlock(d_model=d_model, n_heads=4, dropout=0.1)
        self.ctx_norm = nn.LayerNorm(d_model)

        self.struct_net = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model * 2 + 64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, token_ids: torch.Tensor, struct_feats: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(token_ids) + self.pos_emb
        x = self.block1(x)
        x = self.block2(x)
        x = self.ctx_norm(x)

        mean_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        z_struct = self.struct_net(struct_feats)

        fused = torch.cat([mean_pool, max_pool, z_struct], dim=1)
        return self.fusion_head(fused).squeeze(-1)


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


def augment_obfuscation_text(q: str, idx: int) -> str:
    """In-training syntactic perturbation so Stage-2 specializes in obfuscated SQLi."""
    mode = idx % 4
    if mode == 0:
        return q.replace(" ", "/**/")
    elif mode == 1:
        return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(q))
    elif mode == 2:
        return q.replace("SELECT", "/*!50000SELECT*/").replace("select", "/*!50000select*/")
    return q.replace(" ", "\t\n")


def main() -> None:
    t0 = time.perf_counter_ns()
    set_deterministic(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(TRAIN_CSV, low_memory=False)
    df_val = pd.read_csv(VAL_CSV, low_memory=False)

    train_queries = df_train["query"].fillna("").astype(str).tolist()
    train_dialects = df_train["dialect"].fillna("mysql").astype(str).tolist()
    train_labels = df_train["label"].astype(np.float32).tolist()

    # Add 20% syntax-perturbed views strictly from Train set (zero Val/Test leakage)
    aug_step = 5
    extra_q, extra_d, extra_y = [], [], []
    for i in range(0, len(train_queries), aug_step):
        extra_q.append(augment_obfuscation_text(train_queries[i], i))
        extra_d.append(train_dialects[i])
        extra_y.append(train_labels[i])

    all_train_q = train_queries + extra_q
    all_train_d = train_dialects + extra_d
    y_train = np.array(train_labels + extra_y, dtype=np.float32)

    fe_t0 = time.perf_counter_ns()
    X_tok_train, X_str_train = batch_extract_stage2(all_train_q, all_train_d)
    X_tok_val, X_str_val = batch_extract_stage2(
        df_val["query"].fillna("").astype(str).tolist(),
        df_val["dialect"].fillna("mysql").astype(str).tolist(),
    )
    fe_sec = (time.perf_counter_ns() - fe_t0) / 1e9
    y_val = df_val["label"].astype(np.int64).values

    n_pos = float((y_train == 1.0).sum())
    n_neg = float((y_train == 0.0).sum())
    pos_weight_val = n_neg / max(1.0, n_pos)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tok_train),
            torch.from_numpy(X_str_train),
            torch.from_numpy(y_train),
        ),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tok_val), torch.from_numpy(X_str_val)),
        batch_size=512,
        shuffle=False,
    )

    model = Stage2ASTContextualVerifier().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state = None
    epoch_history = []

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter_ns()
        model.train()
        running_loss = 0.0
        seen = 0
        for b_tok, b_str, b_y in train_loader:
            b_tok = b_tok.to(device, non_blocking=True)
            b_str = b_str.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(b_tok, b_str)
            loss = criterion(logits, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(b_y)
            seen += len(b_y)
        scheduler.step()

        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for b_tok, b_str in val_loader:
                val_logits_list.append(model(b_tok.to(device), b_str.to(device)).cpu().numpy())
        val_logits = np.concatenate(val_logits_list, axis=0)
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -50.0, 50.0)))
        val_m = compute_binary_metrics(y_val, val_probs, 0.5)
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
    ckpt_path = CKPT_DIR / "stage2_ast_verifier_best.pt"
    torch.save(best_state, ckpt_path)

    model_cpu = Stage2ASTContextualVerifier().cpu()
    model_cpu.load_state_dict(best_state)
    model_cpu.eval()

    pt_logits_list = []
    with torch.no_grad():
        for b_tok, b_str in val_loader:
            pt_logits_list.append(model_cpu(b_tok, b_str).numpy())
    pt_logits = np.concatenate(pt_logits_list, axis=0)
    pt_probs = 1.0 / (1.0 + np.exp(-np.clip(pt_logits, -50.0, 50.0)))

    onnx_fp32_path = ONNX_DIR / "stage2_ast_verifier_fp32.onnx"
    onnx_int8_path = ONNX_DIR / "stage2_ast_verifier_int8.onnx"

    dummy_tok = torch.zeros((1, MAX_TOK_LEN), dtype=torch.long)
    dummy_str = torch.zeros((1, NUM_STRUCT_FEATS), dtype=torch.float32)

    export_kwargs = dict(
        input_names=["token_ids", "struct_feats"],
        output_names=["logits"],
        dynamic_axes={
            "token_ids": {0: "batch_size"},
            "struct_feats": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    try:
        torch.onnx.export(model_cpu, (dummy_tok, dummy_str), str(onnx_fp32_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, (dummy_tok, dummy_str), str(onnx_fp32_path), **export_kwargs)

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
        [
            sess_fp32.run(["logits"], {"token_ids": X_tok_val[i:i+512], "struct_feats": X_str_val[i:i+512]})[0]
            for i in range(0, len(X_tok_val), 512)
        ],
        axis=0,
    )
    ort_int8_logits = np.concatenate(
        [
            sess_int8.run(["logits"], {"token_ids": X_tok_val[i:i+512], "struct_feats": X_str_val[i:i+512]})[0]
            for i in range(0, len(X_tok_val), 512)
        ],
        axis=0,
    )

    ort_fp32_probs = 1.0 / (1.0 + np.exp(-np.clip(ort_fp32_logits, -50.0, 50.0)))
    ort_int8_probs = 1.0 / (1.0 + np.exp(-np.clip(ort_int8_logits, -50.0, 50.0)))

    # Measure real end-to-end Stage-2 latency (sqlglot AST feature extraction + ONNX INT8 inference)
    from feature_extractor_stage2 import extract_single_stage2_features
    val_q_sample = df_val["query"].fillna("").astype(str).tolist()[:1000]
    val_d_sample = df_val["dialect"].fillna("mysql").astype(str).tolist()[:1000]

    for i in range(30):
        tk, sf = extract_single_stage2_features(val_q_sample[i], val_d_sample[i])
        sess_int8.run(["logits"], {"token_ids": tk[None, :], "struct_feats": sf[None, :]})

    parse_lat_ms, infer_lat_ms, total_s2_ms = [], [], []
    for q_i, d_i in zip(val_q_sample, val_d_sample):
        t_a = time.perf_counter_ns()
        tk, sf = extract_single_stage2_features(q_i, d_i)
        t_b = time.perf_counter_ns()
        sess_int8.run(["logits"], {"token_ids": tk[None, :], "struct_feats": sf[None, :]})
        t_c = time.perf_counter_ns()
        parse_lat_ms.append((t_b - t_a) / 1e6)
        infer_lat_ms.append((t_c - t_b) / 1e6)
        total_s2_ms.append((t_c - t_a) / 1e6)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_execution_sec": round((time.perf_counter_ns() - t0) / 1e9, 3),
        "feature_extraction_sec": round(fe_sec, 3),
        "training_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "parameter_count": int(param_count),
        "structural_features_count": NUM_STRUCT_FEATS,
        "structural_feature_names": STRUCTURAL_FEATURE_NAMES,
        "epoch_history": epoch_history,
        "val_metrics_pytorch_fp32": compute_binary_metrics(y_val, pt_probs, 0.5),
        "val_metrics_onnx_fp32": compute_binary_metrics(y_val, ort_fp32_probs, 0.5),
        "val_metrics_onnx_int8": compute_binary_metrics(y_val, ort_int8_probs, 0.5),
        "parity_verification": {
            "max_abs_logit_diff_pt_vs_onnx_fp32": float(np.max(np.abs(pt_logits - ort_fp32_logits))),
            "max_abs_prob_diff_fp32_vs_int8": float(np.max(np.abs(ort_fp32_probs - ort_int8_probs))),
            "mean_abs_prob_diff_fp32_vs_int8": float(np.mean(np.abs(ort_fp32_probs - ort_int8_probs))),
        },
        "single_query_stage2_latency_ms": {
            "ast_parse_mean_ms": round(float(np.mean(parse_lat_ms)), 4),
            "onnx_int8_infer_mean_ms": round(float(np.mean(infer_lat_ms)), 4),
            "total_mean_ms": round(float(np.mean(total_s2_ms)), 4),
            "total_p50_ms": round(float(np.percentile(total_s2_ms, 50)), 4),
            "total_p95_ms": round(float(np.percentile(total_s2_ms, 95)), 4),
            "total_p99_ms": round(float(np.percentile(total_s2_ms, 99)), 4),
        },
        "artifacts": {
            "feature_extractor_py": {"sha256": sha256_of(ROOT / "02_models_colab/src/ast_gnn/feature_extractor_stage2.py")},
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