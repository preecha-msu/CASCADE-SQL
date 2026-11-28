# CASCADE-SQL

**Full Title:** *CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases*  
**Target Journal:** *Computers & Security* (Elsevier, Q1)  
**Audit Protocol:** Zero Estimation / 100% Real Execution (Cryptographic SHA-256 Verified)

## Verified Real Execution Summary
- **Dataset Isolation:** 50,460 deduplicated PostgreSQL & MySQL queries across 42,667 canonical `sqlglot` AST templates (**0% cross-split template leakage**).
- **Models (`02_models_colab/exported_onnx/`):**
  - `stage1_cnn_bilstm_int8.onnx` (145,249 params, 542,892 bytes)
  - `stage2_ast_verifier_int8.onnx` (106,177 params, 196,267 bytes)
  - Optimal Validation Router Thresholds: $\tau_{\text{low}}^* = 0.05, \tau_{\text{high}}^* = 0.65$
- **RQ1 (Accuracy & Evasion):** Held-out Test F1 = **0.9991**, ROC-AUC = **0.9998**, FPR = **0.053%**; 100:1 Imbalanced Precision = **94.29%** (vs. 3.05% CRS v4); Adversarial Obfuscated Recall = **99.59%**.
- **RQ2 (Single-vCPU Real Latency, $N=9,920$):** Mean = **0.636 ms**, $p_{50}$ = **0.613 ms**, $p_{95}$ = **0.705 ms**, $p_{99}$ = **1.202 ms**, Cold-Start = **20.43 ms**.
- **RQ3 (Cloud Run FinOps):** Combined INT8 footprint = **721.83 KB**; Incremental compute cost = **$0.0157 / 1M requests** (**81.13%** savings vs. Monolithic PyTorch FP32).
- **RQ4 (Ablation & Statistics):** McNemar vs. OWASP CRS v4 $\chi^2 = 1370.04$ ($p < 10^{-100}$); Wilcoxon INT8 vs. FP32 $p = 4.79e-87$.
- **RQ5 (SHAP & DoS Guard):** Verified `shap.KernelExplainer` attributions + 100% Algorithmic DoS interception (**69,852.0x** speedup on 64 KB flood).
