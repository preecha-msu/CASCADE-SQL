# CASCADE-SQL

**Full Title:** *CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases*
**Target Journal:** *Computers & Security* (Elsevier)
**Verification Policy:** Zero Estimation / 100% Real Execution (Cryptographic SHA-256 Manifest Audited)

## Repository Structure
- `01_datasets/`: Multi-dialect SQL datasets (PostgreSQL, MySQL), Canonical AST template-isolated splits (0% leakage), 100:1 imbalanced splits, and SQLMap/WAF-A-MoLE adversarial payloads.
- `02_models_colab/`: Stage-1 CNN-BiLSTM Fast Filter, Stage-2 AST-GNN + Distilled Transformer, constrained threshold optimizer, and INT8 ONNX exports.
- `03_cloud_testbed/`: Serverless API functions, CASCADE-SQL ONNX Proxy, ModSecurity CRS v4 baseline, and OpenTelemetry tracing.
- `04_evaluation/`: Empirical benchmarks across RQ1-RQ5 (Detection/Evasion, Tail Latency p50/p95/p99, FinOps Cost, Ablation/McNemar/Wilcoxon, SHAP/DoS Resilience).
- `05_results_and_paper/`: SHA-256 execution logs (`raw_logs/sha256_manifest.json`), LaTeX tables, vector figures, `elsarticle` manuscript, and `CASCADE-SQL_manuscript.zip`.
- `tools/`: Automated SHA-256 manifest verifier and zero-backup LaTeX ZIP packager.