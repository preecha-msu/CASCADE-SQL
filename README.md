# CASCADE-SQL (Phase-2 Hardened Q1 Release)

**Manuscript Title:** *CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases*  
**Author:** Preecha Noiumkar (Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand; `preecha.n@msu.ac.th`)  
**Funding:** Financially supported by Mahasarakham University.  
**Target Journal:** *Computers & Security* (Elsevier, Q1)  
**Verification Standard:** Zero Estimation / 100% Real Execution / Cryptographic SHA-256 Audited

## Complete Empirical & Hardening Highlights
- **5-Fold Zero-Leakage AST Cross-Validation ($N=50,460$):** **0.9989 ± 0.0002 F1**, **0.9998 ± 0.0001 ROC-AUC**, **0% template overlap** across all 42,667 canonical templates.
- **6-System SOTA Dominance (Exp-1):** Matches 4-Layer Subword SQL Transformer adversarial recall (**99.63%** vs. 99.76%, $p=0.1338$) while running **4.74x faster** in a **7.48x smaller** quantized ONNX footprint (**721.8 KB**), outperforming OWASP CRS v4, LibInjection, TF-IDF+GBDT, and BiLSTM-Attention.
- **Prefix-Padding Bypass Eliminated (Exp-2):** Strided Multi-Window Pooling ($W=256, S=192$) lifts tail-injected SQLi recall across 256B–4KB benign prefixes from **0.00%** (naive truncation) to **99.10%–100.00%**.
- **Adaptive Grey-Box WAF-A-MoLE Resilience (Exp-3):** Dynamic Stage-2 escalation ($\rho: 9.50\% \rightarrow 34.25\%$) defeats evolutionary attacks across 40,000 queries ($p = 1.24 \times 10^{-12}$).
- **Live HTTP Proxy + SQL Backend Concurrency (Exp-4):** FastAPI + Uvicorn + OpenTelemetry proxy sustains **549.0 RPS** across $C=1..100$ concurrent clients with **0.00% error rate** and **1.18–2.55 ms** internal AI spans.
- **Hard-Benign Modern ORM & 1,000:1 Imbalance (Exp-5):** **0.00% FPR** (0/2,000 false alarms on CTEs, Window functions, PostgreSQL JSONB, and SQL-homograph prose vs. 80.0% FPR in CRS v4) and **0.9927 PR-AUC** at 1,000:1 imbalance.
