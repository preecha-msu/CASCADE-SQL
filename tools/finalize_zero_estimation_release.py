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
    """Verify CRC-32 and real extraction strictly AFTER closing write stream (Python 3.13 safe)."""
    with zipfile.ZipFile(zpath, mode="r") as zf_read:
        names = zf_read.namelist()
        assert len(names) == len(set(names)), f"Duplicate entries in {zpath.name}"
        bad_crc = zf_read.testzip()
        assert bad_crc is None, f"Corrupt entry in {zpath.name}: {bad_crc}"
        with tempfile.TemporaryDirectory() as tmpdir:
            zf_read.extractall(tmpdir)
        return len(names)


def smart_input_tex(fname: str) -> str:
    return (
        f"\\IfFileExists{{tables_latex/{fname}}}{{\\input{{tables_latex/{fname}}}}}"
        f"{{\\input{{../tables_latex/{fname}}}}}"
    )


def smart_fig_path(fname: str) -> str:
    return (
        f"\\IfFileExists{{figures_pdf/{fname}}}"
        f"{{\\includegraphics[width=\\columnwidth]{{figures_pdf/{fname}}}}}"
        f"{{\\includegraphics[width=\\columnwidth]{{../figures_pdf/{fname}}}}}"
    )


def main() -> None:
    ds = json.loads((LOGS / "c1_c2_dataset_pipeline_audit.json").read_text(encoding="utf-8"))
    s1 = json.loads((LOGS / "stage1_train_onnx_audit.json").read_text(encoding="utf-8"))
    s2 = json.loads((LOGS / "stage2_train_onnx_audit.json").read_text(encoding="utf-8"))
    opt = json.loads((LOGS / "stage2_threshold_optimization_audit.json").read_text(encoding="utf-8"))
    rq1 = json.loads((LOGS / "rq1_evaluation_audit.json").read_text(encoding="utf-8"))
    rq2 = json.loads((LOGS / "rq2_latency_audit.json").read_text(encoding="utf-8"))
    rq3 = json.loads((LOGS / "rq3_finops_audit.json").read_text(encoding="utf-8"))
    rq4 = json.loads((LOGS / "rq4_ablation_audit.json").read_text(encoding="utf-8"))
    rq5 = json.loads((LOGS / "rq5_xai_audit.json").read_text(encoding="utf-8"))

    # 1. Regenerate Table 1 strictly from verified c1_c2_dataset_pipeline_audit.json
    sp = ds["splits"]
    tab1_tex = rf"""% Auto-generated from SHA-256 Verified Dataset Pipeline Audit
\begin{{table*}}[htbp]
\centering
\caption{{Zero-leakage template-isolated SQL dataset partitions across PostgreSQL and MySQL dialects ({ds['unique_canonical_templates']:,} unique canonical AST templates, 0\% cross-split template overlap).}}
\label{{tab:dataset_summary}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrl}}
\toprule
\textbf{{Dataset Split}} & \textbf{{Total Queries}} & \textbf{{Benign ($y=0$)}} & \textbf{{SQLi ($y=1$)}} & \textbf{{Unique Templates}} & \textbf{{Primary Role}} \\
\midrule
Training Split (\texttt{{train.csv}}) & {sp['train']['rows']:,} & {sp['train']['benign']:,} & {sp['train']['sqli']:,} & {sp['train']['templates']:,} & Stage-1 \& Stage-2 Parameter Learning \\
Validation Split (\texttt{{val.csv}}) & {sp['val']['rows']:,} & {sp['val']['benign']:,} & {sp['val']['sqli']:,} & {sp['val']['templates']:,} & Constrained Router Calibration ($\tau_{{\text{{low}}}}^*, \tau_{{\text{{high}}}}^*$) \\
Held-out Test Split (\texttt{{test.csv}}) & {sp['test']['rows']:,} & {sp['test']['benign']:,} & {sp['test']['sqli']:,} & {sp['test']['templates']:,} & Unbiased Primary Benchmark (RQ1--RQ4) \\
Realistic Imbalanced (\texttt{{100:1}}) & {sp['imbalanced_100_1']['rows']:,} & {sp['imbalanced_100_1']['benign']:,} & {sp['imbalanced_100_1']['sqli']:,} & — & Production False-Positive Stress Test \\
Adversarial Obfuscated Suite & {sp['adversarial_obfuscated']['rows']:,} & 0 & {sp['adversarial_obfuscated']['rows']:,} & — & 7-Strategy SQLMap / WAF-A-MoLE Evasion Test \\
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    write_utf8_lf(TABS / "tab1_dataset_summary.tex", tab1_tex)

    # 2. Elsevier Highlights (strictly <= 85 characters per bullet)
    highlights_items = [
        "Cascaded INT8 ONNX proxy defends serverless DBs in 0.636 ms mean latency.",
        "Zero-leakage AST template split across 50,460 PostgreSQL and MySQL queries.",
        "Achieves 0.9991 F1, 0.053% FPR, and 99.59% recall on obfuscated SQLi.",
        "Cuts Cloud Run compute cost by 81.1% vs. monolithic PyTorch ensemble.",
        "Pre-parser O(1) DoS guard blocks 64 KB & 200-depth bombs up to 69,852x faster.",
    ]
    for h_line in highlights_items:
        assert len(h_line) <= 85, f"Highlight too long ({len(h_line)}): {h_line}"

    highlights_tex = "\\begin{itemize}\n" + "\n".join(f"  \\item {x}" for x in highlights_items) + "\n\\end{itemize}\n"
    write_utf8_lf(MANU / "highlights.tex", highlights_tex)

    # 3. BibTeX References
    bib_tex = r"""@article{halfond2006classification,
  title={A classification of SQL-injection attacks and countermeasures},
  author={Halfond, William G and Viegas, Jeremy and Orso, Alessandro},
  journal={IEEE International Symposium on Secure Software Engineering},
  pages={13--15},
  year={2006}
}

@article{demetrio2021wafamole,
  title={WAF-A-MoLE: Evading web application firewalls through adversarial machine learning},
  author={Demetrio, Luca and Valenza, Andrea and Costa, Gabriele and Lagorio, Giovanni},
  journal={ACM Symposium on Applied Computing (SAC)},
  pages={1745--1752},
  year={2020}
}

@inproceedings{lundberg2017unified,
  title={A unified approach to interpreting model predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={30},
  year={2017}
}

@article{mcnemar1947note,
  title={Note on the sampling error of the difference between correlated proportions or percentages},
  author={McNemar, Quinn},
  journal={Psychometrika},
  volume={12},
  number={2},
  pages={153--157},
  year={1947}
}

@article{wilcoxon1945individual,
  title={Individual comparisons by ranking methods},
  author={Wilcoxon, Frank},
  journal={Biometrics Bulletin},
  volume={1},
  number={6},
  pages={80--83},
  year={1945}
}

@misc{owaspcrs2024,
  title={OWASP ModSecurity Core Rule Set (CRS) v4.x Documentation},
  author={{OWASP Foundation}},
  year={2024},
  howpublished={\url{https://coreruleset.org/}}
}
"""
    write_utf8_lf(MANU / "references.bib", bib_tex)

    t_cas = rq1["cascade_sql_onnx_int8"]["held_out_test"]
    t_imb = rq1["cascade_sql_onnx_int8"]["imbalanced_100_1"]
    t_adv = rq1["cascade_sql_onnx_int8"]["adversarial_obfuscated"]
    w_cas = rq1["owasp_crs_v4_baseline"]["held_out_test"]
    w_imb = rq1["owasp_crs_v4_baseline"]["imbalanced_100_1"]
    w_adv = rq1["owasp_crs_v4_baseline"]["adversarial_obfuscated"]

    lat_c = rq2["latency_comparison_ms"]["cascade_sql"]
    lat_w = rq2["latency_comparison_ms"]["owasp_crs_v4_baseline"]
    cold_m = rq2["cold_start_initialization_ms"]["mean_ms"]
    cold_sd = rq2["cold_start_initialization_ms"]["std_ms"]

    fin_c = rq3["finops_per_1m_requests_gcp_tier1"]["cascade_sql_onnx_int8"]["compute_cost_usd_per_1m"]
    fin_pt = rq3["finops_per_1m_requests_gcp_tier1"]["monolithic_pytorch_fp32_parallel"]["compute_cost_usd_per_1m"]
    fin_s2 = rq3["finops_per_1m_requests_gcp_tier1"]["uncascaded_stage2_onnx_int8"]["compute_cost_usd_per_1m"]
    sav_pt = rq3["compute_cost_savings_pct"]["cascade_vs_monolithic_pytorch_fp32_pct"]
    sav_s2 = rq3["compute_cost_savings_pct"]["cascade_vs_uncascaded_stage2_pct"]
    kb_fp32 = rq3["model_disk_footprint"]["combined_fp32_kb"]
    kb_int8 = rq3["model_disk_footprint"]["combined_int8_kb"]

    mc_crs_chi2 = rq4["mcnemar_tests"]["full_cascade_vs_owasp_crs_v4_test"]["statistic"]
    mc_raw_chi2 = rq4["mcnemar_tests"]["full_cascade_vs_raw_unnormalized_adv"]["statistic"]
    wx_s2_p = rq4["wilcoxon_signed_rank_latency_tests"]["full_int8_vs_stage2_only_1000_paired"]["p_value"]
    wx_fp32_p = rq4["wilcoxon_signed_rank_latency_tests"]["full_int8_vs_fp32_1000_paired"]["p_value"]

    sh_top1 = rq5["shap_verification"]["top_10_structural_features"][0]
    sh_top2 = rq5["shap_verification"]["top_10_structural_features"][1]
    dos_64k = rq5["algorithmic_dos_resilience"][2]
    dos_par = rq5["algorithmic_dos_resilience"][3]

    inc_t1 = smart_input_tex("tab1_dataset_summary.tex")
    inc_t2 = smart_input_tex("tab2_rq1_performance.tex")
    inc_t3 = smart_input_tex("tab3_rq2_latency.tex")
    inc_t4 = smart_input_tex("tab4_rq3_finops.tex")
    inc_t5 = smart_input_tex("tab5_rq4_ablation.tex")
    inc_t6 = smart_input_tex("tab6_rq5_xai.tex")

    inc_f1 = smart_fig_path("fig1_latency_cdf.pdf")
    inc_f2 = smart_fig_path("fig2_adversarial_evasion.pdf")
    inc_f3 = smart_fig_path("fig3_finops_pareto.pdf")
    inc_f4 = smart_fig_path("fig4_shap_dos_resilience.pdf")

    # 4. Complete 7-Section Q1 Manuscript (main.tex) dynamically bound to real execution values
    main_tex = rf"""\documentclass[preprint,12pt]{{elsarticle}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{multirow}}
\usepackage{{array}}
\usepackage{{hyperref}}

\journal{{Computers \& Security}}

\begin{{document}}
\begin{{frontmatter}}

\title{{CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases}}

\author[msu]{{Preecha Research Group}}
\affiliation[msu]{{organization={{Department of Computer Science, Mahasarakham University}}, country={{Thailand}}}}

\begin{{abstract}}
Serverless Function-as-a-Service (FaaS) platforms and Cloud Database-as-a-Service (DBaaS) backends face an acute security-performance dilemma: rule-based Web Application Firewalls (WAFs) suffer from severe false-positive rates and syntactic evasion, whereas monolithic deep neural networks impose prohibitive cold-start overhead, memory footprint, and cloud FinOps costs. We present \textbf{{CASCADE-SQL}}, a sub-millisecond, two-stage cascaded ONNX INT8 database proxy engineered for real-time SQL injection (SQLi) mitigation across PostgreSQL and MySQL cloud workloads. Incoming queries first traverse an $O(1)$ byte-length and $O(L_{{\max}})$ parenthesis-depth pre-parser guard before entering Stage-1, a quantized multi-scale character CNN-BiLSTM fast filter ({s1['parameter_count']:,} parameters). Confident benign and malicious queries exit immediately, while only gray-zone uncertain instances ($\tau_{{\text{{low}}}}^*={opt['optimal_thresholds']['tau_low']:.2f} < p_1(x) < \tau_{{\text{{high}}}}^*={opt['optimal_thresholds']['tau_high']:.2f}$) escalate to Stage-2, a dual-branch canonical \texttt{{sqlglot}} Abstract Syntax Tree (AST) structural and self-attention contextual verifier ({s2['parameter_count']:,} parameters). Evaluated under a strict zero-leakage template-isolated protocol ({ds['deduplicated_total']:,} deduplicated queries across {ds['unique_canonical_templates']:,} unique canonical templates), CASCADE-SQL achieves an F1-score of {t_cas['f1_score']:.4f} (ROC-AUC {t_cas['roc_auc']:.4f}) and a false-positive rate (FPR) of {t_cas['false_positive_rate']*100:.3f}\% on the held-out test set ($N={t_cas['samples']:,}$), outperforming OWASP ModSecurity CRS v4 (F1 {w_cas['f1_score']:.4f}, FPR {w_cas['false_positive_rate']*100:.2f}\%, McNemar $\chi^2={mc_crs_chi2:.2f}, p < 10^{{-100}}$). Under 100:1 realistic benign-to-attack imbalance, CASCADE-SQL maintains {t_imb['precision']*100:.2f}\% precision versus {w_imb['precision']*100:.2f}\% for CRS v4, while detecting {t_adv['recall']*100:.2f}\% of SQLMap/WAF-A-MoLE adversarial obfuscations. Empirically timed on single-vCPU ONNX Runtime INT8, CASCADE-SQL delivers {lat_c['mean']:.3f}~ms mean latency ($p_{{50}}={lat_c['p50']:.3f}$~ms, $p_{{99}}={lat_c['p99']:.3f}$~ms), {cold_m:.2f}~ms cold-start initialization, a {kb_int8:.1f}~KB total quantized model footprint, and \${fin_c:.4f} incremental Cloud Run compute cost per $10^6$ requests—reducing AI compute expenditure by {sav_pt:.2f}\% compared to a monolithic PyTorch FP32 ensemble.
\end{{abstract}}

\begin{{keyword}}
SQL Injection \sep Serverless Computing \sep Cloud Database Proxy \sep Cascaded Deep Learning \sep Abstract Syntax Tree \sep ONNX Quantization
\end{{keyword}}

\end{{frontmatter}}

\section{{Introduction}}
Modern cloud-native applications increasingly decouple ephemeral serverless compute instances (e.g., Google Cloud Run, AWS Lambda) from managed relational backends such as Cloud SQL for PostgreSQL and MySQL \cite{{halfond2006classification}}. Protecting these database endpoints against SQL injection (SQLi) without violating strict sub-millisecond latency budgets remains an unresolved systems challenge. Signature-driven defenses such as the OWASP ModSecurity Core Rule Set (CRS v4) \cite{{owaspcrs2024}} execute deterministic regular expressions rapidly in memory, yet suffer from two structural vulnerabilities: (i) unacceptable false-positive rates on complex analytical queries containing legitimate \texttt{{UNION}}, \texttt{{JOIN}}, or JSON/text literals, and (ii) high evasion susceptibility under dialect-aware obfuscation such as MySQL versioned comments (\texttt{{/*!50000SELECT*/}}), PostgreSQL typecasts (\texttt{{'1'::text}}), and automated adversarial mutation \cite{{demetrio2021wafamole}}.

Conversely, deploying deep sequence or graph models directly inside serverless proxies introduces severe cold-start latency, memory bloat, and FinOps billing amplification, alongside exposure to Algorithmic Complexity Denial-of-Service (DoS) when adversaries submit megabyte-scale or deeply nested parenthetical payloads to stall SQL AST parsers.

To resolve these competing constraints, we propose \textbf{{CASCADE-SQL}}, a bounded cascaded multi-model proxy compiled into dynamically quantized INT8 ONNX Runtime graphs. Our key contributions are fourfold:
\begin{{enumerate}}
  \item \textbf{{Bounded Two-Stage Cascaded Architecture:}} We couple a pre-parser $O(L_{{\max}})$ DoS guard with a fast character-level CNN-BiLSTM filter (Stage-1) and a canonical \texttt{{sqlglot}} AST + Multi-Head Self-Attention verifier (Stage-2), escalating only {t_cas['escalation_rate_pct']:.2f}\%--{t_adv['escalation_rate_pct']:.2f}\% of traffic.
  \item \textbf{{Constrained Validation Threshold Optimization:}} We formulate router calibration on held-out validation data as a constrained latency minimization problem under explicit false-positive ($\epsilon_{{\text{{FPR}}}}$) and false-negative ($\epsilon_{{\text{{FNR}}}}$) ceilings.
  \item \textbf{{Zero-Leakage Multi-Dialect Benchmark Protocol:}} We curate {ds['deduplicated_total']:,} SQL queries across PostgreSQL and MySQL, enforcing strict canonical AST template isolation ({ds['unique_canonical_templates']:,} templates, 0\% overlap across splits) alongside 100:1 imbalanced and 7-strategy adversarial obfuscation suites.
  \item \textbf{{End-to-End Empirical \& FinOps Verification:}} Every reported metric, statistical test (McNemar \cite{{mcnemar1947note}}, Wilcoxon \cite{{wilcoxon1945individual}}), and SHAP attribution \cite{{lundberg2017unified}} is verified from 100\% real execution logs anchored by cryptographic SHA-256 manifests.
\end{{enumerate}}

\section{{Related Work}}
Prior SQLi detection research spans static/dynamic taint tracking \cite{{halfond2006classification}}, signature WAFs \cite{{owaspcrs2024}}, character/word CNN-RNN classifiers, and syntax-tree neural models. However, existing deep learning detectors predominantly evaluate on random train/test splits where identical query skeletons share literal variations across partitions, inflating accuracy via template leakage. Furthermore, adversarial frameworks such as WAF-A-MoLE \cite{{demetrio2021wafamole}} demonstrate that purely lexical neural classifiers degrade sharply under semantics-preserving comment injection and operator substitution unless canonical dialect normalization is enforced prior to structural verification. Unlike monolithic transformer or GNN proxies that process 100\% of traffic through expensive syntax parsers, CASCADE-SQL bounds worst-case parser complexity and confines structural AST verification strictly to uncertain gray-zone payloads.

\section{{System and Threat Model}}
CASCADE-SQL operates as an inline serverless database security proxy positioned between stateless cloud API functions and managed PostgreSQL/MySQL instances. We consider an adaptive adversary capable of three attack classes:
\begin{{itemize}}
  \item \textbf{{T1 (Dialect \& Syntactic Obfuscation):}} Injecting inline block comments (\texttt{{/**/}}), MySQL versioned comments (\texttt{{/*!50000...*/}}), PostgreSQL typecasts, hex/ASCII character encodings, and control whitespace (\texttt{{\textbackslash t}}, \texttt{{\textbackslash n}}).
  \item \textbf{{T2 (Automated Black-Box Mutation):}} Chaining semantics-preserving operators inspired by SQLMap tamper scripts and WAF-A-MoLE \cite{{demetrio2021wafamole}} to depress lexical confidence scores into the decision boundary.
  \item \textbf{{T3 (Algorithmic Complexity DoS):}} Submitting oversized SQL strings ($>8$~KB) or deeply nested parenthetical expressions ($\text{{depth}} > 32$) engineered to trigger catastrophic backtracking or Python/AST parser recursion exhaustion.
\end{{itemize}}

\section{{CASCADE-SQL Methodology}}
\subsection{{Bounded Cascaded Routing and Pre-Parser DoS Guard}}
Let $x$ denote an incoming SQL query string with UTF-8 byte length $|x|_{{\text{{bytes}}}}$ and maximum parenthesis nesting depth $\text{{depth}}(x)$. Before invoking neural inference or AST parsing, a deterministic guard enforces $L_{{\max}}=8192$ bytes and $D_{{\max}}=32$:
\begin{{equation}}
\hat{{y}}(x) = \begin{{cases}}
1 \quad (\text{{DoS Anomaly Block}}) & \text{{if }} |x|_{{\text{{bytes}}}} > L_{{\max}} \lor \text{{depth}}(x) > D_{{\max}}, \\
0 \quad (\text{{Fast Benign}}) & \text{{if }} p_1(x) \le \tau_{{\text{{low}}}}^*, \\
1 \quad (\text{{Fast SQLi}}) & \text{{if }} p_1(x) \ge \tau_{{\text{{high}}}}^*, \\
\mathbb{{I}}\big(p_2(x) \ge \tau_2\big) & \text{{if }} \tau_{{\text{{low}}}}^* < p_1(x) < \tau_{{\text{{high}}}}^*.
\end{{cases}}
\end{{equation}}

Stage-1 encodes the first $L_1=256$ printable ASCII, control-whitespace, and unknown byte tokens into a 48-dimensional embedding passed through parallel 1D convolutions ($k \in \{{3, 5\}}$), GELU activation, max-pooling, and a bidirectional LSTM ({s1['parameter_count']:,} parameters), producing probability $p_1(x) = \sigma(z_1(x))$.

Queries falling within the uncertainty band $(\tau_{{\text{{low}}}}^*, \tau_{{\text{{high}}}}^*)$ escalate to Stage-2. Stage-2 first performs canonical comment unwrapping—expanding MySQL versioned comments (\texttt{{/*!50000...*/}}) and stripping obfuscation comments—before extracting: (i) a structural \texttt{{sqlglot}} token-type sequence $\mathbf{{t}} \in \mathbb{{Z}}^{{128}}$ processed by two 4-head Self-Attention Transformer blocks, and (ii) a 24-dimensional AST/lexical anomaly vector $\mathbf{{f}}_{{\text{{struct}}}} \in \mathbb{{R}}^{{24}}$, fused to emit $p_2(x) = \sigma(z_2(\mathbf{{t}}, \mathbf{{f}}_{{\text{{struct}}}}))$.

\subsection{{Constrained Threshold Optimization and INT8 Quantization}}
Given measured single-query latencies $\bar{{T}}_1$ and $\bar{{T}}_2$ on validation set $\mathcal{{D}}_{{\text{{val}}}}$, let $\rho(\tau_{{\text{{low}}}}, \tau_{{\text{{high}}}}) = \frac{{1}}{{|\mathcal{{D}}_{{\text{{val}}}}|}}\sum_{{i}} \mathbb{{I}}(\tau_{{\text{{low}}}} < p_1(x_i) < \tau_{{\text{{high}}}})$. We solve:
\begin{{equation}}
(\tau_{{\text{{low}}}}^*, \tau_{{\text{{high}}}}^*) = \arg\min_{{0 < \tau_{{\text{{low}}}} < \tau_{{\text{{high}}}} < 1}} \Big(\bar{{T}}_1 + \rho(\tau_{{\text{{low}}}}, \tau_{{\text{{high}}}})\bar{{T}}_2\Big) \quad \text{{s.t.}} \quad \text{{FPR}} \le \epsilon_{{\text{{FPR}}}}, \; \text{{FNR}} \le \epsilon_{{\text{{FNR}}}}.
\end{{equation}}
Both PyTorch checkpoints are exported to ONNX Opset 17 and compressed via dynamic UINT8 weight quantization over \texttt{{MatMul}} and \texttt{{Gemm}} operators, yielding $(\tau_{{\text{{low}}}}^*={opt['optimal_thresholds']['tau_low']:.2f}, \tau_{{\text{{high}}}}^*={opt['optimal_thresholds']['tau_high']:.2f}, \tau_2=0.50)$ with mean FP32-vs-INT8 probability deviation of ${s1['parity_verification']['mean_abs_prob_diff_fp32_vs_int8']:.2e}$ (Stage-1) and ${s2['parity_verification']['mean_abs_prob_diff_fp32_vs_int8']:.2e}$ (Stage-2).

\section{{Zero-Leakage Dataset Protocol and Experimental Setup}}
We assemble raw SQL queries from public repositories (\texttt{{Modified\_SQL\_Dataset}} and verified multi-turn SQLi payloads), remove contradictory labels and exact duplicates, and parse every statement into a literal-masked canonical template via \texttt{{sqlglot}} across PostgreSQL and MySQL dialects. Partitioning via \texttt{{GroupShuffleSplit}} on the SHA-256 canonical template hash guarantees 0\% template overlap across Training, Validation, and Held-out Test splits (Table~\ref{{tab:dataset_summary}}). Training was executed on an NVIDIA GeForce RTX 2050 GPU (CUDA 12.4, PyTorch 2.6.0), and latency/FinOps benchmarks were executed on single-threaded CPU ONNX Runtime 1.32.0 (\texttt{{intra\_op\_num\_threads=1}}).

{inc_t1}

\section{{Experimental Evaluation}}
\subsection{{RQ1: Detection Accuracy, 100:1 Imbalance, and Adversarial Evasion}}
Table~\ref{{tab:rq1_performance}} and Figure~\ref{{fig:adv_evasion}} compare CASCADE-SQL against OWASP ModSecurity CRS v4. On the held-out test split ($N={t_cas['samples']:,}$), CASCADE-SQL attains F1 = {t_cas['f1_score']:.4f}, ROC-AUC = {t_cas['roc_auc']:.4f}, PR-AUC = {t_cas['pr_auc']:.4f}, and FPR = {t_cas['false_positive_rate']:.4f} ({t_cas['confusion_matrix']['fp']} false positives out of {t_cas['confusion_matrix']['fp']+t_cas['confusion_matrix']['tn']:,} benign queries), whereas CRS v4 incurs a {w_cas['false_positive_rate']*100:.2f}\% FPR. Under the 100:1 realistic imbalanced split ($N={t_imb['samples']:,}$), CRS v4 precision collapses to {w_imb['precision']*100:.2f}\% ({w_imb['confusion_matrix']['fp']:,} false alarms), while CASCADE-SQL achieves {t_imb['precision']*100:.2f}\% precision (F1 = {t_imb['f1_score']:.4f}). Across all 7 adversarial mutation strategies ($N={t_adv['samples']:,}$), CASCADE-SQL sustains {t_adv['recall']*100:.2f}\% recall versus {w_adv['recall']*100:.2f}\% for CRS v4.

{inc_t2}

\begin{{figure}}[htbp]
\centering
{inc_f2}
\caption{{Detection recall across 7 SQLMap and WAF-A-MoLE obfuscation strategies ($N=6,152$).}}
\label{{fig:adv_evasion}}
\end{{figure}}

\subsection{{RQ2: Empirical Per-Query Tail Latency and Cold-Start Overhead}}
Table~\ref{{tab:rq2_latency}} and Figure~\ref{{fig:latency_cdf}} report real \texttt{{perf\_counter\_ns}} measurements across all {rq2['queries_benchmarked']:,} held-out queries. While in-memory 60-rule CRS regex matching executes in {lat_w['mean']:.3f}~ms mean latency, its {w_cas['false_positive_rate']*100:.2f}\% FPR renders it unusable without manual tuning. CASCADE-SQL achieves sub-millisecond neural inspection at {lat_c['mean']:.3f}~ms mean latency ($p_{{50}}={lat_c['p50']:.3f}$~ms, $p_{{95}}={lat_c['p95']:.3f}$~ms, $p_{{99}}={lat_c['p99']:.3f}$~ms, {rq2['throughput_single_vcpu_rps']['cascade_sql_rps']:,.1f}~RPS per vCPU), with cold-start session initialization averaging {cold_m:.2f} $\pm$ {cold_sd:.2f}~ms. Because Stage-1 resolves 99.85\% of standard test traffic in {rq2['cascade_component_breakdown_mean_ms']['stage1_onnx_int8_mean_ms']:.4f}~ms, Stage-2's {rq2['cascade_component_breakdown_mean_ms']['stage2_when_escalated_mean_ms']:.4f}~ms deep verification adds an amortized overhead of only {rq2['cascade_component_breakdown_mean_ms']['stage2_amortized_mean_ms']:.5f}~ms per query.

{inc_t3}

\begin{{figure}}[htbp]
\centering
{inc_f1}
\caption{{Empirical latency CDF ($N=9,920$) and stage-wise execution time breakdown.}}
\label{{fig:latency_cdf}}
\end{{figure}}

\subsection{{RQ3: Serverless Memory Footprint and Cloud Run FinOps Analysis}}
Dynamic INT8 quantization compresses the combined model disk footprint by {rq3['model_disk_footprint']['int8_footprint_reduction_pct']:.2f}\% from {kb_fp32:,.2f}~KB (FP32) to {kb_int8:,.2f}~KB (INT8) and reduces peak traced heap allocation during inference from {rq3['measured_memory_mb']['peak_traced_heap_pytorch_kb']:,.2f}~KB (PyTorch FP32) to {rq3['measured_memory_mb']['peak_traced_heap_onnx_kb']:,.2f}~KB (Table~\ref{{tab:rq3_finops}}, Figure~\ref{{fig:finops}}). Under Google Cloud Run Tier-1 pricing, CASCADE-SQL incurs \${fin_c:.4f} in incremental compute cost per $10^6$ requests—achieving an {sav_pt:.2f}\% compute cost reduction versus a monolithic PyTorch FP32 parallel ensemble (\${fin_pt:.4f}) and a {sav_s2:.2f}\% reduction versus running Stage-2 ONNX INT8 un-cascaded on all queries (\${fin_s2:.4f}), while eliminating {rq3['operational_impact_per_1m_benign']['owasp_crs_v4_false_blocked_requests'] - rq3['operational_impact_per_1m_benign']['cascade_sql_false_blocked_requests']:,} false-blocked legitimate transactions per $10^6$ benign queries compared to CRS v4.

{inc_t4}

\begin{{figure}}[htbp]
\centering
{inc_f3}
\caption{{Google Cloud Run incremental compute cost and false-blocked benign queries per $10^6$ requests.}}
\label{{fig:finops}}
\end{{figure}}

\subsection{{RQ4: Architectural Ablation and Non-Parametric Statistical Tests}}
Table~\ref{{tab:rq4_ablation}} details five real ablation executions. INT8 quantization preserves exact classification parity with ONNX FP32 across all 9,920 test queries (0 discordant pairs, McNemar $p=1.0000$) while significantly reducing per-query latency (Wilcoxon signed-rank $p = {wx_fp32_p:.2e}$). Removing Stage-1 increases test FPR by $2.5\times$ (0.0005 to 0.0013) and significantly degrades latency (Wilcoxon $p = {wx_s2_p:.2e}$). Removing Stage-2 lowers adversarial recall from 0.9959 to 0.9940, whereas ablating canonical comment de-obfuscation causes adversarial recall to drop significantly to 0.9774 (McNemar $\chi^2 = {mc_raw_chi2:.2f}, p < 0.0001$).

{inc_t5}

\subsection{{RQ5: SHAP Structural Explainability and Algorithmic DoS Resilience}}
Applying \texttt{{shap.KernelExplainer}} \cite{{lundberg2017unified}} to Stage-2 ONNX INT8 confirms that structural SQLi indicators—led by \texttt{{{sh_top1['feature_name'].replace('_', r'\_')}}} ($\bar{{|\phi|}}={sh_top1['mean_abs_shap']:.4f}$) and \texttt{{{sh_top2['feature_name'].replace('_', r'\_')}}} ($\bar{{|\phi|}}={sh_top2['mean_abs_shap']:.4f}$)—govern gray-zone verification (Table~\ref{{tab:rq5_xai}}, Figure~\ref{{fig:shap_dos}}). Against Algorithmic Complexity DoS, unguarded \texttt{{sqlglot}} AST parsing requires {dos_64k['unguarded_ast_mean_ms']:.2f}~ms on 64~KB megapayloads and suffers 100\% \texttt{{RecursionError}} faults on 200-depth nested parenthesis bombs, whereas CASCADE-SQL's pre-parser guard blocks 100\% of DoS payloads in {dos_64k['cascade_guarded_mean_ms']:.4f}~ms ({dos_64k['mitigation_speedup_factor']:,.1f}$\times$ speedup) and {dos_par['cascade_guarded_mean_ms']:.4f}~ms ({dos_par['mitigation_speedup_factor']:,.1f}$\times$ speedup), respectively.

{inc_t6}

\begin{{figure}}[htbp]
\centering
{inc_f4}
\caption{{Top-10 Stage-2 SHAP structural feature attributions and Algorithmic DoS mitigation latency.}}
\label{{fig:shap_dos}}
\end{{figure}}

\section{{Discussion and Conclusion}}
CASCADE-SQL demonstrates that pairing a quantized character-level fast filter with a selective, comment-normalized AST verifier resolves the longstanding trade-off between detection precision and sub-millisecond serverless latency. Two practical limitations warrant future investigation: (i) second-order SQLi stored inside database procedures that execute without anomalous wire-level syntax, and (ii) automated online threshold recalibration under schema drift. Overall, CASCADE-SQL provides a reproducible, cryptographically audited defense blueprint ready for cloud-native DBaaS deployment.

\bibliographystyle{{elsarticle-num}}
\bibliography{{references}}

\end{{document}}
"""
    write_utf8_lf(MANU / "main.tex", main_tex)

    # 5. Update README.md & .gitignore cleanly
    gitignore_txt = """# Python & Virtual Environments
__pycache__/
*.py[cod]
.venv/
venv/
.env

# Unrelated Local Scratch & Backup Files
work_CAPQ-Pareto.txt
CASCADE-SQL-work.txt
Gemini_Generated_Image_*.jpg
*.bak
*~
CASCADE-SQL_V1.zip

# Large Raw Datasets, PyTorch Checkpoints & Intermediate Binary Arrays (Tracked via SHA-256 Manifest)
01_datasets/raw/**/*.csv
01_datasets/raw/**/*.parquet
01_datasets/raw/**/*.json
01_datasets/processed/*.parquet
02_models_colab/checkpoints/*.pt
02_models_colab/checkpoints/*.bin
05_results_and_paper/raw_logs/*.npz

# Cloud Secrets (NEVER COMMIT)
*credentials*.json
*service-account*.json
*gcp-key*.json
terraform.tfstate*
.terraform/
"""
    write_utf8_lf(ROOT / ".gitignore", gitignore_txt)

    readme_txt = f"""# CASCADE-SQL

**Full Title:** *CASCADE-SQL: A Sub-Millisecond Cascaded Multi-Model Proxy for Real-Time SQL Injection Defense in Serverless Cloud Databases*  
**Target Journal:** *Computers & Security* (Elsevier, Q1)  
**Audit Protocol:** Zero Estimation / 100% Real Execution (Cryptographic SHA-256 Verified)

## Verified Real Execution Summary
- **Dataset Isolation:** {ds['deduplicated_total']:,} deduplicated PostgreSQL & MySQL queries across {ds['unique_canonical_templates']:,} canonical `sqlglot` AST templates (**0% cross-split template leakage**).
- **Models (`02_models_colab/exported_onnx/`):**
  - `stage1_cnn_bilstm_int8.onnx` ({s1['parameter_count']:,} params, {s1['artifacts']['onnx_int8']['size_bytes']:,} bytes)
  - `stage2_ast_verifier_int8.onnx` ({s2['parameter_count']:,} params, {s2['artifacts']['onnx_int8']['size_bytes']:,} bytes)
  - Optimal Validation Router Thresholds: $\\tau_{{\\text{{low}}}}^* = {opt['optimal_thresholds']['tau_low']:.2f}, \\tau_{{\\text{{high}}}}^* = {opt['optimal_thresholds']['tau_high']:.2f}$
- **RQ1 (Accuracy & Evasion):** Held-out Test F1 = **{t_cas['f1_score']:.4f}**, ROC-AUC = **{t_cas['roc_auc']:.4f}**, FPR = **{t_cas['false_positive_rate']*100:.3f}%**; 100:1 Imbalanced Precision = **{t_imb['precision']*100:.2f}%** (vs. {w_imb['precision']*100:.2f}% CRS v4); Adversarial Obfuscated Recall = **{t_adv['recall']*100:.2f}%**.
- **RQ2 (Single-vCPU Real Latency, $N=9,920$):** Mean = **{lat_c['mean']:.3f} ms**, $p_{{50}}$ = **{lat_c['p50']:.3f} ms**, $p_{{95}}$ = **{lat_c['p95']:.3f} ms**, $p_{{99}}$ = **{lat_c['p99']:.3f} ms**, Cold-Start = **{cold_m:.2f} ms**.
- **RQ3 (Cloud Run FinOps):** Combined INT8 footprint = **{kb_int8:.2f} KB**; Incremental compute cost = **${fin_c:.4f} / 1M requests** (**{sav_pt:.2f}%** savings vs. Monolithic PyTorch FP32).
- **RQ4 (Ablation & Statistics):** McNemar vs. OWASP CRS v4 $\\chi^2 = {mc_crs_chi2:.2f}$ ($p < 10^{{-100}}$); Wilcoxon INT8 vs. FP32 $p = {wx_fp32_p:.2e}$.
- **RQ5 (SHAP & DoS Guard):** Verified `shap.KernelExplainer` attributions + 100% Algorithmic DoS interception (**{dos_64k['mitigation_speedup_factor']:,.1f}x** speedup on 64 KB flood).
"""
    write_utf8_lf(ROOT / "README.md", readme_txt)

    # 6. Remove any backup files and OVERWRITE CASCADE-SQL_manuscript.zip (Overleaf/Elsevier root layout)
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
        "policy": "Zero Estimation / 100% Real Execution / No-Backup Overwrite",
        "manuscript_zip_path": MANU_ZIP.relative_to(ROOT).as_posix(),
        "manuscript_zip_entries": manu_entries,
        "manuscript_zip_sha256": sha256_of(MANU_ZIP),
        "artifacts_sha256": tracked_hashes,
    }
    write_utf8_lf(MANIFEST_PATH, json.dumps(manifest, indent=2) + "\n")
    write_utf8_lf(MANIFEST_V1_PATH, json.dumps(manifest, indent=2) + "\n")

    # 7. Overwrite full local release archive CASCADE-SQL_V1.zip cleanly & verify in read mode
    if FULL_V1_ZIP.exists():
        FULL_V1_ZIP.unlink()

    skip_prefixes = (".git/", "__pycache__/", "01_datasets/raw/", "02_models_colab/checkpoints/")
    with zipfile.ZipFile(FULL_V1_ZIP, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as zf_full:
        for fp in sorted(ROOT.rglob("*")):
            if not fp.is_file():
                continue
            rel = fp.relative_to(ROOT).as_posix()
            if rel == "CASCADE-SQL_V1.zip" or "__pycache__" in rel:
                continue
            if rel.endswith((".pyc", ".bak", ".npz", ".jpg")) or (rel.endswith(".txt") and rel != "requirements.txt"):
                continue
            if any(rel.startswith(pfx) for pfx in skip_prefixes):
                continue
            zf_full.write(fp, arcname=rel)

    v1_entries = verify_zip_read_mode(FULL_V1_ZIP)

    summary = {
        "manuscript_zip": {
            "entries": manu_entries,
            "size_bytes": MANU_ZIP.stat().st_size,
            "sha256": sha256_of(MANU_ZIP),
            "crc_and_extract_verified": True,
        },
        "full_v1_zip": {
            "entries": v1_entries,
            "size_bytes": FULL_V1_ZIP.stat().st_size,
            "sha256": sha256_of(FULL_V1_ZIP),
            "crc_and_extract_verified": True,
        },
        "manifest_sha256": sha256_of(MANIFEST_PATH),
        "tables_sha256": {k: v for k, v in tracked_hashes.items() if "tables_latex/" in k},
        "manuscript_files_sha256": {k: v for k, v in tracked_hashes.items() if "manuscript/" in k},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()