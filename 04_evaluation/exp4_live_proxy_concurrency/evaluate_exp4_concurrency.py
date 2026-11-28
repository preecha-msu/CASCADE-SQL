from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = ROOT / "01_datasets" / "processed" / "test.csv"
DB_PATH = ROOT / "03_cloud_testbed" / "sql_init" / "live_testbed_relational.db"
PROXY_APP_PATH = ROOT / "03_cloud_testbed" / "cascade_sql_proxy" / "app" / "live_proxy_server.py"
LOG_PATH = ROOT / "05_results_and_paper" / "raw_logs" / "exp4_live_proxy_concurrency_audit.json"
TAB10_TEX = ROOT / "05_results_and_paper" / "tables_latex" / "tab10_exp4_live_proxy_concurrency.tex"


def log_step(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def initialize_relational_db(db_path: Path) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("""
        CREATE TABLE enterprise_users (
            user_id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            region TEXT NOT NULL,
            tier TEXT NOT NULL,
            active INTEGER NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE enterprise_orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            sku TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES enterprise_users(user_id)
        );
    """)
    cur.execute("CREATE INDEX idx_users_region_tier ON enterprise_users(region, tier, active);")
    cur.execute("CREATE INDEX idx_orders_user_status ON enterprise_orders(user_id, status);")

    users_data = [
        (i, f"user{i}@cloud-corp.example", ["APAC", "EMEA", "AMER"][i % 3], ["FREE", "PRO", "ENT"][i % 3], 1 if i % 10 != 0 else 0)
        for i in range(1, 4001)
    ]
    orders_data = [
        (j, (j % 4000) + 1, f"SKU-{(j % 250):04d}", round(19.99 + (j % 500) * 1.75, 2), ["PAID", "PENDING", "SHIPPED"][j % 3])
        for j in range(1, 6001)
    ]
    cur.executemany("INSERT INTO enterprise_users VALUES (?, ?, ?, ?, ?);", users_data)
    cur.executemany("INSERT INTO enterprise_orders VALUES (?, ?, ?, ?, ?);", orders_data)
    conn.commit()
    conn.close()
    return {"users_rows": 4000, "orders_rows": 6000, "sqlite_version": sqlite3.sqlite_version}


REAL_EXEC_SQL_TEMPLATES = [
    "SELECT u.user_id, u.email, COUNT(o.order_id) FROM enterprise_users u JOIN enterprise_orders o ON u.user_id = o.user_id WHERE u.region = 'APAC' AND o.status = 'PAID' GROUP BY u.user_id LIMIT 8;",
    "SELECT sku, SUM(amount) AS rev, COUNT(*) AS cnt FROM enterprise_orders WHERE status = 'SHIPPED' GROUP BY sku ORDER BY rev DESC LIMIT 5;",
    "SELECT user_id, email, tier FROM enterprise_users WHERE region = 'EMEA' AND tier = 'ENT' AND active = 1 LIMIT 10;",
    "SELECT o.order_id, o.sku, o.amount FROM enterprise_orders o WHERE o.user_id = 42 AND o.amount > 25.0 ORDER BY o.order_id DESC;",
]


def main() -> None:
    t0 = time.perf_counter_ns()
    log_step("[1/4] Initializing indexed relational SQL testbed database (10,000 live rows)...")
    db_meta = initialize_relational_db(DB_PATH)

    df_test = pd.read_csv(TEST_CSV, low_memory=False)
    # Realistic mixed workload: 90% benign queries + 10% SQLi attacks (800 requests per concurrency tier = 3,200 live HTTP requests)
    benign_pool = df_test[df_test["label"] == 0].iloc[:720]
    sqli_pool = df_test[df_test["label"] == 1].iloc[:80]
    workload_df = pd.concat([benign_pool, sqli_pool], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    payloads = []
    for idx, row in enumerate(workload_df.itertuples(index=False)):
        payloads.append({
            "query": str(row.query),
            "dialect": str(row.dialect),
            "label": int(row.label),
            "exec_sql": REAL_EXEC_SQL_TEMPLATES[idx % len(REAL_EXEC_SQL_TEMPLATES)],
        })

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    log_step(f"[2/4] Launching live FastAPI + Uvicorn + OpenTelemetry ASGI server on {base_url}...")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "live_proxy_server:app",
        "--app-dir", str(PROXY_APP_PATH.parent),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "error",
        "--no-access-log",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Poll health check until ready
        ready = False
        for _ in range(60):
            try:
                r = requests.get(f"{base_url}/health", timeout=0.5)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.25)
        if not ready:
            err_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"Uvicorn server failed to start: {err_out}")

        # Warmup 40 live HTTP requests
        with requests.Session() as warm_sess:
            for i in range(40):
                p = payloads[i]
                warm_sess.post(
                    f"{base_url}/v1/query",
                    json={"query": p["query"], "dialect": p["dialect"], "exec_sql": p["exec_sql"]},
                    timeout=5.0,
                )

        concurrency_levels = [1, 10, 50, 100]
        tier_reports = []

        log_step("[3/4] Running multi-concurrency live HTTP + OpenTelemetry + SQL DB load tests...")
        for c_level in concurrency_levels:
            tls_sess = threading.local()

            def send_one(item: dict) -> dict:
                sess = getattr(tls_sess, "session", None)
                if sess is None:
                    sess = requests.Session()
                    adapter = HTTPAdapter(pool_connections=c_level, pool_maxsize=max(32, c_level))
                    sess.mount("http://", adapter)
                    tls_sess.session = sess

                req_t0 = time.perf_counter_ns()
                resp = sess.post(
                    f"{base_url}/v1/query",
                    json={"query": item["query"], "dialect": item["dialect"], "exec_sql": item["exec_sql"]},
                    timeout=10.0,
                )
                e2e_ms = (time.perf_counter_ns() - req_t0) / 1e6
                body = resp.json()
                net_http_ms = max(0.0, e2e_ms - float(body["server_total_ms"]))
                return {
                    "status_code": resp.status_code,
                    "ok_error_check": resp.status_code in (200, 403),
                    "pred": 0 if resp.status_code == 200 else 1,
                    "label": item["label"],
                    "e2e_ms": e2e_ms,
                    "net_http_ms": net_http_ms,
                    "guard_ms": float(body["guard_ms"]),
                    "stage1_ms": float(body["stage1_ms"]),
                    "stage2_ms": float(body["stage2_ms"]),
                    "db_ms": float(body["db_ms"]),
                }

            wall_t0 = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=c_level) as pool:
                futs = [pool.submit(send_one, item) for item in payloads]
                outs = [f.result() for f in as_completed(futs)]
            wall_sec = (time.perf_counter_ns() - wall_t0) / 1e9

            e2e_arr = np.array([o["e2e_ms"] for o in outs], dtype=np.float64)
            net_arr = np.array([o["net_http_ms"] for o in outs], dtype=np.float64)
            ai_arr = np.array([o["guard_ms"] + o["stage1_ms"] + o["stage2_ms"] for o in outs], dtype=np.float64)
            db_arr = np.array([o["db_ms"] for o in outs], dtype=np.float64)
            err_cnt = sum(1 for o in outs if not o["ok_error_check"])

            preds = np.array([o["pred"] for o in outs], dtype=np.int64)
            labels = np.array([o["label"] for o in outs], dtype=np.int64)
            acc = float((preds == labels).mean())

            rps = len(outs) / max(1e-6, wall_sec)
            mean_e2e = float(np.mean(e2e_arr))
            littles_law_L = rps * (mean_e2e / 1000.0)

            log_step(
                f"   -> Concurrency C={c_level:3d}: RPS={rps:6.1f} | Mean E2E={mean_e2e:6.2f} ms | "
                f"AI={np.mean(ai_arr):.3f} ms | DB={np.mean(db_arr):.3f} ms | HTTP Errors={err_cnt}"
            )

            tier_reports.append({
                "concurrency": c_level,
                "requests": len(outs),
                "wall_time_sec": round(wall_sec, 4),
                "sustained_rps": round(rps, 2),
                "http_5xx_error_rate_pct": round((err_cnt / float(len(outs))) * 100.0, 4),
                "live_proxy_accuracy": round(acc, 6),
                "littles_law_effective_concurrency": round(littles_law_L, 2),
                "e2e_latency_ms": {
                    "mean": round(mean_e2e, 3),
                    "p50": round(float(np.percentile(e2e_arr, 50)), 3),
                    "p95": round(float(np.percentile(e2e_arr, 95)), 3),
                    "p99": round(float(np.percentile(e2e_arr, 99)), 3),
                },
                "otel_span_breakdown_mean_ms": {
                    "network_and_http_serialization_ms": round(float(np.mean(net_arr)), 3),
                    "cascade_ai_inspection_ms": round(float(np.mean(ai_arr)), 3),
                    "relational_db_execution_ms": round(float(np.mean(db_arr)), 3),
                },
            })

    finally:
        log_step("[4/4] Shutting down live Uvicorn proxy server cleanly and writing Table 10...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    tex_rows = []
    for r in tier_reports:
        e2e = r["e2e_latency_ms"]
        sp = r["otel_span_breakdown_mean_ms"]
        tex_rows.append(
            f"$C = {r['concurrency']}$ & {r['sustained_rps']:,.1f} & "
            f"{sp['network_and_http_serialization_ms']:.2f} & \\textbf{{{sp['cascade_ai_inspection_ms']:.3f}}} & "
            f"{sp['relational_db_execution_ms']:.3f} & {e2e['mean']:.2f} & {e2e['p50']:.2f} & "
            f"{e2e['p95']:.2f} & {e2e['p99']:.2f} & {r['http_5xx_error_rate_pct']:.2f}\\% \\\\"
        )
    joined_rows = "\n".join(tex_rows)

    tab10_tex = rf"""% Auto-generated from 100% Real Live FastAPI/Uvicorn + OpenTelemetry + SQL DB Concurrency Load Test (Exp-4)
\begin{{table*}}[htbp]
\centering
\caption{{End-to-end live HTTP ASGI proxy performance (\texttt{{FastAPI}} + \texttt{{Uvicorn}} + \texttt{{OpenTelemetry}} SDK + indexed relational SQL backend, 10,000 rows) across concurrent client loads ($N=800$ mixed queries per concurrency tier, 3,200 live HTTP requests total).}}
\label{{tab:exp4_live_concurrency}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrr}}
\toprule
\textbf{{Concurrency}} & \textbf{{Throughput (RPS)}} & \textbf{{Net/HTTP (ms)}} & \textbf{{AI Proxy (ms)}} & \textbf{{SQL DB (ms)}} & \textbf{{E2E Mean (ms)}} & \textbf{{E2E $p_{{50}}$}} & \textbf{{E2E $p_{{95}}$}} & \textbf{{E2E $p_{{99}}$}} & \textbf{{Error Rate}} \\
\midrule
{joined_rows}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    write_utf8_lf(TAB10_TEX, tab10_tex)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_execution_sec": round((time.perf_counter_ns() - t0) / 1e9, 3),
        "relational_db_backend": db_meta,
        "total_live_http_requests_measured": int(len(concurrency_levels) * len(payloads)),
        "concurrency_tiers": tier_reports,
        "sha256_verification": {
            "live_proxy_server_py": sha256_of(PROXY_APP_PATH),
            "tab10_exp4_live_proxy_concurrency_tex": sha256_of(TAB10_TEX),
        },
    }

    write_utf8_lf(LOG_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()