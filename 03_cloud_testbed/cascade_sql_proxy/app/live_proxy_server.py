import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_models_colab" / "src" / "cascade_router"))

from cascade_onnx_engine import CascadeSQLOnnxEngine

# Initialize authentic OpenTelemetry SDK TracerProvider
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("cascade_sql.serverless_proxy")

ROUTER_CFG = ROOT / "02_models_colab" / "exported_onnx" / "router_thresholds.json"
DB_PATH = ROOT / "03_cloud_testbed" / "sql_init" / "live_testbed_relational.db"

cfg = json.loads(ROUTER_CFG.read_text(encoding="utf-8"))
engine = CascadeSQLOnnxEngine(
    cfg["tau_low"], cfg["tau_high"], cfg["tau_stage2"], window_mode="strided_multi_window"
)

_tls = threading.local()


def get_ro_db_conn() -> sqlite3.Connection:
    conn = getattr(_tls, "conn", None)
    if conn is None:
        uri = f"file:{DB_PATH.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=10.0)
        _tls.conn = conn
    return conn


app = FastAPI(title="CASCADE-SQL Serverless Database Security Proxy", version="1.0.0")


class SQLProxyRequest(BaseModel):
    query: str
    dialect: str = "postgresql"
    exec_sql: str | None = None


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_mode": engine.window_mode}


@app.post("/v1/query")
def proxy_query_endpoint(req: SQLProxyRequest):
    t_req_start = time.perf_counter_ns()
    with tracer.start_as_current_span("cascade_proxy.request") as root_span:
        with tracer.start_as_current_span("cascade_proxy.ai_inspection") as ai_span:
            verdict = engine.inspect_single_query_timed(req.query, req.dialect)
            ai_span.set_attribute("cascade.route", verdict["route"])
            ai_span.set_attribute("cascade.pred", verdict["pred"])

        if verdict["pred"] == 1:
            root_span.set_attribute("http.status_code", 403)
            server_ms = (time.perf_counter_ns() - t_req_start) / 1e6
            return JSONResponse(
                status_code=403,
                content={
                    "allowed": False,
                    "route": verdict["route"],
                    "guard_ms": verdict["guard_ms"],
                    "stage1_ms": verdict["stage1_ms"],
                    "stage2_ms": verdict["stage2_ms"],
                    "db_ms": 0.0,
                    "server_total_ms": server_ms,
                },
            )

        # Execute real relational SQL query on backend database for allowed benign traffic
        db_t0 = time.perf_counter_ns()
        rows_returned = 0
        with tracer.start_as_current_span("cascade_proxy.db_execute") as db_span:
            conn = get_ro_db_conn()
            target_sql = req.exec_sql if req.exec_sql else "SELECT user_id, email, tier FROM enterprise_users WHERE active = 1 LIMIT 5"
            cur = conn.execute(target_sql)
            rows = cur.fetchall()
            rows_returned = len(rows)
            db_span.set_attribute("db.rows_returned", rows_returned)
        db_ms = (time.perf_counter_ns() - db_t0) / 1e6

        root_span.set_attribute("http.status_code", 200)
        server_ms = (time.perf_counter_ns() - t_req_start) / 1e6
        return {
            "allowed": True,
            "rows": rows_returned,
            "route": verdict["route"],
            "guard_ms": verdict["guard_ms"],
            "stage1_ms": verdict["stage1_ms"],
            "stage2_ms": verdict["stage2_ms"],
            "db_ms": db_ms,
            "server_total_ms": server_ms,
        }