import re
import time
from typing import Dict, List, Tuple
import urllib.parse

# Representative OWASP CRS v4.x REQUEST-942-APPLICATION-ATTACK-SQLI regex ruleset (PL1 & PL2)
CRS_942_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("942100_libinjection_tautology", re.compile(r"(?:'\s*(?:or|and)\s*'[^']*'\s*=\s*'[^']*'|\"\s*(?:or|and)\s*\"[^\"]*\"\s*=\s*\"[^\"]*\"|\b(?:or|and)\s+\d+\s*=\s*\d+\b|\bor\s+true\b)", re.IGNORECASE)),
    ("942190_union_select", re.compile(r"\bunion\b[\s/\*!\d]*?(?:all[\s/\*!\d]*?)?\bselect\b", re.IGNORECASE)),
    ("942160_blind_sleep_benchmark", re.compile(r"\b(?:sleep\s*\(\s*\d+|pg_sleep\s*\(\s*\d+|benchmark\s*\(\s*\d+|waitfor\s+delay\b)", re.IGNORECASE)),
    ("942150_sql_func_names", re.compile(r"\b(?:extractvalue|updatexml|load_file|group_concat|information_schema|pg_catalog|xp_cmdshell|@@version)\b", re.IGNORECASE)),
    ("942200_comment_obfuscation", re.compile(r"(?:/\*!\d*.*?\*/|--\s*$|#\s*$|'\s*(?:/\*|--|#))", re.IGNORECASE | re.DOTALL)),
    ("942260_stacked_execution", re.compile(r";\s*(?:select|insert|update|delete|drop|alter|create|exec|execute|declare|waitfor)\b", re.IGNORECASE)),
    ("942330_classic_quote_probe", re.compile(r"(?:['\"]\s*(?:or|and|union|select|where|having)\b|\b(?:select|union)\b.{1,120}\bfrom\b.{1,120}\bwhere\b)", re.IGNORECASE | re.DOTALL)),
    ("942370_hex_char_encoding", re.compile(r"(?:\b0x[0-9a-f]{4,}\b|\b(?:chr|char|ascii)\s*\(\s*\d+\s*\))", re.IGNORECASE)),
    ("942430_restricted_char_anomaly", re.compile(r"(?:['\"].*?['\"].*?[=<>]|(?:\/\*.*?\*\/){2,})", re.IGNORECASE | re.DOTALL)),
]


class OwaspCrsV4Baseline:
    """Deterministic multi-rule regex engine executing full CRS 942 pass (simulating WAF inspection)."""
    def __init__(self, anomaly_threshold: int = 1):
        self.rules = CRS_942_PATTERNS
        self.anomaly_threshold = anomaly_threshold

    def inspect_query(self, raw_query: str) -> Tuple[int, float, int]:
        t0 = time.perf_counter_ns()
        if not isinstance(raw_query, str) or not raw_query:
            return 0, (time.perf_counter_ns() - t0) / 1e6, 0

        # Standard CRS transformation pipeline: urlDecodeUni + replaceNulls + compressWhitespace
        decoded = urllib.parse.unquote_plus(raw_query).replace("\x00", " ")
        normalized = re.sub(r"\s+", " ", decoded)

        matched_rules = 0
        # Evaluate all active 942 rules sequentially (anomaly scoring mode, as in ModSecurity CRS v4)
        for _, rx in self.rules:
            if rx.search(decoded) or rx.search(normalized):
                matched_rules += 1

        pred = 1 if matched_rules >= self.anomaly_threshold else 0
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        return pred, elapsed_ms, matched_rules