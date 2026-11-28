import logging
import math
import re
from collections import Counter
from typing import Tuple

import numpy as np
import sqlglot
from sqlglot import TokenType

logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_TOK_LEN = 128
MAX_SQL_BYTES = 8192
MAX_PAREN_DEPTH = 32

# Deterministic mapping of key sqlglot structural token types
TRACKED_TOKEN_TYPES = [
    TokenType.SELECT, TokenType.UNION, TokenType.ALL, TokenType.FROM, TokenType.WHERE,
    TokenType.AND, TokenType.OR, TokenType.NOT, TokenType.XOR, TokenType.EQ,
    TokenType.NEQ, TokenType.LT, TokenType.LTE, TokenType.GT, TokenType.GTE,
    TokenType.LIKE, TokenType.ILIKE, TokenType.RLIKE, TokenType.IN, TokenType.IS,
    TokenType.NULL, TokenType.TRUE, TokenType.FALSE, TokenType.CASE, TokenType.WHEN,
    TokenType.THEN, TokenType.ELSE, TokenType.END, TokenType.JOIN, TokenType.LEFT,
    TokenType.RIGHT, TokenType.INNER, TokenType.OUTER, TokenType.CROSS, TokenType.ON,
    TokenType.GROUP_BY, TokenType.ORDER_BY, TokenType.HAVING, TokenType.LIMIT, TokenType.OFFSET,
    TokenType.INSERT, TokenType.INTO, TokenType.VALUES, TokenType.UPDATE, TokenType.SET,
    TokenType.DELETE, TokenType.DROP, TokenType.CREATE, TokenType.ALTER, TokenType.TRUNCATE,
    TokenType.EXEC, TokenType.EXECUTE, TokenType.COMMAND, TokenType.WITH, TokenType.CAST,
    TokenType.CONVERT, TokenType.SUBSTRING, TokenType.LPARE, TokenType.RPARE, TokenType.COMMA,
    TokenType.SEMICOLON, TokenType.DOT, TokenType.STAR, TokenType.SLASH, TokenType.PLUS,
    TokenType.DASH, TokenType.MOD, TokenType.AMP, TokenType.PIPE, TokenType.CARET,
    TokenType.COLON, TokenType.DCOLON, TokenType.PLACEHOLDER, TokenType.PARAMETER,
    TokenType.STRING, TokenType.NUMBER, TokenType.HEX_STRING, TokenType.BIT_STRING,
    TokenType.VAR, TokenType.COLUMN, TokenType.COMMENT,
] if hasattr(TokenType, "LPARE") else []

# Build version-agnostic TokenType lookup table safely
_BASE_TOKEN_NAMES = [
    "SELECT", "UNION", "ALL", "FROM", "WHERE", "AND", "OR", "NOT", "XOR",
    "EQ", "NEQ", "LT", "LTE", "GT", "GTE", "LIKE", "ILIKE", "RLIKE", "IN", "IS",
    "NULL", "TRUE", "FALSE", "CASE", "WHEN", "THEN", "ELSE", "END",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "ON",
    "GROUP_BY", "ORDER_BY", "HAVING", "LIMIT", "OFFSET",
    "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "DROP", "CREATE", "ALTER",
    "EXEC", "EXECUTE", "COMMAND", "WITH", "CAST", "L_PAREN", "R_PAREN",
    "COMMA", "SEMICOLON", "DOT", "STAR", "SLASH", "PLUS", "DASH", "MOD",
    "AMP", "PIPE", "CARET", "COLON", "DCOLON", "PLACEHOLDER", "PARAMETER",
    "STRING", "NUMBER", "HEX_STRING", "BIT_STRING", "VAR", "COMMENT",
]
TOKEN_NAME_TO_ID = {name: idx + 2 for idx, name in enumerate(_BASE_TOKEN_NAMES)}
PAD_TOK_ID = 0
UNK_TOK_ID = 1

# Reserved special semantic IDs for high-risk SQLi functions/identifiers inside VAR tokens
SPECIAL_VAR_TO_ID = {
    "SLEEP": len(_BASE_TOKEN_NAMES) + 2,
    "PG_SLEEP": len(_BASE_TOKEN_NAMES) + 3,
    "BENCHMARK": len(_BASE_TOKEN_NAMES) + 4,
    "WAITFOR": len(_BASE_TOKEN_NAMES) + 5,
    "DELAY": len(_BASE_TOKEN_NAMES) + 6,
    "INFORMATION_SCHEMA": len(_BASE_TOKEN_NAMES) + 7,
    "PG_CATALOG": len(_BASE_TOKEN_NAMES) + 8,
    "EXTRACTVALUE": len(_BASE_TOKEN_NAMES) + 9,
    "UPDATEXML": len(_BASE_TOKEN_NAMES) + 10,
    "CHR": len(_BASE_TOKEN_NAMES) + 11,
    "CHAR": len(_BASE_TOKEN_NAMES) + 12,
    "ASCII": len(_BASE_TOKEN_NAMES) + 13,
    "CONCAT": len(_BASE_TOKEN_NAMES) + 14,
    "GROUP_CONCAT": len(_BASE_TOKEN_NAMES) + 15,
    "LOAD_FILE": len(_BASE_TOKEN_NAMES) + 16,
    "XP_CMDSHELL": len(_BASE_TOKEN_NAMES) + 17,
    "VERSION": len(_BASE_TOKEN_NAMES) + 18,
}
STAGE2_VOCAB_SIZE = len(_BASE_TOKEN_NAMES) + 20

STRUCTURAL_FEATURE_NAMES = [
    "norm_byte_length",
    "max_paren_depth_norm",
    "shannon_entropy_norm",
    "comment_block_count",
    "mysql_versioned_comment_count",
    "line_comment_count",
    "semicolon_stacked_count",
    "single_quote_count",
    "double_quote_count",
    "hex_literal_count",
    "control_whitespace_ratio",
    "uppercase_transition_ratio",
    "token_count_norm",
    "union_select_flag",
    "boolean_or_and_count",
    "comparison_op_count",
    "tautology_pattern_flag",
    "time_delay_func_flag",
    "schema_probe_flag",
    "xml_error_func_flag",
    "char_encode_func_flag",
    "typecast_op_count",
    "null_byte_present_flag",
    "dos_guard_triggered_flag",
]
NUM_STRUCT_FEATS = len(STRUCTURAL_FEATURE_NAMES)

_MYSQL_VER_COMMENT_RE = re.compile(r"/\*!\d*(.*?)\*/", re.DOTALL)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"(--[^\n]*|#[^\n]*)")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_TAUTOLOGY_RE = re.compile(
    r"(\b(\d+)\s*=\s*\2\b|'([^']*)'\s*=\s*'\3'|\"([^\"]*)\"\s*=\s*\"\4\"|\bOR\s+1\b|\bOR\s+TRUE\b"
    r"|0x[0-9a-fA-F]+\s*=\s*0x[0-9a-fA-F]+|\bCHR\s*\(\d+\)\s*=\s*CHR\s*\(\d+\)|\bLIKE\b)",
    re.IGNORECASE,
)
_TIME_FN_RE = re.compile(r"\b(pg_sleep|sleep|benchmark|waitfor\s+delay)\b", re.IGNORECASE)
_SCHEMA_RE = re.compile(r"\b(information_schema|pg_catalog|sys\.tables|sqlite_master|@@version)\b", re.IGNORECASE)
_XML_ERR_RE = re.compile(r"\b(extractvalue|updatexml|floor\s*\(\s*rand)\b", re.IGNORECASE)
_CHAR_FN_RE = re.compile(r"\b(chr|char|ascii|ord|hex|unhex|substring|substr)\s*\(", re.IGNORECASE)
_CAST_RE = re.compile(r"(::\w+|\bcast\s*\(|\bconvert\s*\()", re.IGNORECASE)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def max_paren_depth(s: str) -> int:
    depth = 0
    best = 0
    for ch in s:
        if ch == "(":
            depth += 1
            if depth > best:
                best = depth
        elif ch == ")":
            depth = max(0, depth - 1)
    return best


def check_dos_guard(raw_sql: str) -> bool:
    if not isinstance(raw_sql, str):
        return False
    b_len = len(raw_sql.encode("utf-8", errors="ignore"))
    if b_len > MAX_SQL_BYTES:
        return True
    if max_paren_depth(raw_sql) > MAX_PAREN_DEPTH:
        return True
    return False


def extract_single_stage2_features(raw_sql: str, dialect: str = "mysql") -> Tuple[np.ndarray, np.ndarray]:
    tok_arr = np.zeros(MAX_TOK_LEN, dtype=np.int64)
    feat_arr = np.zeros(NUM_STRUCT_FEATS, dtype=np.float32)

    if not isinstance(raw_sql, str) or not raw_sql:
        return tok_arr, feat_arr

    has_null_byte = 1.0 if "\x00" in raw_sql else 0.0
    s = raw_sql.replace("\x00", " ").strip()
    if not s:
        feat_arr[22] = has_null_byte
        return tok_arr, feat_arr

    byte_len = len(s.encode("utf-8", errors="ignore"))
    p_depth = max_paren_depth(s)
    dos_flag = 1.0 if (byte_len > MAX_SQL_BYTES or p_depth > MAX_PAREN_DEPTH) else 0.0

    # Bound string slice for safe bounded-time feature & lexical analysis
    s_bounded = s[:MAX_SQL_BYTES]
    n_chars = max(1, len(s_bounded))

    ver_comments = len(_MYSQL_VER_COMMENT_RE.findall(s_bounded))
    blk_comments = len(_BLOCK_COMMENT_RE.findall(s_bounded))
    line_comments = len(_LINE_COMMENT_RE.findall(s_bounded))
    hex_lits = len(_HEX_RE.findall(s_bounded))
    ctrl_ws = sum(1 for c in s_bounded if c in ("\t", "\n", "\r", "\x0b", "\x0c"))

    alpha_chars = [c for c in s_bounded if c.isalpha()]
    case_transitions = 0
    if len(alpha_chars) > 1:
        case_transitions = sum(
            1 for i in range(1, len(alpha_chars))
            if alpha_chars[i].isupper() != alpha_chars[i - 1].isupper()
        )
    case_trans_ratio = case_transitions / max(1, len(alpha_chars) - 1)

    # De-obfuscate MySQL versioned comments and inline comment splits before sqlglot tokenization
    unwrapped = _MYSQL_VER_COMMENT_RE.sub(r" \1 ", s_bounded)
    unwrapped = _BLOCK_COMMENT_RE.sub(" ", unwrapped)

    sg_dialect = "postgres" if str(dialect).lower().startswith("postgres") else "mysql"
    try:
        tokens = sqlglot.tokenize(unwrapped, read=sg_dialect)
    except Exception:
        tokens = []

    has_union = False
    has_select_after_union = False
    bool_op_cnt = 0
    cmp_op_cnt = 0

    limit = min(len(tokens), MAX_TOK_LEN)
    for idx in range(limit):
        tok = tokens[idx]
        t_name = tok.token_type.name
        if t_name == "UNION":
            has_union = True
        elif t_name == "SELECT" and has_union:
            has_select_after_union = True

        if t_name in ("OR", "AND", "XOR"):
            bool_op_cnt += 1
        elif t_name in ("EQ", "NEQ", "LT", "LTE", "GT", "GTE", "LIKE", "ILIKE", "RLIKE"):
            cmp_op_cnt += 1

        if t_name == "VAR":
            up_txt = (tok.text or "").upper()
            tok_arr[idx] = SPECIAL_VAR_TO_ID.get(up_txt, TOKEN_NAME_TO_ID["VAR"])
        else:
            tok_arr[idx] = TOKEN_NAME_TO_ID.get(t_name, UNK_TOK_ID)

    # Inject explicit comment anomaly token marker if comments were stripped
    if (ver_comments > 0 or blk_comments > 0 or line_comments > 0) and limit < MAX_TOK_LEN:
        tok_arr[limit] = TOKEN_NAME_TO_ID["COMMENT"]

    feat_arr[0] = min(1.0, byte_len / 1024.0)
    feat_arr[1] = min(1.0, p_depth / float(MAX_PAREN_DEPTH))
    feat_arr[2] = min(1.0, shannon_entropy(s_bounded) / 6.5)
    feat_arr[3] = min(1.0, blk_comments / 10.0)
    feat_arr[4] = min(1.0, ver_comments / 5.0)
    feat_arr[5] = min(1.0, line_comments / 3.0)
    feat_arr[6] = min(1.0, s_bounded.count(";") / 4.0)
    feat_arr[7] = min(1.0, s_bounded.count("'") / 12.0)
    feat_arr[8] = min(1.0, s_bounded.count('"') / 12.0)
    feat_arr[9] = min(1.0, hex_lits / 4.0)
    feat_arr[10] = min(1.0, ctrl_ws / float(n_chars))
    feat_arr[11] = float(case_trans_ratio)
    feat_arr[12] = min(1.0, len(tokens) / float(MAX_TOK_LEN))
    feat_arr[13] = 1.0 if has_select_after_union else 0.0
    feat_arr[14] = min(1.0, bool_op_cnt / 8.0)
    feat_arr[15] = min(1.0, cmp_op_cnt / 8.0)
    feat_arr[16] = 1.0 if _TAUTOLOGY_RE.search(unwrapped) else 0.0
    feat_arr[17] = 1.0 if _TIME_FN_RE.search(unwrapped) else 0.0
    feat_arr[18] = 1.0 if _SCHEMA_RE.search(unwrapped) else 0.0
    feat_arr[19] = 1.0 if _XML_ERR_RE.search(unwrapped) else 0.0
    feat_arr[20] = 1.0 if _CHAR_FN_RE.search(unwrapped) else 0.0
    feat_arr[21] = min(1.0, len(_CAST_RE.findall(s_bounded)) / 4.0)
    feat_arr[22] = has_null_byte
    feat_arr[23] = dos_flag

    return tok_arr, feat_arr


def batch_extract_stage2(queries: list[str], dialects: list[str] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    n = len(queries)
    tok_mat = np.zeros((n, MAX_TOK_LEN), dtype=np.int64)
    feat_mat = np.zeros((n, NUM_STRUCT_FEATS), dtype=np.float32)
    if dialects is None:
        dialects = ["mysql"] * n
    for i, (q, d) in enumerate(zip(queries, dialects)):
        t_row, f_row = extract_single_stage2_features(q, d)
        tok_mat[i] = t_row
        feat_mat[i] = f_row
    return tok_mat, feat_mat