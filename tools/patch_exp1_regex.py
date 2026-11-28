from pathlib import Path

p = Path("04_evaluation/exp1_sota_baselines/evaluate_exp1_sota.py")
txt = p.read_text(encoding="utf-8")

old_re = """    _TOK_SPEC = re.compile(
        r\"\"\"
        (?P<c>/\\*!\\d*.*?\\*/|/\\*.*?\\*/|--[^\\n]*|#[^\\n]*) |
        (?P<s>'(?:''|\\\\'|[^'])*'|"(?:""|\\\\"|[^"])*") |
        (?P<n>\\b0x[0-9a-fA-F]+\\b|\\b\\d+(?:\\.\\d+)?\\b) |
        (?P<u>\\bUNION\\b) |
        (?P<f>\\b(?:SLEEP|PG_SLEEP|BENCHMARK|EXTRACTVALUE|UPDATEXML|CHR|CHAR|ASCII|CONCAT|LOAD_FILE|CONVERT|CAST|COUNT|USER|VERSION|DATABASE)\\s*\\() |
        (?P<k>\\b(?:SELECT|FROM|WHERE|INSERT|INTO|VALUES|UPDATE|SET|DELETE|DROP|ALTER|CREATE|EXEC|EXECUTE|WAITFOR|DELAY|HAVING|ORDER|GROUP|BY|LIMIT|CASE|WHEN|THEN|ELSE|END|INFORMATION_SCHEMA|PG_CATALOG)\\b) |
        (?P<l>\\b(?:AND|OR|XOR|NOT|IS|LIKE|RLIKE|ILIKE|IN)\\b) |
        (?P<v>@@\\w+|@\\w+|\\b[a-zA-Z_]\\w*\\b) |
        (?P<o>[=<>!~|&+-/*^]+) |
        (?P<p>[(),;])
        \"\"\",
        re.IGNORECASE | re.VERBOSE | re.DOTALL,
    )"""

new_re = """    _TOK_SPEC = re.compile(
        "|".join([
            r"(?P<c>/\\*!\\d*.*?\\*/|/\\*.*?\\*/|--[^\\n]*|#[^\\n]*)",
            r"(?P<s>'(?:''|\\\\'|[^'])*'|\\"(?:\\"\\"|\\\\\\"|[^\\"])*\\")",
            r"(?P<n>\\b0x[0-9a-fA-F]+\\b|\\b\\d+(?:\\.\\d+)?\\b)",
            r"(?P<u>\\bUNION\\b)",
            r"(?P<f>\\b(?:SLEEP|PG_SLEEP|BENCHMARK|EXTRACTVALUE|UPDATEXML|CHR|CHAR|ASCII|CONCAT|LOAD_FILE|CONVERT|CAST|COUNT|USER|VERSION|DATABASE)\\s*\\()",
            r"(?P<k>\\b(?:SELECT|FROM|WHERE|INSERT|INTO|VALUES|UPDATE|SET|DELETE|DROP|ALTER|CREATE|EXEC|EXECUTE|WAITFOR|DELAY|HAVING|ORDER|GROUP|BY|LIMIT|CASE|WHEN|THEN|ELSE|END|INFORMATION_SCHEMA|PG_CATALOG)\\b)",
            r"(?P<l>\\b(?:AND|OR|XOR|NOT|IS|LIKE|RLIKE|ILIKE|IN)\\b)",
            r"(?P<v>@@\\w+|@\\w+|\\b[a-zA-Z_]\\w*\\b)",
            r"(?P<o>[=<>!~|&+\\-/*^]+)",
            r"(?P<p>[(),;])",
        ]),
        re.IGNORECASE | re.DOTALL,
    )"""

assert old_re in txt, "Target block not found!"
txt = txt.replace(old_re, new_re, 1)
with p.open("w", encoding="utf-8", newline="\n") as f:
    f.write(txt)
print("[OK] Patched _TOK_SPEC regex cleanly (removed re.VERBOSE '#' collision).")