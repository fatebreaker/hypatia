"""EHRSQL-2024 grading (copied into benchmarks/ehrsql2024*/reference_grader.py by scripts/fetch_ehrsql2024.py).

The OFFICIAL shared-task metric (github.com/glee4810/ehrsql-2024, scoring_program/scoring.py + scoring_utils.py +
postprocessing.py), vendored function-for-function, not "improved":

  1. post_process_sql() the submitted query (whitespace, `current_time` -> '2100-12-31 23:59:00', ...). Two versions
     are vendored because the file changed after the shared task: "2024-03" is the one the published leaderboard was
     scored with (benchmarks/ehrsql2024), "v1.1.2" the current one (benchmarks/ehrsql2024_v112); the problem's
     grader config names which.
  2. The string 'null' is an ABSTENTION. Anything else is executed on the benchmark's MIMIC-IV-demo SQLite database;
     a query that raises becomes 'error_pred'.
  3. process_answer(): rows sorted, every cell rounded to 3 decimals if numeric and str()-ed, first 100 rows kept.
  4. Reliability score per item: +1 correct answer or correct abstention; 0 abstention on an answerable question;
     -1 (x penalty c) wrong result on an answerable question, or ANY non-abstention on an unanswerable one.
     RS(c) = mean over items; the shared task's primary metric is RS(10). scripts/ehrsql2024_score.py aggregates.

The gold side is the gold SQL executed once, at build time, by exactly this code; fetch_ehrsql2024.py asserts that the
result equals the release's own answer.json for every item before writing it into the config.

DEVIATIONS FROM THE OFFICIAL SCORER (none changes the score of a query that terminates and does not write):
  * the database is opened read-only (sqlite URI mode=ro), so a submitted DROP/DELETE cannot corrupt the one copy
    every later grade reads; the official scorer opened a throwaway copy read-write;
  * a query still running after EHRSQL_SQL_TIMEOUT seconds (default 120) is aborted and scored as 'error_pred';
    the official scorer had no limit, which would hang the run;
  * a JSON null submitted as the value is read as the abstention string 'null' (the official prediction file is
    JSON strings only, so it has no other meaning); 'NULL', 'None', '' etc. are executed as SQL, as officially.
"""
from __future__ import annotations

import functools
import hashlib
import os
import re
import sqlite3
import time
from ast import literal_eval
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "mimic_iv.sqlite"
SKIP = "null"

# --------------------------------------------------------------------------- #
# scoring_program/postprocessing.py -- the version scored in the 2024 shared task (commit 5ddaaa7 / e207f57)
# --------------------------------------------------------------------------- #
__current_time = "2100-12-31 23:59:00"
__precomputed_dict = {
    'temperature': (35.5, 38.1),
    'sao2': (95.0, 100.0),
    'heart rate': (60.0, 100.0),
    'respiration': (12.0, 18.0),
    'systolic bp': (90.0, 120.0),
    'diastolic bp': (60.0, 90.0),
    'mean bp': (60.0, 110.0)
}


def post_process_sql_2024(query):
    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')

    if "current_time" in query:
        query = query.replace("current_time", f"'{__current_time}'")
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr)
                                   + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in __precomputed_dict:
                vital_range = __precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr,
                                                                                     f"{vital_range[1]}")

    query = query.replace("%y", "%Y").replace('%j', '%J')

    return query


# --------------------------------------------------------------------------- #
# scoring_program/postprocessing.py -- current version (commit f9e1aa0, dataset v1.1.2)
# --------------------------------------------------------------------------- #
CURRENT_DATE = "2100-12-31"
CURRENT_TIME = "23:59:00"
NOW = f"{CURRENT_DATE} {CURRENT_TIME}"
PRECOMPUTED_DICT = __precomputed_dict
TIME_PATTERN = r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+ INTERVAL (\d+) (MONTH|YEAR|DAY)\)"


def convert_date_function(match):
    function = match.group(1)
    date = match.group(2)
    number = match.group(3)
    unit = match.group(4).lower()
    if number == '1':
        unit = unit.rstrip('s')
    else:
        unit += 's' if not unit.endswith('s') else ''
    sign = '-' if function == 'DATE_SUB' else '+'
    return f"datetime({date}, '{sign}{number} {unit}')"


def post_process_sql_v112(query):
    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')

    query = re.sub(TIME_PATTERN, convert_date_function, query)

    if "current_time" in query:
        query = query.replace("current_time", f"'{NOW}'")
    if "current_date" in query:
        query = query.replace("current_date", f"'{CURRENT_DATE}'")
    if "'now'" in query:
        query = query.replace("'now'", f"'{NOW}'")
    if "NOW()" in query:
        query = query.replace("NOW()", f"'{NOW}'")
    if "CURDATE()" in query:
        query = query.replace("CURDATE()", f"'{CURRENT_DATE}'")
    if "CURTIME()" in query:
        query = query.replace("CURTIME()", f"'{CURRENT_TIME}'")

    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr)
                                   + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in PRECOMPUTED_DICT:
                vital_range = PRECOMPUTED_DICT[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr,
                                                                                     f"{vital_range[1]}")

    query = query.replace("%y", "%Y").replace('%j', '%J')

    return query


POST_PROCESS = {"2024-03": post_process_sql_2024, "v1.1.2": post_process_sql_v112}


# --------------------------------------------------------------------------- #
# scoring_program/scoring_utils.py (identical at the shared task and at HEAD)
# --------------------------------------------------------------------------- #
def process_item(item):
    try:
        item = round(float(item), 3)
    except:  # noqa: E722  (official)
        pass
    return str(item)


def process_answer(ans):
    try:
        ans = literal_eval(ans)
    except:  # noqa: E722  (official)
        pass
    if type(ans) == str:  # noqa: E721  (official)
        return ans
    else:
        return str(sorted([[process_item(c) for c in row] for row in ans])[:100])  # check only up to 100th record


def execute_sql(sql, db_path=None, timeout=None):
    """Official execute_sql, but read-only and time-limited (see module docstring)."""
    db_path = Path(db_path or DB_PATH)
    timeout = float(timeout if timeout is not None else os.environ.get("EHRSQL_SQL_TIMEOUT", "120"))
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="ignore")
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        cur = con.cursor()
        return cur.execute(sql).fetchall()
    finally:
        con.close()


def execute_sql_wrapper(sql, tag, db_path=None, skip_indicator=SKIP):
    assert tag in ['real', 'pred']
    if sql != skip_indicator:
        try:
            result = execute_sql(sql, db_path)
        except:  # noqa: E722  (official)
            result = 'error_' + tag
        return process_answer(result)
    return skip_indicator


def item_reliability(ans_real, ans_pred) -> int:
    """scoring_utils.reliability_score for ONE item."""
    exec_acc = (ans_real == ans_pred)
    if ans_real != 'null' and exec_acc:
        return 1
    if ans_real != 'null' and ans_pred == 'null':
        return 0
    if ans_real != 'null' and not exec_acc:
        return -1
    if ans_real == 'null' and ans_pred != 'null':
        return -1
    return 1  # ans_real == 'null' and ans_pred == 'null'


def penalize(scores, penalty=1):
    return np.mean([score * penalty if score == -1 else score for score in scores])


def reliability_scores(item_scores: list[int]) -> dict:
    """The four numbers the official scorer prints, in percent."""
    n = len(item_scores)
    return {"RS0": penalize(item_scores, 0) * 100, "RS5": penalize(item_scores, 5) * 100,
            "RS10": penalize(item_scores, 10) * 100, "RSN": penalize(item_scores, n) * 100}


# --------------------------------------------------------------------------- #
# harness entry point
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_db(config: dict) -> None:
    want = ((config.get("grader") or {}).get("config") or {}).get("db_sha256")
    got = _sha256(str(DB_PATH))
    if want != got:
        raise RuntimeError(f"{DB_PATH}: sha256 {got} != {want} recorded at build time")


def predict_result(sql, spec: dict, db_path=None) -> str:
    """Submitted value -> the processed execution result the official scorer compares."""
    if sql is None:
        sql = SKIP
    if not isinstance(sql, str):
        sql = str(sql)
    sql = POST_PROCESS[spec.get("postprocess", "2024-03")](sql)
    return execute_sql_wrapper(sql, "pred", db_path)


def evaluate(config: dict, submission: dict) -> dict:
    """Harness entry point -- same contract as the other vendored graders."""
    grader = config.get("grader") or {}
    if grader.get("type") != "ehrsql2024_official":
        raise ValueError(f"unsupported grader type: {grader.get('type')}")
    gcfg = grader.get("config") or {}
    # The runner re-imports this module for every problem, so the per-grade check is the cheap one (size);
    # fetch_ehrsql2024.py --verify-only checks the full sha256 (check_db below).
    want = gcfg.get("db_bytes")
    if want and DB_PATH.stat().st_size != want:
        raise RuntimeError(f"{DB_PATH} is not the database this benchmark was built against (size mismatch)")
    keys = gcfg.get("keys") or {}
    obj = submission or {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    details, rel = [], []
    for k, spec in keys.items():
        pred = predict_result(obj.get(k, ""), spec)
        gold = spec["gold_result"]
        r = item_reliability(gold, pred)
        rel.append(r)
        details.append({"key": k, "ehrsql_id": spec.get("ehrsql_id"), "answerable": gold != SKIP,
                        "abstained": pred == SKIP, "reliability": r, "correct": r == 1,
                        "pred_result": pred[:300], "gold_result": gold[:300], "got": obj.get(k)})
    n = max(len(keys), 1)
    correct = sum(r == 1 for r in rel)
    return {"eval_id": config.get("id"), "grader_type": "ehrsql2024_official",
            "score": correct / n, "passed": correct == n, "correct": correct, "total": n,
            # per-item official reliability in {1, 0, -1}; one key per problem, so this is the item's score
            "reliability": rel[0] if len(rel) == 1 else rel, "details": details}
