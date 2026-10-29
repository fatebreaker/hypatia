"""MedCalc-Bench grading (copied into benchmarks/medcalc_*/reference_grader.py by scripts/fetch_medcalc.py).

The official rules by output type (MedCalc-Bench evaluate.py), not "improved":
  decimal          Lower Limit <= answer <= Upper Limit (the dataset's +/-5% limits)
  integer          round(answer) == round(ground truth)
  date             the M/D/Y date equals the ground-truth date
  gestational_age  (weeks, days) equals the ground-truth pair
The answer is the first number / date / pair found in the submitted value.
"""
from __future__ import annotations

import re
from datetime import datetime

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")
DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2,4})")


def _first_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = NUM.search(str(v).replace(",", ""))
    return float(m.group()) if m else None


def _date(v):
    m = DATE.search(str(v))
    if not m:
        return None
    mo, d, y = (int(x) for x in m.groups())
    if y < 100:
        y += 2000
    try:
        return datetime(y, mo, d).date()
    except ValueError:
        return None


def _pair(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        v = f"{v[0]} {v[1]}"
    s = str(v).lower()
    w = re.search(r"(\d+)\s*weeks?", s)
    d = re.search(r"(\d+)\s*days?", s)
    if w and d:
        return int(w.group(1)), int(d.group(1))
    nums = re.findall(r"\d+", s)
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


def grade_key(pred, gold, spec: dict) -> bool:
    k = spec.get("kind", "decimal")
    if pred is None or pred == "":
        return False
    if k == "date":
        p, g = _date(pred), _date(gold)
        return p is not None and p == g
    if k == "gestational_age":
        p, g = _pair(pred), _pair(gold)
        return p is not None and p == g
    p = _first_number(pred)
    if p is None:
        return False
    if k == "integer":
        g = _first_number(gold)
        return g is not None and round(p) == round(g)
    lo, hi = _first_number(spec.get("lower")), _first_number(spec.get("upper"))
    if lo is None or hi is None:
        g = _first_number(gold)
        lo, hi = sorted((g * 0.95, g * 1.05))
    return lo <= p <= hi


def evaluate(config: dict, submission: dict) -> dict:
    """Harness entry point -- same contract as the other vendored graders."""
    grader = config.get("grader") or {}
    if grader.get("type") != "medcalc_official":
        raise ValueError(f"unsupported grader type: {grader.get('type')}")
    gt = config.get("ground_truth") or {}
    keys = (grader.get("config") or {}).get("keys") or {}
    obj = submission or {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    details, correct = [], 0
    for k, spec in keys.items():
        ok = grade_key(obj.get(k, ""), gt.get(k), spec)
        correct += bool(ok)
        details.append({"key": k, "correct": bool(ok), "expected": gt.get(k), "got": obj.get(k)})
    n = max(len(keys), 1)
    return {"eval_id": config.get("id"), "grader_type": "medcalc_official",
            "score": correct / n, "passed": correct == n, "correct": correct, "total": n, "details": details}
