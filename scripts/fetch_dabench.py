"""Adapt InfiAgent-DABench into harness-compatible problems.

DABench (Hu et al., ICML 2024, arXiv:2401.05507; github.com/InfiAgent/InfiAgent, Apache-2.0) is 257
closed-form data-analysis questions over 68 real CSV files -- summary statistics, correlation,
distributions, feature engineering, preprocessing, outliers and machine learning -- each with one or
more named answers.

WHY THIS BENCHMARK
------------------
It is a third data-analysis benchmark, beside QRData (statistical and causal questions) and
CauSciBench (causal effects), and its concept labels let the paper's "fine-tuning keeps what its
data exercise" be tested on questions no corpus was built for: e.g. wrangling-style preprocessing
and feature engineering, which the data-wrangling corpus exercises and the genomics corpus does not.

GRADING
-------
Ported from the official examples/DA-Agent/eval_closed_form.py: a named answer is correct if its
string equals the label or, as numbers, they differ by less than 1e-6; a question is solved when
every named answer is correct ("accuracy by question", the benchmark's headline metric). The
official agent answers "@name[value]"; ours answers the same names in the harness's JSON, so only
the extraction differs, not the comparison.

Run:
  venv/bin/python scripts/fetch_dabench.py              # all 257
  venv/bin/python scripts/fetch_dabench.py --verify-only
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/InfiAgent/InfiAgent/main/examples/DA-Agent/data"
CACHE = Path("/scratch4/workspace/hieutran_umass_edu-deeprs-bench/dabench")
ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "dabench"

GRADER = '''"""DABench grading, ported in behaviour from the official examples/DA-Agent/eval_closed_form.py:
a named answer is correct if its string equals the label or, as numbers, they differ by less than
1e-6; the question is passed when every named answer is correct (accuracy by question)."""
from __future__ import annotations


def is_equal(response, label) -> bool:
    if response is None:
        return False
    response = response if isinstance(response, str) else (repr(response) if isinstance(response, float) else str(response))
    response = response.strip()
    if response == label:
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except (TypeError, ValueError):
        return False


def evaluate(config: dict, submission: dict) -> dict:
    """Harness entry point -- same contract as the other vendored graders."""
    grader = config.get("grader") or {}
    if grader.get("type") != "dabench_official":
        raise ValueError(f"unsupported grader type: {grader.get('type')}")
    gt = config.get("ground_truth") or {}
    obj = submission or {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    details = [{"key": k, "correct": is_equal(obj.get(k), v), "expected": v, "got": obj.get(k)} for k, v in gt.items()]
    correct = sum(d["correct"] for d in details)
    n = max(len(gt), 1)
    return {"eval_id": config.get("id"), "grader_type": "dabench_official", "score": correct / n,
            "passed": correct == n, "correct": correct, "total": n, "details": details}


def grade(submission: dict, cfg: dict) -> dict:
    return evaluate(cfg, submission)
'''


def fetch() -> None:
    (CACHE / "tables").mkdir(parents=True, exist_ok=True)
    for f in ("da-dev-questions.jsonl", "da-dev-labels.jsonl"):
        if not (CACHE / f).exists():
            urllib.request.urlretrieve(f"{REPO}/{f}", CACHE / f)
    for q in (json.loads(line) for line in open(CACHE / "da-dev-questions.jsonl")):
        t = CACHE / "tables" / q["file_name"]
        if not t.exists():
            urllib.request.urlretrieve(f"{REPO}/da-dev-tables/{urllib.request.quote(q['file_name'])}", t)


def build() -> int:
    fetch()
    qs = [json.loads(line) for line in open(CACHE / "da-dev-questions.jsonl")]
    labels = {r["id"]: r["common_answers"] for r in (json.loads(line) for line in open(CACHE / "da-dev-labels.jsonl"))}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "reference_grader.py").write_text(GRADER)
    made = 0
    for q in qs:
        pid = f"da-{q['id']:03d}"
        names = [a[0] for a in labels[q["id"]]]
        assert names == re.findall(r"@(\w+)\[", q["format"]) or set(names) <= set(re.findall(r"@(\w+)\[", q["format"])), pid
        pdir = ROOT / "problems" / pid
        (pdir / "data_files").mkdir(parents=True, exist_ok=True)
        if not (pdir / "data_files" / q["file_name"]).exists():
            shutil.copy2(CACHE / "tables" / q["file_name"], pdir / "data_files" / q["file_name"])
        fields = ", ".join(f'"{n}": <value>' for n in names)
        task = ("You are given a data file and a question about it.\n"
                "Answer the question by computing the result from the data.\n\n"
                f"File in data_files/:\n  {q['file_name']}\n\n"
                f"Question: {q['question']}\n\n"
                f"Constraints: {q['constraints']}\n\n"
                f"Answer format: {q['format']}\n\n"
                "Report each @name[...] of the answer format as a field of the JSON answer, its value formatted "
                "as the answer format specifies. Return your answer as JSON: {\"answer\": {" + fields + "}}")
        cfg = {"id": pid, "task": task, "data_files": [q["file_name"]],
               "ground_truth": {a[0]: a[1] for a in labels[q["id"]]},
               "grader": {"type": "dabench_official", "config": {}},
               "_provenance": {"benchmark": "InfiAgent-DABench (dev)", "paper": "arXiv:2401.05507",
                               "concepts": q["concepts"], "level": q["level"], "dabench_id": q["id"]}}
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        made += 1
    return made


def verify() -> None:
    probs = sorted((ROOT / "problems").iterdir())
    bad = [f for p in probs for f in (p / "data_files").iterdir() if f.suffix.lower() != ".csv"]
    assert not bad, f"non-CSV files reached data_files/: {bad[:5]}"
    sys.path.insert(0, str(ROOT))
    import reference_grader as g  # noqa: E402
    for p in probs:
        cfg = json.loads((p / "eval_config.json").read_text())
        assert g.evaluate(cfg, {"answer": cfg["ground_truth"]})["passed"], p.name
        assert not g.evaluate(cfg, {"answer": {}})["passed"], p.name
    print(f"verify OK: {len(probs)} problems, CSV-only data, gold answers pass their own grader")


if __name__ == "__main__":
    if "--verify-only" not in sys.argv:
        print(f"built {build()} problems in {ROOT}")
    verify()
