"""Adapt MedCalc-Bench into harness-compatible problems (healthcare benchmark; docs/RL_PLAN.md 2026-09-30).

MedCalc-Bench (Khandekar et al., NeurIPS 2024 D&B, https://arxiv.org/abs/2406.12036): 55 clinical calculators
(lab tests, risk / severity scores, dosage conversions, diagnosis criteria, dates), each question a real patient note
plus a calculation request, graded WITHOUT a judge. Published code-agent numbers exist in exactly our setting
(MedAgentGym, arXiv 2506.04405, test n = 1,046 of v1.0): gpt-4.1 86.2, o4-mini 78.5, gpt-4o 74.0, Qwen3-8B 54.6.

TWO VERSIONS, BOTH BUILT
-----------------------
  benchmarks/medcalc_ver   nsk7153/MedCalc-Bench-Verified (corrected labels; ~27% of v1.0 labels were found clinically
                           wrong, arXiv 2512.19691) -- the PRIMARY instrument.
  benchmarks/medcalc_orig  ncbi/MedCalc-Bench test_data_11_18_final.csv (original labels, open). The published frontier
                           numbers were run on ncbi/MedCalc-Bench-v1.0 (gated, n = 1,047); this file is its open
                           predecessor (n = 1,100), so comparisons to published numbers are CLOSE, NOT IDENTICAL -- say so.
Both are pinned by dataset revision below.

GRADING (reference_grader.py, the official rules by output type, not "improved")
  decimal  -- correct iff Lower Limit <= answer <= Upper Limit (the dataset's own +/-5% limits)
  integer  -- correct iff round(answer) == round(ground truth)
  date     -- correct iff the M/D/Y date equals the ground truth date
  gestational age -- correct iff (weeks, days) equals the ground truth pair
The answer is the first number (or date / pair) in the submitted value, as the official evaluator extracts it.

Run:
  venv/bin/python scripts/fetch_medcalc.py            # both versions, all 1,100 test items each
  venv/bin/python scripts/fetch_medcalc.py --limit 5  # smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, data_root  # noqa: E402

SOURCES = {
    "medcalc_ver": ("nsk7153/MedCalc-Bench-Verified", "58fc3de568bfc53a68f32327e58f60dc6eaa17f8", "test_data.csv"),
    "medcalc_orig": ("ncbi/MedCalc-Bench", "83a1a56d6b53f8d4c0e03f0ba9699becf524a595", "test_data_11_18_final.csv"),
}
CACHE = data_root() / "cache" / "medcalc"
GRADER = Path(__file__).resolve().parent / "medcalc_reference_grader.py"


def fetch(repo: str, rev: str, fname: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{repo.replace('/', '__')}__{rev[:8]}__{fname}"
    if not out.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{fname}"
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(out)
    return out


def kind(row) -> str:
    if "Gestational Age" in str(row["Calculator Name"]):
        return "gestational_age"
    return str(row["Output Type"]).strip().lower()          # decimal | integer | date


def build(name: str, limit: int | None) -> int:
    repo, rev, fname = SOURCES[name]
    df = pd.read_csv(fetch(repo, rev, fname), dtype=str).fillna("")
    if limit:
        df = df.head(limit)
    root = PROJECT_ROOT / "benchmarks" / name
    if (root / "problems").exists():
        shutil.rmtree(root / "problems")
    (root / "problems").mkdir(parents=True)
    shutil.copy2(GRADER, root / "reference_grader.py")
    for i, row in df.reset_index(drop=True).iterrows():
        pid = f"mc-{i:04d}"
        pdir = root / "problems" / pid
        (pdir / "data_files").mkdir(parents=True)
        (pdir / "data_files" / "patient_note.txt").write_text(row["Patient Note"])
        task = (
            "You are given a patient note (also saved as data_files/patient_note.txt) and a clinical calculation to "
            "perform. Extract the values you need from the note and compute the result.\n\n"
            f"Patient note:\n{row['Patient Note']}\n\n"
            f"Calculator: {row['Calculator Name']}\n"
            f"Question: {row['Question']}\n\n"
            'Return your answer as JSON: {"answer": {"' + pid + '-q1": <value>}}'
        )
        cfg = {
            "id": pid,
            "task": task,
            "data_files": ["patient_note.txt"],
            "ground_truth": {f"{pid}-q1": row["Ground Truth Answer"]},
            "grader": {"type": "medcalc_official", "config": {"keys": {f"{pid}-q1": {
                "kind": kind(row), "lower": row["Lower Limit"], "upper": row["Upper Limit"]}}}},
            "_provenance": {
                "benchmark": name, "dataset": f"{repo}@{rev}", "file": fname, "row_number": row["Row Number"],
                "calculator_id": row["Calculator ID"], "calculator": row["Calculator Name"],
                "category": row["Category"], "note_type": row["Note Type"], "paper": "arXiv:2406.12036",
            },
        }
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
    return len(df)


def verify(name: str) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "benchmarks" / name))
    import importlib
    g = importlib.import_module("reference_grader")
    importlib.reload(g)
    n = ok = 0
    for cfg_p in sorted((PROJECT_ROOT / "benchmarks" / name / "problems").glob("*/eval_config.json")):
        cfg = json.loads(cfg_p.read_text())
        n += 1
        ok += g.evaluate(cfg, {"answer": dict(cfg["ground_truth"])})["passed"]
        assert not g.evaluate(cfg, {"answer": {k: "banana" for k in cfg["ground_truth"]}})["passed"]
    assert ok == n, f"{name}: the ground truth passes only {ok}/{n}"
    print(f"[medcalc] {name}: {n} problems; ground truth passes {ok}/{n}; garbage fails all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for nm in SOURCES:
        print(f"[medcalc] {nm}: built {build(nm, a.limit)} problems")
        verify(nm)
