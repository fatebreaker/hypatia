"""MedCalc-Bench RL training pool (healthcare RL; docs/RL_PLAN.md 2026-09-30 H-MEDCALC-RL).

generated/medcalc_ver/train/<mct-XXXX>/ from MedCalc-Bench-Verified's TRAIN split, in the same problem format as
benchmarks/medcalc_ver (scripts/fetch_medcalc.py), graded by the same official rules (rl/reward.py rewards it through
the vendored grader).

  * CONTAMINATION: every train row whose patient note also appears in the TEST split (either version) is dropped
    (297 of 10,538 in Verified) -- the RL policy never sees a test note.
  * BALANCE: at most PER_CALC rows per calculator (55 calculators; raw counts run 16..927), drawn with a fixed seed, so
    the pool is not dominated by a few lab-test calculators.

Run:  venv/bin/python scripts/build_medcalc_train.py [--per-calc 40] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fetch_medcalc import SOURCES, fetch, kind  # noqa: E402
from paths import data_root  # noqa: E402

TRAIN = ("nsk7153/MedCalc-Bench-Verified", "58fc3de568bfc53a68f32327e58f60dc6eaa17f8", "train_data.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-calc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tr = pd.read_csv(fetch(*TRAIN), dtype=str).fillna("")
    test_notes = set()
    for repo, rev, fname in SOURCES.values():
        test_notes |= set(pd.read_csv(fetch(repo, rev, fname), dtype=str).fillna("")["Patient Note"].str.strip())
    clean = tr[~tr["Patient Note"].str.strip().isin(test_notes)]
    pool = (clean.groupby("Calculator Name", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), a.per_calc), random_state=a.seed))
            .sample(frac=1.0, random_state=a.seed).reset_index(drop=True))
    root = data_root() / "generated" / "medcalc_ver" / "train"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for i, row in pool.iterrows():
        pid = f"mct-{i:04d}"
        pdir = root / pid
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
        cfg = {"id": pid, "task": task, "data_files": ["patient_note.txt"],
               "ground_truth": {f"{pid}-q1": row["Ground Truth Answer"]},
               "grader": {"type": "medcalc_official", "config": {"keys": {f"{pid}-q1": {
                   "kind": kind(row), "lower": row["Lower Limit"], "upper": row["Upper Limit"]}}}},
               "_provenance": {"benchmark": "medcalc_ver_train", "dataset": f"{TRAIN[0]}@{TRAIN[1]}",
                               "row_number": row["Row Number"], "calculator_id": row["Calculator ID"],
                               "calculator": row["Calculator Name"], "category": row["Category"],
                               "note_type": row["Note Type"]}}
        (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[medcalc_train] {len(tr)} train rows; {len(tr) - len(clean)} dropped (note in test); pool {len(pool)} "
          f"({pool['Calculator Name'].nunique()} calculators, <= {a.per_calc} each) -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
