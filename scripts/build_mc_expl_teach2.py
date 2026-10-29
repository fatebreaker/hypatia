"""build_mc_expl_teach2.py (10-27, docs/RL_PLAN.md H-MEET-ALL arm 7, B-GEN-X2) -- the teacher problems of B-GEN-X2:
  * the five calculators where B-GEN-X still trails Claude Opus 5.5 (SOFA, MME, Wells' DVT, RCRI, Delta Gap): every clean TRAIN row
    (build_medcalc_train.py's filter: the patient note appears in neither TEST version), at most PER_CALC per calculator (seed 0),
    written exactly as build_mc_expl_teach.py writes its 13 calculators --
      generated/medcalc_xw2/train/<mcx2-NNNN>            the student's task (explanation-free)
      generated/medcalc_expl_teach2/train/<mcx2-NNNN>    the same task + the row's own TRAIN 'Ground Truth Explanation';
  * the 831 problems of B-GEN-X's 13 calculators with no kept notebook (generated/medcalc_expl_teach/train minus the mcx-
    instances in results/pods/distill27genx/gen_x.jsonl), for two more attempts;
  * teacher wrappers, both benchmark trees: mc_expl_teach2_{a,b,c,d} (attempt 2 of the 831 + attempt 1 of the new problems) and
    mc_expl_teach3_{a,b,c,d} (attempt 3 + attempt 2) -- an attempt of a problem lives in its own wrapper, so results do not collide;
  * generated/medcalc_xwall/train: the student tasks of both sets (symlinks), the --inst-root of scripts/build_bgen_x2.sh.
Refuses to overwrite.   venv/bin/python scripts/build_mc_expl_teach2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_medcalc import SOURCES, fetch, kind  # noqa: E402

R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"
G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
TREES = (f"{R}/benchmarks", "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/deeprs_benchmarks")
TRAIN = ("nsk7153/MedCalc-Bench-Verified", "58fc3de568bfc53a68f32327e58f60dc6eaa17f8", "train_data.csv")
PER_CALC, SEED = 120, 0
W2 = ["Sequential Organ Failure Assessment (SOFA) Score", "Morphine Milligram Equivalents (MME) Calculator", "Wells' Criteria for DVT",
      "Revised Cardiac Risk Index for Pre-Operative Risk", "Delta Gap"]
EXPL_HEAD = ("\n\nWorked explanation of this problem (documentation for your own use; do not mention or quote this explanation in "
             "your answer -- compute the result in code from the patient note):\n")   # build_mc_expl_teach.py's, verbatim
XW2, TEACH2, XWALL = f"{G}/medcalc_xw2/train", f"{G}/medcalc_expl_teach2/train", f"{G}/medcalc_xwall/train"
TEACH1, XW1 = f"{G}/medcalc_expl_teach/train", f"{G}/medcalc_xw/train"

for x in [XW2, TEACH2, XWALL] + [f"{t}/mc_expl_teach{k}_{h}" for t in TREES for k in (2, 3) for h in "abcd"]:
    if os.path.exists(x):
        sys.exit(f"{x} exists -- not overwriting")
kept = {json.loads(l)["instance"] for l in open(f"{R}/results/pods/distill27genx/gen_x.jsonl") if l.strip()}
redo = sorted(i for i in os.listdir(TEACH1) if i.startswith("mcx-") and i not in kept)
tr = pd.read_csv(fetch(*TRAIN), dtype=str).fillna("")
missing = [c for c in W2 if c not in set(tr["Calculator Name"].str.strip())]
assert not missing, f"calculator names not in the TRAIN csv: {missing}"
test_notes = set()
for repo, rev, fname in SOURCES.values():
    test_notes |= set(pd.read_csv(fetch(repo, rev, fname), dtype=str).fillna("")["Patient Note"].str.strip())
clean = tr[~tr["Patient Note"].str.strip().isin(test_notes)]
sel = clean[clean["Calculator Name"].str.strip().isin(W2)]
sel = (sel.groupby("Calculator Name", group_keys=False)
       .apply(lambda g: g.sample(n=min(len(g), PER_CALC), random_state=SEED))
       .sample(frac=1.0, random_state=SEED).reset_index(drop=True))
assert sel["Ground Truth Explanation"].str.strip().ne("").all(), "a selected row has no explanation"
os.makedirs(XW2); os.makedirs(TEACH2)
new = []
for i, row in sel.iterrows():
    pid = f"mcx2-{i:04d}"
    task = (
        "You are given a patient note (also saved as data_files/patient_note.txt) and a clinical calculation to "
        "perform. Extract the values you need from the note and compute the result.\n\n"
        f"Patient note:\n{row['Patient Note']}\n\n"
        f"Calculator: {row['Calculator Name']}\n"
        f"Question: {row['Question']}\n\n"
        'Return your answer as JSON: {"answer": {"' + pid + '-q1": <value>}}'
    )   # build_medcalc_train.py's task, verbatim
    cfg = {"id": pid, "task": task, "data_files": ["patient_note.txt"],
           "ground_truth": {f"{pid}-q1": row["Ground Truth Answer"]},
           "grader": {"type": "medcalc_official", "config": {"keys": {f"{pid}-q1": {
               "kind": kind(row), "lower": row["Lower Limit"], "upper": row["Upper Limit"]}}}},
           "_provenance": {"benchmark": "medcalc_ver_train", "dataset": f"{TRAIN[0]}@{TRAIN[1]}",
                           "row_number": row["Row Number"], "calculator_id": row["Calculator ID"],
                           "calculator": row["Calculator Name"], "category": row["Category"],
                           "note_type": row["Note Type"], "pool": "medcalc_xw2 (H-MEET-ALL arm 7)"}}
    for root, t in ((XW2, task), (TEACH2, task + EXPL_HEAD + row["Ground Truth Explanation"].strip())):
        d = Path(root) / pid
        (d / "data_files").mkdir(parents=True)
        (d / "data_files" / "patient_note.txt").write_text(row["Patient Note"])
        (d / "eval_config.json").write_text(json.dumps({**cfg, "task": t}, indent=1))
    new.append(pid)
os.makedirs(XWALL)
for i in sorted(os.listdir(XW1)):
    os.symlink(f"{XW1}/{i}", f"{XWALL}/{i}")
for i in new:
    os.symlink(f"{XW2}/{i}", f"{XWALL}/{i}")
probs = [(i, f"{TEACH1}/{i}") for i in redo] + [(i, f"{TEACH2}/{i}") for i in new]   # each wrapper set holds every problem once
q = (len(probs) + 3) // 4
for t in TREES:
    for k in (2, 3):
        for j, h in enumerate("abcd"):
            w = f"{t}/mc_expl_teach{k}_{h}"; os.makedirs(f"{w}/problems")
            for i, src in probs[j * q:(j + 1) * q]:
                os.symlink(src, f"{w}/problems/{i}")
            os.symlink(f"{R}/benchmarks/medcalc_ver/reference_grader.py", f"{w}/reference_grader.py")
counts = sel["Calculator Name"].value_counts().to_dict()
json.dump({"built": "2026-10-27", "registration": "docs/RL_PLAN.md H-MEET-ALL arm 7 (2026-10-27 22:47 UTC)", "new_calculators": counts,
           "new_problems": len(new), "redo_problems": len(redo), "per_wrapper": q,
           "wrapper_sizes": [len(probs[j * q:(j + 1) * q]) for j in range(4)]},
          open(f"{G}/medcalc_xw2/MANIFEST.json", "w"), indent=1)
print(f"{len(new)} new problems ({counts}); {len(redo)} B-GEN-X problems to re-attempt; wrappers of up to {q}: "
      f"{[len(probs[j * q:(j + 1) * q]) for j in range(4)]}")
