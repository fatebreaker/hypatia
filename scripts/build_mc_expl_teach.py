"""build_mc_expl_teach.py (10-27, docs/RL_PLAN.md H-MEET-ALL arm 5, B-GEN-X) -- teacher problems for the 13 weak calculators W.

Every MedCalc-Bench-Verified TRAIN row of W that survives build_medcalc_train.py's contamination filter (its patient note appears in neither
TEST version), at most PER_CALC per calculator (seed 0):
  * generated/medcalc_xw/train/<mcx-NNNN>     the problem exactly as build_medcalc_train.py writes it (explanation-free; the student's task)
  * generated/medcalc_expl_teach/train/<id>   the same problem with the row's own TRAIN 'Ground Truth Explanation' appended as a reference
                                              the teacher may use but must not mention or quote
  * benchmarks/mc_expl_teach_{a,b,c,d}         teacher wrappers (both benchmark trees; problems/ symlinks, the MedCalc grader)
Refuses to overwrite.   venv/bin/python scripts/build_mc_expl_teach.py
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
W = ["Caprini Score for Venous Thromboembolism (2005)", "Framingham Risk Score for Hard Coronary Heart Disease",
     "Wells' Criteria for Pulmonary Embolism", "APACHE II Score", "FeverPAIN Score for Strep Pharyngitis",
     "HEART Score for Major Cardiac Events", "Charlson Comorbidity Index (CCI)",
     "Centor Score (Modified/McIsaac) for Strep Pharyngitis", "Child-Pugh Score for Cirrhosis Mortality",
     "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk", "PERC Rule for Pulmonary Embolism",
     "PSI Score: Pneumonia Severity Index for CAP", "MELD Na (UNOS/OPTN)"]
EXPL_HEAD = ("\n\nWorked explanation of this problem (documentation for your own use; do not mention or quote this explanation in "
             "your answer -- compute the result in code from the patient note):\n")
XW, TEACH = f"{G}/medcalc_xw/train", f"{G}/medcalc_expl_teach/train"

for x in [XW, TEACH] + [f"{t}/mc_expl_teach_{h}" for t in TREES for h in "abcd"]:
    if os.path.exists(x):
        sys.exit(f"{x} exists -- not overwriting")
tr = pd.read_csv(fetch(*TRAIN), dtype=str).fillna("")
missing = [c for c in W if c not in set(tr["Calculator Name"].str.strip())]
assert not missing, f"calculator names not in the TRAIN csv: {missing}"
test_notes = set()
for repo, rev, fname in SOURCES.values():
    test_notes |= set(pd.read_csv(fetch(repo, rev, fname), dtype=str).fillna("")["Patient Note"].str.strip())
clean = tr[~tr["Patient Note"].str.strip().isin(test_notes)]
sel = clean[clean["Calculator Name"].str.strip().isin(W)]
sel = (sel.groupby("Calculator Name", group_keys=False)
       .apply(lambda g: g.sample(n=min(len(g), PER_CALC), random_state=SEED))
       .sample(frac=1.0, random_state=SEED).reset_index(drop=True))
assert sel["Ground Truth Explanation"].str.strip().ne("").all(), "a selected row has no explanation"
os.makedirs(XW); os.makedirs(TEACH)
made = []
for i, row in sel.iterrows():
    pid = f"mcx-{i:04d}"
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
                           "note_type": row["Note Type"], "pool": "medcalc_xw (H-MEET-ALL arm 5)"}}
    for root, t in ((XW, task), (TEACH, task + EXPL_HEAD + row["Ground Truth Explanation"].strip())):
        d = Path(root) / pid
        (d / "data_files").mkdir(parents=True)
        (d / "data_files" / "patient_note.txt").write_text(row["Patient Note"])
        (d / "eval_config.json").write_text(json.dumps({**cfg, "task": t}, indent=1))
    made.append(pid)
q = (len(made) + 3) // 4
for t in TREES:
    for k, h in enumerate("abcd"):
        w = f"{t}/mc_expl_teach_{h}"; os.makedirs(f"{w}/problems")
        for i in made[k * q:(k + 1) * q]:
            os.symlink(f"{TEACH}/{i}", f"{w}/problems/{i}")
        os.symlink(f"{R}/benchmarks/medcalc_ver/reference_grader.py", f"{w}/reference_grader.py")
counts = sel["Calculator Name"].value_counts().to_dict()
json.dump({"built": "2026-10-27", "registration": "docs/RL_PLAN.md H-MEET-ALL arm 5 (2026-10-27 01:42 UTC)", "train_rows": len(tr),
           "dropped_note_in_test": len(tr) - len(clean), "per_calc_cap": PER_CALC, "selected": len(made), "per_calculator": counts},
          open(f"{G}/medcalc_xw/MANIFEST.json", "w"), indent=1)
print(f"selected {len(made)} TRAIN rows of {len(W)} calculators ({counts}); dropped {len(tr) - len(clean)} rows whose note is in TEST; "
      f"wrappers a-d of about {q}")
