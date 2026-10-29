"""build_mc_expl_teach3.py (10-28, docs/RL_PLAN.md H-MEET-ALL arm 9, B-GEN-X3) -- the teacher problems of B-GEN-X3, for the five
calculators where B-GEN-X2 trails Claude Opus 5.5 by >= 1 problem on MedCalc s220 (APACHE II, Caprini, MELD Na, CHA2DS2-VASc, RCRI):
  * their explanation-guided teacher problems with no kept notebook in results/pods/distill27genx2/gen_x2.jsonl (mcx- from
    generated/medcalc_expl_teach, mcx2- from generated/medcalc_expl_teach2), for two more attempts;
  * the clean TRAIN rows of the five that no pool has used (generated/medcalc_ver, medcalc_xw, medcalc_xw2, by row number), at most
    PER_CALC per calculator (seed 0), written exactly as build_mc_expl_teach.py writes its problems --
      generated/medcalc_xw3/train/<mcx3-NNNN>            the student's task (explanation-free)
      generated/medcalc_expl_teach3/train/<mcx3-NNNN>    the same task + the row's own TRAIN 'Ground Truth Explanation';
  * teacher wrappers, both benchmark trees: mc_expl_teach4_{a,b,c,d} (one attempt of every problem) and mc_expl_teach5_{a,b,c,d} (the
    next attempt) -- an attempt of a problem lives in its own wrapper, so results do not collide;
  * generated/medcalc_xwall3/train: every student task (medcalc_xwall's links + the new problems), the --inst-root of scripts/build_bgen_x3.sh.
Refuses to overwrite.   venv/bin/python scripts/build_mc_expl_teach3.py
"""
from __future__ import annotations

import ast
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
W3 = ["APACHE II Score", "Caprini Score for Venous Thromboembolism (2005)", "MELD Na (UNOS/OPTN)",
      "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk", "Revised Cardiac Risk Index for Pre-Operative Risk"]
EXPL_HEAD = ("\n\nWorked explanation of this problem (documentation for your own use; do not mention or quote this explanation in "
             "your answer -- compute the result in code from the patient note):\n")   # build_mc_expl_teach.py's, verbatim
XW3, TEACH3, XWALL3 = f"{G}/medcalc_xw3/train", f"{G}/medcalc_expl_teach3/train", f"{G}/medcalc_xwall3/train"
XWALL = f"{G}/medcalc_xwall/train"


def prov(d):
    p = json.load(open(f"{d}/eval_config.json")).get("_provenance") or {}
    return ast.literal_eval(p) if isinstance(p, str) else p


for x in [XW3, TEACH3, XWALL3] + [f"{t}/mc_expl_teach{k}_{h}" for t in TREES for k in (4, 5) for h in "abcd"]:
    if os.path.exists(x):
        sys.exit(f"{x} exists -- not overwriting")
kept = {json.loads(l)["instance"] for l in open(f"{R}/results/pods/distill27genx2/gen_x2.jsonl") if l.strip()}
redo = []
for pool, pre in (("medcalc_expl_teach", "mcx-"), ("medcalc_expl_teach2", "mcx2-")):
    for i in sorted(os.listdir(f"{G}/{pool}/train")):
        if i.startswith(pre) and i not in kept and prov(f"{G}/{pool}/train/{i}")["calculator"] in W3:
            redo.append((i, f"{G}/{pool}/train/{i}"))
used = set()
for pool in ("medcalc_ver", "medcalc_xw", "medcalc_xw2"):
    for i in os.listdir(f"{G}/{pool}/train"):
        p = prov(f"{G}/{pool}/train/{i}")
        used.add((p.get("calculator"), str(p.get("row_number"))))
tr = pd.read_csv(fetch(*TRAIN), dtype=str).fillna("")
missing = [c for c in W3 if c not in set(tr["Calculator Name"].str.strip())]
assert not missing, f"calculator names not in the TRAIN csv: {missing}"
test_notes = set()
for repo, rev, fname in SOURCES.values():
    test_notes |= set(pd.read_csv(fetch(repo, rev, fname), dtype=str).fillna("")["Patient Note"].str.strip())
clean = tr[~tr["Patient Note"].str.strip().isin(test_notes)]
sel = clean[clean["Calculator Name"].str.strip().isin(W3)]
sel = sel[[(c.strip(), str(r)) not in used for c, r in zip(sel["Calculator Name"], sel["Row Number"])]]
sel = (sel.groupby("Calculator Name", group_keys=False)
       .apply(lambda g: g.sample(n=min(len(g), PER_CALC), random_state=SEED))
       .sample(frac=1.0, random_state=SEED).reset_index(drop=True))
assert sel["Ground Truth Explanation"].str.strip().ne("").all(), "a selected row has no explanation"
os.makedirs(XW3); os.makedirs(TEACH3)
new = []
for i, row in sel.iterrows():
    pid = f"mcx3-{i:04d}"
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
                           "note_type": row["Note Type"], "pool": "medcalc_xw3 (H-MEET-ALL arm 9)"}}
    for root, t in ((XW3, task), (TEACH3, task + EXPL_HEAD + row["Ground Truth Explanation"].strip())):
        d = Path(root) / pid
        (d / "data_files").mkdir(parents=True)
        (d / "data_files" / "patient_note.txt").write_text(row["Patient Note"])
        (d / "eval_config.json").write_text(json.dumps({**cfg, "task": t}, indent=1))
    new.append(pid)
os.makedirs(XWALL3)
for i in sorted(os.listdir(XWALL)):
    os.symlink(os.path.realpath(f"{XWALL}/{i}"), f"{XWALL3}/{i}")
for i in new:
    os.symlink(f"{XW3}/{i}", f"{XWALL3}/{i}")
probs = redo + [(i, f"{TEACH3}/{i}") for i in new]   # each wrapper set holds every problem once
q = (len(probs) + 3) // 4
for t in TREES:
    for k in (4, 5):
        for j, h in enumerate("abcd"):
            w = f"{t}/mc_expl_teach{k}_{h}"; os.makedirs(f"{w}/problems")
            for i, src in probs[j * q:(j + 1) * q]:
                os.symlink(src, f"{w}/problems/{i}")
            os.symlink(f"{R}/benchmarks/medcalc_ver/reference_grader.py", f"{w}/reference_grader.py")
counts = sel["Calculator Name"].value_counts().to_dict()
redo_counts = {}
for i, src in redo:
    c = prov(src)["calculator"]; redo_counts[c] = redo_counts.get(c, 0) + 1
json.dump({"built": "2026-10-28", "registration": "docs/RL_PLAN.md H-MEET-ALL arm 9 (2026-10-28 10:27 UTC)", "new_rows": counts,
           "new_problems": len(new), "redo_problems": len(redo), "redo_per_calculator": redo_counts, "per_wrapper": q,
           "wrapper_sizes": [len(probs[j * q:(j + 1) * q]) for j in range(4)]},
          open(f"{G}/medcalc_xw3/MANIFEST.json", "w"), indent=1)
print(f"{len(new)} new problems ({counts}); {len(redo)} unkept problems re-attempted ({redo_counts}); wrappers of up to {q}: "
      f"{[len(probs[j * q:(j + 1) * q]) for j in range(4)]}")
