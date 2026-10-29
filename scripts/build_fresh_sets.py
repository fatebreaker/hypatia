"""build_fresh_sets.py (10-25, docs/RL_PLAN.md H-CONFIRM) -- the confirmatory test's fresh benchmark wrappers, in BOTH trees (/project
benchmarks and the /work tree's /scratch4 copy), laid out like the existing subsets (problems/<id> symlinks into the full benchmark,
reference_grader.py symlinked, SUBSET.json):
  ehrsql2024_fresh867   EHRSQL-2024 test questions NOT in ehrsql2024_s300 (867 of 1,167)
  medcalc_ver_fresh880  MedCalc-Bench-Verified problems NOT in medcalc_ver_s220 (880 of 1,100)
Refuses to overwrite.   python3 scripts/build_fresh_sets.py"""
import json
import os
import sys

P = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs/benchmarks"
TREES = (P, "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/deeprs_benchmarks")
SETS = (("ehrsql2024_fresh867", "ehrsql2024", "ehrsql2024_s300", 867), ("medcalc_ver_fresh880", "medcalc_ver", "medcalc_ver_s220", 880))
plan = {}
for name, full, used, n in SETS:
    allp = sorted(os.listdir(f"{P}/{full}/problems"))
    seen = set(json.load(open(f"{P}/{used}/SUBSET.json")))
    fresh = [i for i in allp if i not in seen]
    assert len(fresh) == n and len(seen) + n == len(allp), (name, len(fresh), len(seen), len(allp))
    plan[name] = (full, fresh)
for tree in TREES:
    for name in plan:
        if os.path.exists(f"{tree}/{name}"):
            sys.exit(f"{tree}/{name} exists -- not overwriting")
for tree in TREES:
    for name, (full, fresh) in plan.items():
        d = f"{tree}/{name}"
        os.makedirs(f"{d}/problems")
        for i in fresh:
            os.symlink(f"{P}/{full}/problems/{i}", f"{d}/problems/{i}")
        os.symlink(f"{P}/{full}/reference_grader.py", f"{d}/reference_grader.py")
        json.dump(fresh, open(f"{d}/SUBSET.json", "w"))
print({k: len(v[1]) for k, v in plan.items()}, "written in", len(TREES), "trees")
