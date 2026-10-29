"""build_hard27mix.py (2026-10-21, H-HARD27) -- the 27B's HARD-TAIL training pool: instances of the clean training pools that the
27B both solved and failed in its own past RL rollouts (0 < passed < attempts, pooled over every Qwen3.6-27B run's rollouts.jsonl
under ckpt/, smoke runs -s9NN excluded), at most CAP per pool by a fixed-seed draw. Why: the 27B passes 86-99% of the ordinary
pools, so most GRPO groups are all-correct and carry no gradient (GEN27 and DOM27-QR: 6-7 of 8 groups dead).
Pools: medcalc_ver, ehrsql2024, causal_mix, astro_mix, kepreal_mix, mabgen2, mablab, bdsgen, biocoder4 (an instance found in
several is attributed to the first in that order). Symlinks, as build_gen6mix.py; refuses to overwrite.
Writes generated/hard27mix/train/<instance> and generated/hard27mix/MANIFEST.json (counts, the runs and line counts used, rates).
    python3 scripts/build_hard27mix.py
"""
import json
import os
import random
import re
import sys

ROOT = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s"
C, G = f"{ROOT}/ckpt", f"{ROOT}/generated"
POOLS = ("medcalc_ver", "ehrsql2024", "causal_mix", "astro_mix", "kepreal_mix", "mabgen2", "mablab", "bdsgen", "biocoder4")
CAP, SEED = 200, 20261021
OUT = f"{G}/hard27mix"
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists -- not overwriting")
pool_of = {}
for p in POOLS:
    for i in os.listdir(f"{G}/{p}/train"):
        pool_of.setdefault(i, p)
stat, used = {}, {}
for d in sorted(os.listdir(C)):
    if "q27base" not in d or re.search(r"-s9\d\d$", d):
        continue
    f = f"{C}/{d}/rollouts.jsonl"
    if not os.path.exists(f):
        continue
    n = 0
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        n += 1
        s = stat.setdefault(r["inst"], [0, 0])
        s[0] += bool(r.get("passed"))
        s[1] += 1
    used[d] = n
mixed = {p: sorted(i for i, (ok, n) in stat.items() if pool_of.get(i) == p and 0 < ok < n) for p in POOLS}
rng = random.Random(SEED)
chosen = {p: (sorted(rng.sample(v, CAP)) if len(v) > CAP else v) for p, v in mixed.items()}
os.makedirs(f"{OUT}/train")
for p, names in chosen.items():
    for i in names:
        src, dst = f"{G}/{p}/train/{i}", f"{OUT}/train/{i}"
        assert os.path.isdir(src), src
        os.symlink(src, dst)
json.dump({"built": "2026-10-21", "registration": "docs/RL_PLAN.md H-HARD27 (2026-10-21 13:57 UTC)", "cap_per_pool": CAP, "seed": SEED,
           "criterion": "0 < passed < attempts over all 27B training rollouts (smoke runs excluded)",
           "counts": {p: len(v) for p, v in chosen.items()}, "mixed_available": {p: len(v) for p, v in mixed.items()},
           "runs_used": used, "instances": chosen,
           "pass_rates": {i: stat[i] for v in chosen.values() for i in v}}, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print({p: len(v) for p, v in chosen.items()}, "->", len(os.listdir(f"{OUT}/train")), "instances in", OUT)
