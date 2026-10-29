"""build_gen4mix.py (10-17, H-GEN27) -- the all-domain training pool for one 27B trained with RL on every domain at once:
600 TRAINING-split instances from each of four pools, drawn with a fixed seed, as symlinks (as generated/clinmix was built):
  medcalc_ver (clinical calculation, MedCalc-Bench-Verified training split; no test note), ehrsql2024 (EHRSQL-2024 training
  questions; none equal to a test question), causal_mix (five synthetic causal designs), astro_mix (synthetic light curves).
Writes generated/gen4mix/train/<instance> -> generated/<pool>/train/<instance> and generated/gen4mix/MANIFEST.json.
Refuses to overwrite an existing pool.
    python3 scripts/build_gen4mix.py
"""
import json
import os
import random
import sys

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
POOLS = ("medcalc_ver", "ehrsql2024", "causal_mix", "astro_mix")
N = 600
OUT = f"{G}/gen4mix"
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists -- not overwriting")
rng = random.Random(20261017)
chosen = {}
for p in POOLS:
    names = sorted(os.listdir(f"{G}/{p}/train"))
    assert len(names) >= N, (p, len(names))
    chosen[p] = sorted(rng.sample(names, N))
os.makedirs(f"{OUT}/train")
for p, names in chosen.items():
    for n in names:
        dst = f"{OUT}/train/{n}"
        assert not os.path.exists(dst), dst        # instance names must not collide across pools
        os.symlink(f"{G}/{p}/train/{n}", dst)
json.dump({"built": "2026-10-17", "seed": 20261017, "per_pool": N, "pools": {p: len(v) for p, v in chosen.items()},
           "instances": chosen}, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print({p: len(v) for p, v in chosen.items()}, "->", len(os.listdir(f"{OUT}/train")), "instances in", OUT)
