"""build_gen6mix.py (10-17, H-GEN27 as amended) -- the all-domain training pool with every benchmark domain represented:
gen4mix's 2,400 instances unchanged (the same 600 each of medcalc_ver, ehrsql2024, causal_mix, astro_mix; its MANIFEST lists
them) plus the clean MedAgentBench-style tasks (generated/mabgen2, all 257: the benchmark's task templates on patients no test
task names; scripts/mabgen_build2.py), 600 of the BioDSBench-style tasks (generated/bdsgen: cBioPortal studies BioDSBench does
not use; scripts/bdsgen_build.py) and the BioCoder tasks (generated/biocoder4: MedAgentGym's BioCoder training split, those
that grade cleanly offline; scripts/biocoder_build.py) -- at most 600 per pool, drawn with a fixed seed. Symlinks, as gen4mix.
Writes generated/gen6mix/train/<instance> and generated/gen6mix/MANIFEST.json. Refuses to overwrite an existing pool.
    python3 scripts/build_gen6mix.py
"""
import json
import os
import random
import sys

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
NEW = ("mabgen2", "bdsgen", "biocoder4")
N = 600
OUT = f"{G}/gen6mix"
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists -- not overwriting")
g4 = json.load(open(f"{G}/gen4mix/MANIFEST.json"))
chosen = {p: list(v) for p, v in g4["instances"].items()}
rng = random.Random(20261017)
for p in NEW:
    names = sorted(os.listdir(f"{G}/{p}/train"))
    chosen[p] = sorted(rng.sample(names, min(N, len(names))))
os.makedirs(f"{OUT}/train")
for p, names in chosen.items():
    for n in names:
        src, dst = f"{G}/{p}/train/{n}", f"{OUT}/train/{n}"
        assert os.path.isdir(src), src
        assert not os.path.exists(dst), dst        # instance names must not collide across pools
        os.symlink(src, dst)
json.dump({"built": "2026-10-17", "seed": 20261017, "per_pool": N, "pools": {p: len(v) for p, v in chosen.items()},
           "gen4mix_reused": sorted(g4["instances"]), "instances": chosen}, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print({p: len(v) for p, v in chosen.items()}, "->", len(os.listdir(f"{OUT}/train")), "instances in", OUT)
