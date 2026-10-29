"""build_qrmix.py (10-23, OPD4-QR-9 / QM4-KL in docs/RL_PLAN.md) -- the 4B's QRData training pool with both halves of QRData
represented: every causal_mix instance (2,000 synthetic causal designs: the pool of SCI4-KL and the OPD4-QR runs) plus every
bdsgen instance (811 BioDSBench-style statistics questions on cBioPortal studies BioDSBench does not use; scripts/bdsgen_build.py).
No sampling. Symlinks to each instance's real directory, as gen6mix. Writes generated/qrmix/train/<instance> and
generated/qrmix/MANIFEST.json. Refuses to overwrite an existing pool.
    python3 scripts/build_qrmix.py
"""
import json
import os
import sys

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
POOLS = ("causal_mix", "bdsgen")
OUT = f"{G}/qrmix"
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists -- not overwriting")
chosen = {p: sorted(os.listdir(f"{G}/{p}/train")) for p in POOLS}
os.makedirs(f"{OUT}/train")
for p, names in chosen.items():
    for n in names:
        src, dst = os.path.realpath(f"{G}/{p}/train/{n}"), f"{OUT}/train/{n}"
        assert os.path.isfile(f"{src}/eval_config.json"), src
        assert not os.path.exists(dst), dst        # instance names must not collide across pools
        os.symlink(src, dst)
json.dump({"built": "2026-10-23", "pools": {p: len(v) for p, v in chosen.items()}, "instances": chosen},
          open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print({p: len(v) for p, v in chosen.items()}, "->", len(os.listdir(f"{OUT}/train")), "instances in", OUT)
