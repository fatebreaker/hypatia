"""build_qrdisc2.py (10-25, H-QR4-FG in docs/RL_PLAN.md; build_qrdisc.py with causal_disc2 for causal_disc) -- qrmix (causal_mix 2,000 + bdsgen 811; scripts/build_qrmix.py) plus every
causal_disc TRAIN instance (800 synthetic causal-discovery questions in QRData's format; scripts/causal_disc_build.py): QRData's
three kinds of question -- causal estimation, statistics on real data, causal discovery. No sampling. Symlinks to each
instance's real directory, as qrmix. Writes generated/qrdisc/train/<instance> and generated/qrdisc/MANIFEST.json. Refuses to
overwrite an existing pool.
    python3 scripts/build_qrdisc2.py
"""
import json
import os
import sys

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
POOLS = ("causal_mix", "bdsgen", "causal_disc2")
OUT = f"{G}/qrdisc2"
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
json.dump({"built": "2026-10-25", "pools": {p: len(v) for p, v in chosen.items()}, "instances": chosen},
          open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print({p: len(v) for p, v in chosen.items()}, "->", len(os.listdir(f"{OUT}/train")), "instances in", OUT)
