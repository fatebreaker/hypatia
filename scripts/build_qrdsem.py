"""build_qrdsem.py (10-27, docs/RL_PLAN.md H-SEM) -- the two H-SEM training pools, symlinks to each instance's real directory as
build_qrdisc2.py:
  generated/qrdsem   qrdisc2 (causal_mix 2,000 + bdsgen 811 + causal_disc2 800) + causal_sem train (800)       -> QDS9 (9B)
  generated/qrrsem   qrreal89 (89) + causal_sem train causal_sem_train_00000-00088 (89; equal parts)             -> QRS4 (4B)
Writes <pool>/train/<instance> and <pool>/MANIFEST.json. Refuses to overwrite an existing pool.
    python3 scripts/build_qrdsem.py
"""
import json
import os
import sys

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
SEM = sorted(os.listdir(f"{G}/causal_sem/train"))
assert len(SEM) == 800, len(SEM)
POOLS = {"qrdsem": {"qrdisc2": sorted(os.listdir(f"{G}/qrdisc2/train")), "causal_sem": SEM},
         "qrrsem": {"qrreal89": sorted(os.listdir(f"{G}/qrreal89/train")), "causal_sem": SEM[:89]}}
for out in POOLS:
    if os.path.exists(f"{G}/{out}"):
        sys.exit(f"{G}/{out} exists -- not overwriting")
for out, chosen in POOLS.items():
    os.makedirs(f"{G}/{out}/train")
    for p, names in chosen.items():
        for n in names:
            src, dst = os.path.realpath(f"{G}/{p}/train/{n}"), f"{G}/{out}/train/{n}"
            assert os.path.isfile(f"{src}/eval_config.json"), src
            assert not os.path.exists(dst), dst        # instance names must not collide across pools
            os.symlink(src, dst)
    json.dump({"built": "2026-10-27", "registration": "docs/RL_PLAN.md H-SEM (2026-10-27 06:32 UTC)",
               "pools": {p: len(v) for p, v in chosen.items()}, "instances": chosen}, open(f"{G}/{out}/MANIFEST.json", "w"), indent=1)
    print(out, {p: len(v) for p, v in chosen.items()}, "->", len(os.listdir(f"{G}/{out}/train")), "instances")
