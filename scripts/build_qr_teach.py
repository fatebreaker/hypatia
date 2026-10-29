"""build_qr_teach.py (2026-10-24, B-QR4) -- the QRData-style TRAINING problems the teacher (MC27-GR seed 1 @100) answers:
  1,200 problems of the qrdisc pool = causal_mix 500 (100 per design: att, confounded, did, iv, rdd) + bdsgen 400 + causal_disc 300,
  fixed-seed draws (seed 20261024), shuffled together and cut into four 300-problem wrappers qr_teach_1 .. qr_teach_4 (so four GPUs
  share the load). Each is a benchmark directory of symlinks (benchmarks/<name>/problems/<instance>, as build_astro_teach.py),
  written in BOTH trees: /project/.../benchmarks and the /work tree's benchmark copy on /scratch4. Refuses to overwrite.
    python3 scripts/build_qr_teach.py
"""
import json
import os
import random
import sys

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
TREES = ("/project/pi_hongyu_umass_edu/hieutran/deep/deeprs/benchmarks", "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/deeprs_benchmarks")
SEED = 20261024
DESIGNS = ("att", "confounded", "did", "iv", "rdd")

rng = random.Random(SEED)
cm = sorted(os.listdir(f"{G}/causal_mix/train"))
draw = []   # (source pool, instance)
for d in DESIGNS:
    cand = [n for n in cm if n.startswith(f"causal_{d}_")]
    assert len(cand) >= 100, (d, len(cand))
    draw += [("causal_mix", n) for n in sorted(rng.sample(cand, 100))]
bds = sorted(os.listdir(f"{G}/bdsgen/train"))
draw += [("bdsgen", n) for n in sorted(rng.sample(bds, 400))]
disc = sorted(os.listdir(f"{G}/causal_disc/train"))
draw += [("causal_disc", n) for n in sorted(rng.sample(disc, 300))]
# every drawn instance is in the RL pool qrdisc too (the RL stage trains on all of qrdisc)
qd = set(os.listdir(f"{G}/qrdisc/train"))
missing = [n for _, n in draw if n not in qd]
assert not missing, missing[:5]
rng.shuffle(draw)
sets = {f"qr_teach_{i + 1}": draw[300 * i:300 * (i + 1)] for i in range(4)}
for tree in TREES:
    for name in sets:
        if os.path.exists(f"{tree}/{name}"):
            sys.exit(f"{tree}/{name} exists -- not overwriting")
for tree in TREES:
    for name, inst in sets.items():
        d = f"{tree}/{name}/problems"
        os.makedirs(d)
        for pool, i in inst:
            os.symlink(f"{G}/{pool}/train/{i}", f"{d}/{i}")
        json.dump({"built": "2026-10-24", "registration": "docs/RL_PLAN.md B-QR4 (2026-10-24 15:14 UTC)", "seed": SEED,
                   "source_pools": sorted({p for p, _ in inst}), "n": len(inst),
                   "per_pool": {p: sum(1 for q, _ in inst if q == p) for p in ("causal_mix", "bdsgen", "causal_disc")},
                   "instances": sorted(i for _, i in inst)}, open(f"{tree}/{name}/MANIFEST.json", "w"), indent=1)
print({k: {p: sum(1 for q, _ in v if q == p) for p in ("causal_mix", "bdsgen", "causal_disc")} for k, v in sets.items()},
      "| pools available: causal_mix", len(cm), "bdsgen", len(bds), "causal_disc", len(disc))
