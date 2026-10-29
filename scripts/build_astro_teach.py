"""build_astro_teach.py (2026-10-21, H-ASTRO9-DISTILL) -- the astronomy TRAINING problems the teacher (ASTRO27 seed 2) answers:
  astro_teach_syn  = 100 eclipsing binaries + 100 rotators + 100 transits from generated/astro_mix/train (synthetic)
  astro_teach_real = 150 eclipsing binaries + 150 rotators from generated/kepreal_mix/train (real Kepler TRAINING stars; the
                     test stars were excluded when that pool was built)
fixed-seed draws (seed 20261021). Each is a benchmark directory of symlinks (benchmarks/<name>/problems/<instance>, as
benchmarks/medcalc_train_1000_a), written in BOTH trees: /project/.../benchmarks and the /work tree's benchmark copy on /scratch4.
Refuses to overwrite.   python3 scripts/build_astro_teach.py
"""
import json
import os
import random
import re
import sys

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
TREES = ("/project/pi_hongyu_umass_edu/hieutran/deep/deeprs/benchmarks", "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/deeprs_benchmarks")
SEED = 20261021


def family(name):
    for f in ("eb", "rot", "transit"):
        if re.search(rf"(^|_){f}(_|$)", name):
            return f
    return None


def draw(pool, want, rng):
    names = sorted(os.listdir(f"{G}/{pool}/train"))
    by = {}
    for n in names:
        by.setdefault(family(n), []).append(n)
    out = []
    for fam, k in want.items():
        cand = by.get(fam, [])
        assert len(cand) >= k, (pool, fam, len(cand), k)
        out += sorted(rng.sample(cand, k))
    return out, {f: len(v) for f, v in by.items()}


rng = random.Random(SEED)
sets = {}
sets["astro_teach_syn"], fam_syn = draw("astro_mix", {"eb": 100, "rot": 100, "transit": 100}, rng)
real_pool = [n for n in sorted(os.listdir(f"{G}/kepreal_mix/train")) if not n.startswith("astro_")]   # real stars only
by = {}
for n in real_pool:
    by.setdefault(family(n), []).append(n)
sets["astro_teach_real"] = sorted(rng.sample(by["eb"], 150)) + sorted(rng.sample(by["rot"], 150))
src = {"astro_teach_syn": "astro_mix", "astro_teach_real": "kepreal_mix"}
for tree in TREES:
    for name, inst in sets.items():
        d = f"{tree}/{name}/problems"
        if os.path.exists(f"{tree}/{name}"):
            sys.exit(f"{tree}/{name} exists -- not overwriting")
        os.makedirs(d)
        for i in inst:
            os.symlink(f"{G}/{src[name]}/train/{i}", f"{d}/{i}")
        json.dump({"built": "2026-10-21", "registration": "docs/RL_PLAN.md H-ASTRO9-DISTILL (2026-10-21 14:07 UTC)", "seed": SEED,
                   "source_pool": src[name], "n": len(inst), "instances": inst}, open(f"{tree}/{name}/MANIFEST.json", "w"), indent=1)
print({k: len(v) for k, v in sets.items()}, "| synthetic families available:", fam_syn,
      "| real families available:", {f: len(v) for f, v in by.items()})
