#!/usr/bin/env python
"""What analysis does each arm actually reach for? (failure attribution, per arm, within one draw)

`passed` tells you whether an arm works; this tells you WHY. On causal_confounded the recorded mechanism is that the
causal-order instruction moves the model off `post_treatment_control` (conditioning on a variable the treatment could
have changed) and onto the reference analysis -- not that it catches a bad analysis after the fact.

Files are gated by ga_table.score() first: a dead-server file has a huge `error:URLError` count that would otherwise
read as a dramatic attribution shift.

  attribution_shift.py <out_dir> <family>
"""
import sys, os, json, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score

d, fam = sys.argv[1], sys.argv[2]
root = os.path.join("results/pods", d)
arms = {}
for p in sorted(glob.glob(os.path.join(root, f"*_{fam}.json"))):
    arm = os.path.basename(p)[: -len(f"_{fam}.json")]
    try:
        k, n = score(p)
    except ValueError as e:
        print(f"!! EXCLUDED {arm}: {e}")
        continue
    c = collections.Counter()
    for rolls in json.load(open(p))["groups"].values():
        for r in rolls:
            c[r.get("attribution") or "none"] += 1
    arms[arm] = (k, n, c)

if not arms:
    sys.exit(f"no scoreable arms for {fam} in {d}")
keys = sorted({k for _, _, c in arms.values() for k in c}, key=lambda k: -sum(a[2][k] for a in arms.values()))
w = max(len(k) for k in keys) + 2
order = sorted(arms, key=lambda a: -arms[a][0] / max(arms[a][1], 1))
print(f"\n{d} [{fam}]\n")
print(" " * w + "".join(f"{a:>13}" for a in order))
print(" " * w + "".join(f"{100*arms[a][0]/arms[a][1]:>12.1f}%" for a in order) + "   <- passed")
print("-" * (w + 13 * len(order)))
for k in keys:
    print(f"{k:<{w}}" + "".join(f"{arms[a][2][k]:>13}" for a in order))
