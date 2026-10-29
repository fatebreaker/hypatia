#!/usr/bin/env python
"""How many INDEPENDENT DRAWS support each (family, model, arm-vs-control) comparison?

Built 2026-09-21 after recording "the registered verdict is determined by this draw" about a SINGLE draw, and
having the replication contradict it (v26 on causal_iv @14B: draw 1 p=0.41, draw 2 p=0.0046).

The programme's registration discipline covered WHAT READING follows from a result. It did not cover HOW MANY
DRAWS before reading it at all. This lists every comparison by draw count so a one-draw claim cannot be written up
as settled without noticing.
"""
import sys, os, glob, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score

by = collections.defaultdict(set)
for p in glob.glob("results/pods/general_agent_*/*.json"):
    base = os.path.basename(p)[:-5]
    arm, _, fam = base.partition("_")
    d = os.path.basename(os.path.dirname(p))
    if not os.path.isfile(os.path.join("results/pods", d, f"control_{fam}.json")):
        continue                       # no same-draw control: not a comparison at all
    try:
        k, n = score(p)
        model = json.load(open(p)).get("agent", "").split(":")[1]
    except Exception:
        continue
    if n < 240 or arm == "control":
        continue
    by[(fam, model, arm)].add(d)

rows = sorted(by.items(), key=lambda kv: len(kv[1]))
one = [r for r in rows if len(r[1]) == 1]
print(f"{len(rows)} (family, model, arm) comparisons with a same-draw control")
print(f"  {len(one)} rest on ONE draw\n")
print("ONE-DRAW comparisons involving the headline arms (general / selfver20 / selfver12 / selfver26):")
for (fam, model, arm), draws in one:
    if arm in ("general", "selfver20", "selfver12", "selfver26", "selfver", "ident", "select"):
        print(f"   {fam:<20} {model:<11} {arm:<11} {list(draws)[0]}")
print("\nBest-replicated:")
for (fam, model, arm), draws in rows[-8:]:
    print(f"   {len(draws)} draws  {fam:<20} {model:<11} {arm}")
