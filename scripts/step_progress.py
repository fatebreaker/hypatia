#!/usr/bin/env python
"""When an instruction moves rollouts from one documented trap to another, does the model get FURTHER?

Every trap carries the analysis step it skips (`sage.core.Trap.step`): a higher step means the rollout completed
more of the analysis before going wrong. So the trap-to-trap redirection found by where_do_they_go.py has a
direction, and it distinguishes two very different readings of the same null pass rate:

  forward  -- the instruction advances the analysis; the model fails later than it used to. Partial progress.
  backward -- the instruction costs ground; the model fails earlier. A harm the pass rate cannot see.
  lateral  -- same step, a different mistake of equal depth. Pure substitution.

Reported as the change in MEAN SKIPPED STEP over rollouts that took any documented trap, arm minus control, within
one draw. Traps with step None (the family's "near miss") are excluded -- they have no depth. Exhaustion-confounded
comparisons (>=10pt termination gap) are excluded as everywhere else.
"""
import sys, os, glob, json, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
import numpy as np
from ga_table import score
from sage.registry import families

STEPS = {}
def steps_for(fam):
    if fam in STEPS:
        return STEPS[fam]
    try:
        f = families()[fam]
        rng = np.random.default_rng(0)
        t = f.traps(f.draw(rng, f.knobs(rng)).data)
        STEPS[fam] = {k: v.step for k, v in t.items() if v.step is not None}
    except Exception:
        STEPS[fam] = {}
    return STEPS[fam]

def mean_step(counter, smap):
    tot = sum(counter[k] for k in smap)
    if not tot:
        return None, 0
    return sum(counter[k] * smap[k] for k in smap) / tot, tot

rows = []
for d in sorted(glob.glob("results/pods/general_agent_*")):
    byfam = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*.json")):
        arm, _, fam = os.path.basename(p)[:-5].partition("_")
        byfam[fam][arm] = p
    for fam, arms in byfam.items():
        if "control" not in arms:
            continue
        smap = steps_for(fam)
        if len(smap) < 2:
            continue
        def load(p):
            k, n = score(p)
            c = collections.Counter()
            for rolls in json.load(open(p))["groups"].values():
                for r in rolls:
                    c[r.get("attribution") or "none"] += 1
            return k, n, c
        try:
            ck, cn, cc = load(arms["control"])
        except ValueError:
            continue
        cms, ctot = mean_step(cc, smap)
        if cms is None or cn < 240:
            continue
        for arm, p in sorted(arms.items()):
            if arm == "control":
                continue
            try:
                k, n, c = load(p)
            except ValueError:
                continue
            if n < 240 or abs(100*c["none"]/n - 100*cc["none"]/cn) >= 10:
                continue
            ams, atot = mean_step(c, smap)
            if ams is None or atot < 30:
                continue
            # SELECTION ARTIFACT (found 2026-09-21): the mean skipped step is computed over rollouts that took a
            # trap, so an arm that converts many rollouts to PASSES removes its best failures from that pool and the
            # remainder looks shallower. arr_kinetics/:general reads -0.50 steps while gaining +28 points of pass
            # rate -- the depth "loss" is entirely the pass gain. The comparison is only interpretable where the
            # pass rates are close, so require that and report the rest separately.
            if abs(100*k/n - 100*ck/cn) > 3.0:
                continue
            rows.append((ams - cms, fam, arm, os.path.basename(d), ams, cms,
                         100*k/n, 100*ck/cn))

rows.sort(reverse=True)
fwd = sum(1 for r in rows if r[0] > 0.1); bwd = sum(1 for r in rows if r[0] < -0.1)
print(f"{len(rows)} same-draw comparisons with a usable trap-depth map\n")
print(f"  moved the model FORWARD (fails later):  {fwd}")
print(f"  moved it BACKWARD (fails earlier):      {bwd}")
print(f"  lateral (|change| <= 0.1 steps):        {len(rows)-fwd-bwd}\n")
print(f"{'family / arm':<34}{'mean skipped step':>20}{'change':>9}   pass")
print("-"*84)
for ch, fam, arm, d, ams, cms, pa, pc in rows[:14] + [("...",)*8] + rows[-8:]:
    if fam == "...":
        print("   ..."); continue
    print(f"{fam+' / '+arm:<34}{ams:>9.2f} v {cms:<7.2f}{ch:>+9.2f}   {pa:.1f}% v {pc:.1f}%")
