#!/usr/bin/env python
"""WHICH FAMILIES CAN CARRY AN EXPERIMENT AT ALL? Run this BEFORE choosing evaluation families.

WHY THIS EXISTS, written the night it was needed four times. An experiment on a family whose control cannot
solve anything measures which error the agent makes while failing, not whether it does better; an experiment on
a family whose control never falls into the trap has nothing to reduce. Three designs went out in 2026-09-21/22
without this check and all three came back uninterpretable:

  * E1_M5 held-out transfer chose hc_dx (control pass 0.4%, M5 trap 87%) and hc_quality (0.0%). Both floors.
  * the scaffold-sign prospective test chose epi_r0, whose trap sits at 0.0-0.6% in every arm at both scales --
    no room for the predicted reduction in either direction. Three of its four legs are unreadable.
  * astro_transit and eco_mb were counted in a transfer null while scoring 0.0% in EVERY arm.

The one design that ran this check first (the M2 crossover) rejected five of ten candidate families as floors
before spending a GPU-hour, and picked the two with real headroom. The check costs no GPU time at all: it reads
control arms already on disk.

USABLE, as the M2 crossover used it:
    pass rate between ~5% and ~70%          -- off the floor, off the ceiling
    total trap rate >= 15% among terminated -- something to reduce
    non-termination < 10%                   -- the exhaustion filter would not eat it

    venv/bin/python scripts/headroom.py [--model Qwen3-8B]
"""
import sys, os, json, glob, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ga_table import score, resolved_spec

WANT = None
if "--model" in sys.argv:
    WANT = sys.argv[sys.argv.index("--model") + 1]

best = {}
for path in sorted(glob.glob("results/pods/general_agent_*/control_*.json")):
    if os.path.basename(path).startswith("."):
        continue
    fam = os.path.basename(path)[len("control_"):-len(".json")]
    try:
        d = json.load(open(path))
        k, n = score(path, d)
    except Exception:
        continue
    model, _ = resolved_spec(d.get("agent", ""))
    if WANT and model != WANT:
        continue
    att = collections.Counter()
    for rolls in d["groups"].values():
        for r in rolls:
            att[r.get("attribution") or "none"] += 1
    term = n - att["none"]
    if not term:
        continue
    traps = sum(v for kk, v in att.items()
                if kk not in ("reference", "none") and not kk.startswith("error:"))
    row = (n, 100*k/n, 100*traps/term, 100*att["none"]/n, os.path.basename(os.path.dirname(path)))
    key = (fam, model)
    if key not in best or row[0] > best[key][0]:
        best[key] = row

print("HEADROOM — the best control arm on record for each family and scale\n")
print(f"{'family':20s}{'model':11s}{'n':>5s}{'pass':>7s}{'traps':>8s}{'unfin':>7s}  {'verdict':26s} best draw")
def verdict(p, t, u):
    if u >= 10:      return "EXHAUSTION-PRONE"
    if p < 5:        return "FLOOR (nothing passes)"
    if p > 70:       return "CEILING"
    if t < 15:       return "NO TRAP HEADROOM"
    return "USABLE"
rows = sorted(best.items(), key=lambda kv: (kv[0][1], -(verdict(*kv[1][1:4]) == "USABLE"), kv[0][0]))
for (fam, model), (n, p, t, u, d) in rows:
    v = verdict(p, t, u)
    print(f"{fam:20s}{model:11s}{n:5d}{p:6.1f}%{t:7.1f}%{u:6.1f}%  {v:26s} {d[14:]}")
usable = [k for k, v in best.items() if verdict(*v[1:4]) == "USABLE"]
print(f"\n{len(usable)} of {len(best)} (family, scale) cells are USABLE for an experiment that needs both")
print("a measurable pass rate and a trap with room to fall:")
for fam, model in sorted(usable):
    print(f"    {fam} @ {model}")
