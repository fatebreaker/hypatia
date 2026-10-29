#!/usr/bin/env python
"""For every UNSEEN family: does the control arm have headroom, and does the spec do anything?

A null on a family where the control scores 0% is nearly uninformative -- nothing could have been detected. A null
where the control scores 9% is real evidence. The "eight zero-floor families" framing conflates the two and
understates the result wherever headroom exists. Worst case per family is reported: the draw with the MOST control
headroom, since that is the one with the most power to show an effect.
"""
import sys, os, glob, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score
from scipy.stats import fisher_exact

UNSEEN = {"epi_rt","epi_r0","eco_mb","eco_size","enzyme_mm","osc_damped","hc_quality","astro_transit","pk_oral"}
best = {}
for p in glob.glob("results/pods/general_agent_*/control_*.json"):
    fam = os.path.basename(p)[len("control_"):-5]
    if fam not in UNSEEN:
        continue
    d = os.path.dirname(p)
    arm = os.path.join(d, f"general_{fam}.json")
    if not os.path.isfile(arm):
        continue
    try:
        ck, cn = score(p); ak, an = score(arm)
        model = json.load(open(p)).get("agent", "").split(":")[1]
    except Exception:
        continue
    if not cn or not an:
        continue
    key = (fam, model)
    if key not in best or ck / cn > best[key][0] / best[key][1]:
        best[key] = (ck, cn, ak, an, os.path.basename(d))

print(f"{'family':<15}{'model':<11}{'control':>10}{'spec':>10}{'p':>10}   power verdict")
print("-" * 82)
for (fam, model), (ck, cn, ak, an, draw) in sorted(best.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
    p = fisher_exact([[ak, an - ak], [ck, cn - ck]])[1]
    cr = 100 * ck / cn
    verdict = ("INFORMATIVE null -- control has real headroom" if cr >= 3
               else "weak null -- control at/near the floor, little could be seen")
    print(f"{fam:<15}{model:<11}{cr:>9.1f}%{100*ak/an:>9.1f}%{p:>10.3g}   {verdict}")
