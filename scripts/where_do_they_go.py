#!/usr/bin/env python
"""When an arm significantly REDUCES the family's trap but the pass rate does not move -- where do those rollouts go?

51 of 241 same-draw comparisons have exactly that shape. There are three possibilities and they mean very different
things for "can this agent do science":
  (a) they become correct   -> the pass rate WOULD move; if it does not, this is not what happened
  (b) they move to ANOTHER documented trap -> the instruction redirects the error without fixing the analysis
  (c) they stop terminating -> the instruction costs budget and the rollout never arrives anywhere

Only (b) and (c) are consistent with a null pass rate, and they are different failures. This measures which.
Exhaustion-confounded rows (>=10pt termination gap) are excluded: for those, (c) is true by construction.
"""
import sys, os, glob, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score
from scipy.stats import fisher_exact

rows = []
for d in sorted(glob.glob("results/pods/general_agent_*")):
    byfam = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*.json")):
        arm, _, fam = os.path.basename(p)[:-5].partition("_")
        byfam[fam][arm] = p
    for fam, arms in byfam.items():
        if "control" not in arms:
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
        trap = max((x for x in cc if x not in ("reference", "none")), key=lambda x: cc[x], default=None)
        if trap is None or cn < 240:
            continue
        for arm, p in sorted(arms.items()):
            if arm == "control":
                continue
            try:
                k, n, c = load(p)
            except ValueError:
                continue
            if n < 240:
                continue
            tc, ta = cn - cc["none"], n - c["none"]
            if abs(100*c["none"]/n - 100*cc["none"]/cn) >= 10:
                continue
            p_pass = fisher_exact([[k, n-k], [ck, cn-ck]])[1]
            p_trap = fisher_exact([[c[trap], ta-c[trap]], [cc[trap], tc-cc[trap]]])[1]
            if not (p_trap < 0.05 and c[trap]/ta < cc[trap]/tc and p_pass >= 0.05):
                continue
            # where did the freed rollouts go? compare every OTHER attribution, per 240
            freed = 100*cc[trap]/tc - 100*c[trap]/ta
            others = {x: (100*c[x]/ta - 100*cc[x]/tc) for x in set(list(c)+list(cc))
                      if x not in (trap, "reference", "none")}
            gained = sorted(others.items(), key=lambda kv: -kv[1])[:2]
            nonterm = 100*c["none"]/n - 100*cc["none"]/cn
            rows.append((freed, os.path.basename(d), fam, arm, trap, gained, nonterm,
                         100*k/n, 100*ck/cn))

rows.sort(reverse=True)
print(f"{len(rows)} arms reduce their family's trap significantly with NO pass-rate movement\n")
print(f"{'family / arm':<34}{'trap freed':>11}{'to other traps':>16}{'to non-term':>12}   pass")
print("-"*92)
for freed, d, fam, arm, trap, gained, nonterm, pa, pc in rows:
    moved = " ".join(f"{a}+{v:.0f}" for a, v in gained if v > 1) or "-"
    print(f"{fam+' / '+arm:<34}{freed:>10.1f}pt{moved:>16}{nonterm:>11.1f}pt   {pa:.1f}% v {pc:.1f}%")
    print(f"   {d}  trap={trap}")
