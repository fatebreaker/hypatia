#!/usr/bin/env python
"""THE MODE CROSSOVER: is a failure MODE a teachable unit, or is every gain scaffolding and length?

Written BEFORE the arms landed (2026-09-21 ~22:15), so the criterion cannot be chosen to fit the numbers. The
design is docs/CROSS_DOMAIN_TASK_DESIGN.md section 11; queue items 411 (M2 side) and 412 (M5 side) write into ONE
OUT_DIR, which is one draw.

WHAT IT TESTS AND WHY A CROSSOVER. Two mode-level instructions -- `:hetero` (M2, "which units are comparable with
which") and `:select` (M5, "which rows and which columns is this question about") -- are run on BOTH mode-matched
family sets. Each instruction is the other's ACTIVE CONTROL: same skeleton, same register, and the shorter one
(`:select`, 1302 chars) is the one length-matched to the contentless placebo (1306), while `:hetero` is 1747. So:

  * a MAIN EFFECT -- one instruction better on both mode sets -- is length, or general improvement. Not a mode.
  * an INTERACTION -- each instruction better only on ITS OWN mode -- cannot be length, cannot be "more
    instruction helps", and cannot be "the agent got better". That is the mode account's prediction, and it is
    the only pattern this script reports as a pass.

The interaction is tested on the odds ratios, not by reading two p-values in sequence. Woolf's test for
homogeneity of the two ORs (M2 set, M5 set) IS the interaction test; a significant heterogeneity with opposite
signs is the crossover firing. Reading two separate significances in sequence is exactly how the 28/0 tally went
wrong, so it is not offered here.

Endpoint per family: the union of that family's traps mapped to ITS OWN mode, among TERMINATED rollouts, with the
programme's exhaustion filter (>=10pt non-termination gap = confounded, dropped) and pass rates printed beside
every trap number (the item-396 rule: two arms whose pass rates differ by more than ~15 points are not compared on
a trap rate alone).

    venv/bin/python scripts/mode_crossover.py [DRAW]        default DRAW = general_agent_crossover8
"""
import sys, os, json, collections, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import fisher_exact, chi2
from ga_table import score
from trap_modes import mode_of

DRAW = sys.argv[1] if len(sys.argv) > 1 else "general_agent_crossover8"
ROOT = "results/pods"

# Registered in section 11, before the data. Families carrying BOTH modes cannot separate the two instructions and
# are excluded by construction; causal_confounded is excluded from the M5 side because it is the home family of
# every positive result this programme has, and a transfer test may not be run on the surface it was written for.
SETS = {"M2_heterogeneity": ["surv_cox", "causal_att"],
        "M5_selection":     ["hc_dx", "surv_tvc"]}
ARM_OF = {"M2_heterogeneity": "hetero", "M5_selection": "select"}


def traps_of(fam, mode):
    from sage.registry import families
    f = families()[fam]
    rng = np.random.default_rng(0)
    return [t for t in f.traps(f.draw(rng, f.knobs(rng)).data) if mode_of(t) == mode]


def load(fam, arm):
    p = os.path.join(ROOT, DRAW, f"{arm}_{fam}.json")
    if not os.path.isfile(p):
        return None
    k, n = score(p)                        # raises on >=10% dead rollouts
    att = collections.Counter()
    for rolls in json.load(open(p))["groups"].values():
        for r in rolls:
            att[r.get("attribution") or "none"] += 1
    return {"k": k, "n": n, "att": att, "term": n - att["none"], "unfin": 100 * att["none"] / n}


def woolf(a, b, c, d, a2, b2, c2, d2):
    """Woolf's test that two odds ratios are equal. Returns (or1, or2, chi2, p)."""
    def lor(w, x, y, z):
        w, x, y, z = w + .5, x + .5, y + .5, z + .5          # Haldane-Anscombe, so a zero cell does not blow up
        return math.log((w * z) / (x * y)), 1/w + 1/x + 1/y + 1/z
    l1, v1 = lor(a, b, c, d)
    l2, v2 = lor(a2, b2, c2, d2)
    w1, w2 = 1 / v1, 1 / v2
    lbar = (w1 * l1 + w2 * l2) / (w1 + w2)
    stat = w1 * (l1 - lbar) ** 2 + w2 * (l2 - lbar) ** 2
    return math.exp(l1), math.exp(l2), stat, float(chi2.sf(stat, 1))


def main():
    print(f"MODE CROSSOVER -- draw {DRAW}\n")
    pooled = {}
    for mode, fams in SETS.items():
        own, other = ARM_OF[mode], ARM_OF["M5_selection" if mode == "M2_heterogeneity" else "M2_heterogeneity"]
        print(f"=== {mode}: own instruction :{own}, active control :{other} ===")
        print(f"{'family':12s} {'arm':10s} {'pass':>6s} {'mode traps':>11s} {'unfin':>6s}   vs :{other:8s} vs placebo")
        agg = collections.Counter()
        for fam in fams:
            try:
                traps = traps_of(fam, mode)
            except Exception as e:
                print(f"{fam:12s} registry error: {e}"); continue
            arms = {a: load(fam, a) for a in (own, other, "selfver20", "control")}
            missing = [a for a, v in arms.items() if v is None]
            if missing:
                print(f"{fam:12s} -- not yet complete, missing: {', '.join(missing)}")
                continue
            gaps = [v["unfin"] for v in arms.values()]
            if max(gaps) - min(gaps) >= 10:
                print(f"{fam:12s} -- EXHAUSTION CONFOUNDED (non-termination spread {max(gaps)-min(gaps):.1f}pt): not read")
                continue
            u = {a: sum(v["att"][t] for t in traps) for a, v in arms.items()}
            for a in (own, other, "selfver20", "control"):
                v = arms[a]
                line = f"{fam if a == own else '':12s} {a:10s} {100*v['k']/v['n']:5.1f}% {100*u[a]/v['term']:10.1f}% {v['unfin']:5.1f}%"
                if a != own:
                    p = fisher_exact([[u[own], arms[own]['term']-u[own]], [u[a], v['term']-u[a]]])[1]
                    gap = abs(100*arms[own]['k']/arms[own]['n'] - 100*v['k']/v['n'])
                    flag = "  [pass rates differ >15pt -- not read alone]" if gap > 15 else ""
                    line += f"   p={p:9.3g}{flag}"
                print(line)
            agg["own_t"] += u[own];  agg["own_n"] += arms[own]["term"] - u[own]
            agg["oth_t"] += u[other]; agg["oth_n"] += arms[other]["term"] - u[other]
            print()
        if agg["own_t"] or agg["oth_t"]:
            pooled[mode] = (agg["own_t"], agg["own_n"], agg["oth_t"], agg["oth_n"])

    if len(pooled) < 2:
        print("Both mode sides must be complete before the interaction can be tested. Nothing read.")
        return
    (a, b, c, d) = pooled["M2_heterogeneity"]
    (a2, b2, c2, d2) = pooled["M5_selection"]
    or1, or2, stat, p = woolf(a, b, c, d, a2, b2, c2, d2)
    print("=== THE INTERACTION (Woolf homogeneity of the two odds ratios) ===")
    print(f"  M2 families: odds of the M2 trap under :hetero vs :select = {or1:.3f}   (<1 means :hetero helps)")
    print(f"  M5 families: odds of the M5 trap under :select vs :hetero = {or2:.3f}   (<1 means :select helps)")
    print(f"  heterogeneity chi2(1) = {stat:.2f}, p = {p:.4g}")
    fires = or1 < 1 and or2 < 1
    print()
    if fires and p < 0.05:
        print("  CROSSOVER FIRES: each instruction helps on its OWN mode and the two effects differ. Length and")
        print("  general improvement are both excluded by the design. A failure mode is a teachable unit.")
    elif or1 < 1 and or2 > 1 or or1 > 1 and or2 < 1:
        print("  MAIN EFFECT, NOT AN INTERACTION: one instruction wins on both sides. That is length or general")
        print("  improvement -- section 11 registers this as a FAIL for the mode account, whatever its p-value.")
    else:
        print("  NO CROSSOVER: neither instruction is specific to its own mode. Mode-level content does not")
        print("  transfer either, and the honest conclusion is scaffolding plus one causal_confounded effect.")


if __name__ == "__main__":
    main()
