#!/usr/bin/env python
"""Every same-draw comparison in the tree, read at the TRAP endpoint as well as the pass rate.

Why this exists: on 2026-09-21 three separate arms were being read wrongly from the pass rate alone --
  causal_iv @14B   a real, content-caused HARM that the pass rate calls a null (p=0.22 vs p=5e-4)
  surv_tvc @8B     a real, placebo-controlled REDIRECTION that nets to nothing (the device is not inert)
  eco_mb @8B       apparent trap AVOIDANCE that was really non-termination
The pass rate is a NET of competing shifts. This prints both endpoints for every draw so the next one is caught.

Conventions, each of which cost something to learn:
  * `attribution == "reference"` is identical to `passed`, so the reference row is never the interesting one.
  * Non-termination ("none") is EXCLUDED from trap denominators: a rollout that never finished did not avoid
    the trap, and counting it as avoidance is what made eco_mb look like transfer.
  * Every file goes through ga_table.score(), which refuses >=10% dead rollouts.
  * The placebo is printed beside every arm, because the only question that matters is arm-vs-placebo.

  trap_sweep.py [--min-n 240] [--only SUBSTRING]
"""
import sys, os, glob, json, collections, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score
from scipy.stats import fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("--min-n", type=int, default=240)
ap.add_argument("--only", default="")
a = ap.parse_args()

PLACEBOS = {"selfver20", "selfver16", "selfver25", "selfver24"}
rows = []
for d in sorted(glob.glob("results/pods/general_agent_*")):
    if a.only and a.only not in d:
        continue
    byfam = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*.json")):
        arm, _, fam = os.path.basename(p)[:-5].partition("_")
        byfam[fam][arm] = p
    for fam, arms in byfam.items():
        if "control" not in arms or len(arms) < 2:
            continue
        data = {}
        for arm, p in arms.items():
            try:
                k, n = score(p)
            except ValueError:
                continue
            if n < a.min_n:
                continue
            c = collections.Counter()
            for rolls in json.load(open(p))["groups"].values():
                for r in rolls:
                    c[r.get("attribution") or "none"] += 1
            data[arm] = (k, n, c)
        if "control" not in data or len(data) < 2:
            continue
        ck, cn, cc = data["control"]
        term_c = cn - cc["none"]
        # the family's trap = the most common non-reference, non-none attribution in the CONTROL arm
        # EXCLUDE error keys: if crashes or a dead server dominate the control, `error:RuntimeError` would be
        # picked as "the family's trap" and every arm compared against it (review, 2026-09-21).
        trap = max((x for x in cc if x not in ("reference", "none") and not x.startswith("error:")),
                   key=lambda x: cc[x], default=None)
        if trap is None:
            continue
        model = ""
        try:
            model = json.load(open(arms["control"])).get("agent", "").split(":")[1]
        except Exception:
            pass
        for arm, (k, n, c) in sorted(data.items()):
            if arm == "control":
                continue
            term = n - c["none"]
            if not term or not term_c:
                continue
            p_pass = fisher_exact([[k, n - k], [ck, cn - ck]])[1]
            p_trap = fisher_exact([[c[trap], term - c[trap]], [cc[trap], term_c - cc[trap]]])[1]
            rows.append((p_trap, os.path.basename(d), model, fam, arm, trap,
                         100 * k / n, 100 * ck / cn, p_pass,
                         100 * c[trap] / term, 100 * cc[trap] / term_c,
                         100 * c["none"] / n, 100 * cc["none"] / cn))

print(f"{len(rows)} same-draw arm-vs-control comparisons, sorted by trap-shift significance\n")
hdr = f"{'family / arm':<44}{'pass':>17}{'p':>9}   {'trap-taking':>17}{'p':>9}   nonterm"
print(hdr); print("-" * len(hdr))
for pt, d, model, fam, arm, trap, pa, pc, pp, ta, tc, na, nc in rows:
    # A conditional trap rate is only clean when the arms TERMINATE at similar rates. Exhaustion removes rollouts
    # non-randomly -- the ones still heading for a trap -- so a large gap can manufacture a shift in either
    # direction. Found on 2026-09-21: it explains eco_mb's apparent transfer entirely, and it knocks causal_iv's
    # placebo row (11.2% vs 0.0% non-termination) out of the evidence for the "scaffolding causes over-adjustment"
    # account, leaving that side of the account resting on causal_confounded alone.
    gap = abs(na - nc)
    flag = ""
    if pt < 0.05 and gap >= 10:
        flag = f"  <== CONFOUNDED: non-termination gap {gap:.0f}pt, do not read this shift"
    elif pt < 0.05 and pp >= 0.05:
        flag = "  <== TRAP MOVES, PASS RATE DOES NOT"
    elif pt < 0.05 and ta > tc:
        flag = "  <== TRAP CAUSED"
    if pt < 0.05 and 5 <= gap < 10 and "CONFOUNDED" not in flag:
        flag += f"  (non-termination gap {gap:.0f}pt -- treat as suspect)"
    star = "*" if arm in PLACEBOS else " "
    print(f"{fam+' / '+arm+star:<44}{pa:6.1f}% v{pc:6.1f}%{pp:>9.3g}   "
          f"{ta:6.1f}% v{tc:6.1f}%{pt:>9.3g}   {na:4.1f}/{nc:<4.1f}{flag}")
    print(f"{'   ' + d + '  [' + model + ']  trap=' + trap:<44}")
