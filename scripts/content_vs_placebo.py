#!/usr/bin/env python
"""Does the CONTENT do anything the CONTENTLESS SCAFFOLD does not? Content vs its own placebo, same draw.

WHY THIS REPLACES THE 28/0 TALLY. That tally compared each content arm against its CONTROL and counted the
significant rows. On a family where the SCAFFOLD ALONE moves the trap, an arm-vs-control comparison scores as
evidence for the content when it is evidence for the scaffold -- which is exactly what happened on causal_att:
    draw 1  control 42.9%   content 61.0%   placebo 63.4%   content vs placebo p = 0.64
    draw 2  control 40.8%   content 60.6%   placebo 53.9%   content vs placebo p = 0.16
Both arms far above control, neither distinguishable from the other. Counted as recipe evidence; it is scaffold.

Only draws holding content AND a contentless placebo AND a control are eligible -- everything else cannot answer
the question. Exhaustion-confounded rows (>=10pt gap) are dropped as everywhere else.
"""
import sys, os, glob, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score
from scipy.stats import fisher_exact

# `select` (M5) and `hetero` (M2) are mode-level CONTENT arms: same question, stated about a failure mode
# rather than a family. They belong here so the crossover draws are read by the refusing scorer like
# everything else. Their INTERACTION -- each instruction against the other, per mode -- is scripts/mode_crossover.py.
CONTENT = ["general", "selfver12", "selfver17", "selfver21", "selfver23", "select", "hetero"]
PLACEBO = ["selfver20", "selfver16"]
rows = []
for d in sorted(glob.glob("results/pods/general_agent_*")):
    byfam = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*.json")):
        arm, _, fam = os.path.basename(p)[:-5].partition("_")
        byfam[fam][arm] = p
    for fam, arms in byfam.items():
        if "control" not in arms:
            continue
        con = next((a for a in CONTENT if a in arms), None)
        pla = next((a for a in PLACEBO if a in arms), None)
        if not con or not pla:
            continue
        def load(a):
            k, n = score(arms[a])
            c = collections.Counter()
            for g in json.load(open(arms[a]))["groups"].values():
                for r in g:
                    c[r.get("attribution") or "none"] += 1
            return k, n, c
        try:
            ck, cn, cc = load("control"); gk, gn, gc = load(con); pk, pn, pc = load(pla)
        except ValueError:
            continue
        trap = max((x for x in cc if x not in ("reference", "none") and not x.startswith("error:")),
                   key=lambda x: cc[x], default=None)
        if not trap or cn < 240:
            continue
        gt, pt = gn - gc["none"], pn - pc["none"]
        if not gt or not pt or abs(100*gc["none"]/gn - 100*pc["none"]/pn) >= 10:
            continue
        p_gp = fisher_exact([[gc[trap], gt-gc[trap]], [pc[trap], pt-pc[trap]]])[1]
        p_gc = fisher_exact([[gc[trap], gt-gc[trap]], [cc[trap], (cn-cc['none'])-cc[trap]]])[1]
        model = json.load(open(arms["control"])).get("agent", "").split(":")[1]
        # REDIRECTION GUARD (2026-09-22). The single trap above is the CONTROL's commonest -- a fixed rule, not a
        # post-hoc pick -- but an arm can leave the total error rate untouched and merely move WHICH error it
        # makes. hc_dx_blind did exactly that: `select` vs its placebo reads p = 0.041 on `referred_only`
        # (70.1 v 60.9) while the family's whole M5 union is 85.0 v 80.0, p = 0.18, because the control spreads
        # across three M5 traps and `select` concentrates on one. Reading the single trap alone would have
        # published "the content makes it worse" off a reshuffle. Carry the ALL-TRAPS rate so it is visible.
        def allt(counter, term):
            return 100 * sum(v for k, v in counter.items()
                             if k not in ("reference", "none") and not k.startswith("error:")) / term
        rows.append((p_gp, fam, model, con, pla, 100*gc[trap]/gt, 100*pc[trap]/pt,
                     100*cc[trap]/(cn-cc["none"]), p_gc, os.path.basename(d),
                     allt(gc, gt), allt(pc, pt)))

rows.sort()
sep = [r for r in rows if r[0] < 0.05]
print(f"{len(rows)} draws hold content + contentless placebo + control on the same family\n")
print(f"{'family':<20}{'model':<11}{'content':>9}{'placebo':>9}{'control':>9}{'c-vs-p':>10}{'c-vs-ctl':>10}"
      f"{'ALL traps c/p':>16}")
print("-"*96)
for p_gp, fam, model, con, pla, g, pl, c, p_gc, d, ga, pa in rows:
    # A row whose single trap moves while the ALL-TRAPS rate does not has REDIRECTED errors, not removed them.
    redirect = p_gp < 0.05 and abs(ga - pa) < abs(g - pl) / 2
    mark = ("  <== REDIRECTION, not removal: all-traps barely moves" if redirect
            else "  <== content SEPARATES from scaffold" if p_gp < 0.05 else "")
    print(f"{fam:<20}{model:<11}{g:>8.1f}%{pl:>8.1f}%{c:>8.1f}%{p_gp:>10.3g}{p_gc:>10.3g}"
          f"{ga:>8.1f}%{pa:>7.1f}%{mark}")
print(f"\n{len(sep)} of {len(rows)} draws show the content doing something the contentless scaffold does not.")
