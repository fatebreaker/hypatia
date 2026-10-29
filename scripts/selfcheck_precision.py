#!/usr/bin/env python
"""Does the agent KNOW when it is right? Precision of its own self-check against the grader.

Why this decides the programme's last open route. A worked example installs an unseen family, but mechanically by
TRANSCRIPTION (scripts/demo_transcription.py): the demo's program is copied and run on new data. So a bootstrap
that reuses the agent's own CORRECT attempt as the demo will almost certainly work when the GRADER picks that
attempt -- a correct program is transcribed like any other. That is an oracle trick. A self-sufficient agent has
to pick its own correct attempt WITHOUT the grader, and the only signal it has is its own self-verification check
(SelfVerifyAgent: simulate data with a planted value, confirm the analysis recovers it). The harness records the
outcome verbatim as a "[selfcheck passed ...]" / "[selfcheck not passed ...]" message.

For every family x self-verify arm on disk, over rollouts that ANSWERED:
    base      P(passed)                          what a random pick would give
    coverage  P(selfcheck passed)
    precision P(passed | selfcheck passed)       what a self-selected demo would give
    lift      precision / base
and the question is whether precision clears the level at which a self-selected demo is usually RIGHT (> 50%)
on the families where the plain agent almost never is (base 2-6%).
"""
import json, glob, os, re, collections, sys
from scipy.stats import fisher_exact

SC = re.compile(r"\[selfcheck (passed|not passed)", re.I)

def selfcheck(msgs):
    """Outcome of the LAST selfcheck the harness recorded, or None if the rollout never reached one."""
    out = None
    for m in msgs:
        if m.get("role") == "user":
            for h in SC.findall(m.get("content") or ""):
                out = (h.lower() == "passed")
    return out

rows = []
for path in sorted(glob.glob("results/pods/general_agent_*/*.json")):
    arm, _, fam = os.path.basename(path)[:-5].partition("_")
    try:
        d = json.load(open(path))
    except Exception:
        continue
    if ":selfver" not in d.get("agent", "") and ":general" not in d.get("agent", ""):
        continue
    t = collections.Counter()
    for rolls in d.get("groups", {}).values():
        for r in rolls:
            if not r.get("answered"):
                continue
            s = selfcheck(r.get("messages") or [])
            if s is None:
                t["nocheck"] += 1
                continue
            t[(s, bool(r.get("passed")))] += 1
    tp, fp = t[(True, True)], t[(True, False)]
    fn, tn = t[(False, True)], t[(False, False)]
    n = tp + fp + fn + tn
    if n < 40:
        continue
    rows.append(dict(fam=fam, agent=d["agent"], draw=os.path.basename(os.path.dirname(path))[14:],
                     n=n, base=(tp + fn) / n, cov=(tp + fp) / n,
                     prec=tp / (tp + fp) if tp + fp else float("nan"),
                     npv_fail=tn / (tn + fn) if tn + fn else float("nan"),
                     p=fisher_exact([[tp, fp], [fn, tn]])[1], tp=tp, fp=fp, nocheck=t["nocheck"]))

json.dump(rows, open(os.environ.get("SC_CACHE","/tmp/sc_rows.json"),"w"))

def show(sel, title):
    print(f"\n{title}")
    print(f"{'family':18s}{'agent':34s}{'n':>5s}{'base':>7s}{'cover':>7s}{'PRECISION':>10s}{'lift':>6s}{'p':>10s}")
    for r in sel:
        lift = r["prec"] / r["base"] if r["base"] > 0 else float("inf")
        print(f"{r['fam'][:17]:18s}{r['agent'][4:37]:34s}{r['n']:5d}{100*r['base']:6.1f}%{100*r['cov']:6.1f}%"
              f"{100*r['prec']:9.1f}%{lift:6.1f}{r['p']:10.2g}")

low = [r for r in rows if r["base"] <= 0.10]
show(sorted(low, key=lambda r: r["base"]), "LOW-BASE cells (plain success <= 10%) -- where a bootstrap would have to work")
mid = [r for r in rows if r["base"] > 0.10]
print(f"\n({len(mid)} further cells with base > 10% omitted from the table; summary below)")

agg = collections.Counter()
for r in rows:
    agg["tp"] += r["tp"]; agg["fp"] += r["fp"]
print(f"\nALL {len(rows)} cells pooled: selfcheck-passed rollouts that are actually right = "
      f"{agg['tp']}/{agg['tp']+agg['fp']} = {100*agg['tp']/max(agg['tp']+agg['fp'],1):.1f}%")
lowagg = collections.Counter()
for r in low:
    lowagg["tp"] += r["tp"]; lowagg["fp"] += r["fp"]; lowagg["n"] += r["n"]; lowagg["pos"] += round(r["base"]*r["n"])
if low:
    print(f"LOW-BASE cells pooled: base {100*lowagg['pos']/lowagg['n']:.1f}%  ->  precision "
          f"{100*lowagg['tp']/max(lowagg['tp']+lowagg['fp'],1):.1f}%  ({lowagg['tp']}/{lowagg['tp']+lowagg['fp']})")
