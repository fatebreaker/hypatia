#!/usr/bin/env python
"""How much does the CONTROL arm move between draws, per family and model?

This decides where a same-draw control is mandatory and where a pooled one is honest. It is the generalisation of
what item 301b found the hard way: pooled surv_tvc control 6.5%, that draw's own control 9.2%, and the difference
was the entire "falsify works" result.

A family whose controls are homogeneous across draws (chi-square p high, narrow range) can be read against a pooled
control. A family whose controls are heterogeneous cannot, and every pooled comparison on it is suspect.

Counts come from ga_table.score(), which refuses files with >=10% rollouts that never reached the server.
"""
import sys, os, glob, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score
from scipy.stats import chi2_contingency

ROOT = "results/pods"
ctl = collections.defaultdict(list)          # (model, family) -> [(dir, k, n)]
for p in glob.glob(os.path.join(ROOT, "general_agent_*", "control_*.json")):
    fam = os.path.basename(p)[len("control_"):-5]
    try:
        k, n = score(p)
    except ValueError:
        continue
    if not n:
        continue
    try:
        model = json.load(open(p)).get("agent", "").split(":")[1]
    except Exception:
        continue
    ctl[(model, fam)].append((os.path.basename(os.path.dirname(p)), k, n))

print(f"{'model':<11}{'family':<20}{'draws':>6}{'pooled':>9}{'min':>8}{'max':>8}{'spread':>8}   homogeneity")
print("-" * 92)
rows = []
for (model, fam), draws in ctl.items():
    if len(draws) < 3:
        continue
    K, N = sum(d[1] for d in draws), sum(d[2] for d in draws)
    rates = sorted(100 * k / n for _, k, n in draws)
    try:
        p = chi2_contingency([[k for _, k, _ in draws], [n - k for _, k, n in draws]])[1]
    except ValueError:
        p = float("nan")
    rows.append((p, model, fam, len(draws), 100 * K / N, rates[0], rates[-1], rates[-1] - rates[0]))
for p, model, fam, nd, pooled, lo, hi, spread in sorted(rows):
    flag = "  <-- POOLED CONTROL IS NOT SAFE HERE" if p < 0.05 else ""
    print(f"{model:<11}{fam:<20}{nd:>6}{pooled:>8.1f}%{lo:>7.1f}%{hi:>7.1f}%{spread:>7.1f}pt   p={p:.3g}{flag}")
