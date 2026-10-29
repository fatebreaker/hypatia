"""fdr_rules_v05.py (10-28; the 10-27 mock review's concern 7; descriptive). Benjamini-Hochberg false-discovery control across every
pre-specified rule of Supplementary Table 1 that reports a p value: the p value of the rule's own verdict in its plan entry (v2, 10-29; see rule_p) ('p = 0.008',
'p <= 0.05', 'p = 5e-6', 'one-sided p = 0.14'; a bound 'p <= x' / 'p < x' is taken as p = x, conservatively). Rules decided by an
interval or a threshold with no p value are counted apart. Reports, at q = 0.05: how many MET rules with a p value keep a BH-adjusted
p <= 0.05, by rule type.
    python3 docs/rl4sci/v05/fdr_rules_v05.py   -> docs/rl4sci/v05/supplementary/Supplementary_Table_1b_fdr.csv
"""
import collections
import csv
import os
import re

D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, "supplementary", "Supplementary_Table_1_prespecified_rules.csv"))))
PAT = re.compile(r"\b[pP]\s*(?:=|<=|≤|<)\s*([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)")
# rules where MET means "not significantly different" (level / competitive / non-inferiority) pass on a LARGE p -- BH does not apply
EQUIV = re.compile(r"n\.s\.|COMPETITIVE|\blevel\b|LEVEL|not significant|p > 0\.05|advantage n|within 3|>= -3", re.I)
equiv = [r for r in rows if EQUIV.search(r["rule_as_registered"] + " " + r["rule"])]
rows = [r for r in rows if not EQUIV.search(r["rule_as_registered"] + " " + r["rule"])]
# 10-29 (v2): the p value of the rule ITSELF. v1 took the first p written in the reading, which in a multi-rule read is often
# another rule's (TDJ-RESCUES got TDJ-ABOVE-FRONTIER's p = 0.10; its own is 2e-9). Now: in the plan entry of the read, the last p
# the estimate in parentheses right after the rule's verdict ('**R MET** (+11.7, +48/-13, p = 4e-6)'), else the last p before
# the verdict and after the previous verdict ('... (+111/-39, p = 2e-9) -> TDJ-RESCUES MET'), else none (n.s., an interval or a
# threshold); if the rule's verdict is not found in the entry, the LAST p of the table's reading (it ends where the verdict starts).
ENTRY = re.compile(r"\* 2026-(\d\d-\d\d) (\d\d:\d\d) UTC -- (.*)")
PLAN = collections.defaultdict(list)
for l in open(os.path.join(D, "..", "..", "RL_PLAN.md")):
    e = ENTRY.match(l)
    if e:
        PLAN[f"{e.group(1)} {e.group(2)}"].append(e.group(3))
VERDICT = re.compile(r"\b(?:NOT MET|MET)\b")
# an estimate is written 'p = x' (or 'p < x' for tiny values); 'p <= 0.05' is the rule's bar restated, not a reading
EST = re.compile(r"\b[pP]\s*(?:=|<(?!=))\s*([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)")


def rule_p(rule, when, reading):
    seen = False
    for body in PLAN.get(when, []):
        for m in re.finditer(r"(?<![A-Za-z0-9\-])" + re.escape(rule) + r"(?![A-Za-z0-9\-])", body):
            v = re.match(r"[*:.\s]{0,4}(?:\([^)]{0,60}\)\s*)?(?:NOT MET|MET)\b[.*]*", body[m.end():m.end() + 70])
            if not v:
                continue          # a mention, not the verdict
            seen = True
            # the estimate right after the verdict ('**SHAM9-LIFTS-EHRSQL MET** (+11.7, +48/-13, p = 4e-6)')
            par = re.match(r"\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", body[m.end() + v.end():])
            if par and "bar" not in par.group(1) and EST.search(par.group(1)):
                return float(EST.search(par.group(1)).group(1))
            # else the last p before the verdict, after the previous verdict ('... (+111/-39, p = 2e-9) -> TDJ-RESCUES MET')
            win = body[max(0, m.start() - 400):m.start()]
            prev = list(VERDICT.finditer(win))
            ps = EST.findall(win[prev[-1].end():] if prev else win)
            if ps:
                return float(ps[-1])
    if seen:
        return None               # verdict found with no p of its own (n.s., an interval or a threshold)
    ps = PAT.findall(reading)
    return float(ps[-1]) if ps else None


withp, nop = [], []
for r in rows:
    pv = rule_p(r["rule"], r["evaluated_utc_2026"], r["reading"])
    (withp if pv is not None else nop).append((r, pv))
k = len(withp)
order = sorted(range(k), key=lambda i: withp[i][1])
adj = [0.0] * k; run = 1.0
for rank in range(k - 1, -1, -1):          # BH step-up: adj_(i) = min_{j >= i} p_(j) * k / j
    i = order[rank]; run = min(run, withp[i][1] * k / (rank + 1)); adj[i] = min(1.0, run)
out = []
for (r, p), a in zip(withp, adj):
    out.append({**{c: r[c] for c in ("rule", "experiment", "type", "outcome", "evaluated_utc_2026")}, "p": p, "bh_p": round(a, 5),
                "bh_significant": a <= 0.05})
with open(os.path.join(D, "supplementary", "Supplementary_Table_1b_fdr.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
met_p = [o for o in out if o["outcome"] == "MET"]
print(f"{len(equiv)} level / competitive rules set aside (they pass on a large p); the other "
      f"{len(rows)} rules: {k} report a p value, {len(nop)} do not (interval / threshold rules; {sum(r['outcome'] == 'MET' for r, _ in nop)} of those MET)")
print(f"MET rules with a p value: {len(met_p)}; BH-significant at q = 0.05: {sum(o['bh_significant'] for o in met_p)}")
by = collections.defaultdict(lambda: [0, 0])
for o in met_p:
    by[o["type"]][0] += 1; by[o["type"]][1] += o["bh_significant"]
for t, (a, b) in sorted(by.items()):
    print(f"  {t}: {b} of {a} MET rules keep BH p <= 0.05")
print("NOT MET rules that would be BH-significant (p reported but the rule failed on its estimate/threshold):",
      sum(o["bh_significant"] for o in out if o["outcome"] != "MET"))
