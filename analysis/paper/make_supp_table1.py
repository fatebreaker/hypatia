"""make_supp_table1.py -- Supplementary Table 1: every pre-specified decision rule read since 2026-10-01, with its verdict, from
docs/rl4sci/registry_table.md (regenerate that first with docs/rl4sci/make_registry_table.py after any new READ in docs/RL_PLAN.md).
v2 (10-17, mock review): adds the registration time, the rule as registered, the reading (estimate) and a rule type, and prints
the totals by type that the Methods quote.
    python3 docs/rl4sci/make_registry_table.py && python3 docs/rl4sci/v05/make_supp_table1.py
    ->  docs/rl4sci/v05/supplementary/Supplementary_Table_1_prespecified_rules.csv
"""
import collections
import csv
import os
import re

D = os.path.dirname(os.path.abspath(__file__))
COLS = ["evaluated_utc_2026", "experiment", "rule", "outcome", "registered_utc_2026", "rule_as_registered", "reading"]


def rtype(rule):
    """Coarse type from the rule's name (the names follow one convention across the plan)."""
    r = rule.upper()
    if re.search(r"AUDIT|REPLIC|CERT|JUDGE", r):
        return "auditing and replication"
    if re.search(r"SHAM|PLACEBO|RFT|-NT\b|-NT-|CODE|NOKEY|LABEL-FREE|\bLF|LFI|ANSWER-SHAM|ASH", r):
        return "control"
    if re.search(r"SOL|GPT|FRONTIER|O4|API|ABOVE-PROMPT|LEVEL-|CLAUDE", r):   # 10-29: + CLAUDE (BEAT-CLAUDE-*)
        return "comparison with a frontier or reference model"
    if re.search(r"-REP\b|-REP-|REPLICATES|-3S\b|-3S-|SEED|^REP4?-", r):   # 10-29: + the fresh-seed rules REP-* / REP4-*
        return "replication across seeds"
    if re.search(r"TRANSFER|XFER|SCI|EPI|-PK\b|HELD|-HO\b|-HO-|BREADTH|ASTRO|SYN|REAL", r):
        return "transfer and generalisation"
    return "effect of training (lift over a baseline)"


rows = []
for l in open(f"{D}/../registry_table.md"):
    if not l.startswith("| ") or l.startswith("| Read") or l.startswith("|---"):
        continue
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) == 7:
        d = dict(zip(COLS, c))
        d["type"] = rtype(d["rule"])
        rows.append(d)
os.makedirs(f"{D}/supplementary", exist_ok=True)
with open(f"{D}/supplementary/Supplementary_Table_1_prespecified_rules.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS + ["type"]); w.writeheader(); w.writerows(rows)
met = sum(r["outcome"] == "MET" for r in rows)
print(f"Supplementary Table 1: {len(rows)} pre-specified tests, {met} met, {len(rows) - met} not met; "
      f"{sum(1 for r in rows if r['registered_utc_2026'])} with a registration entry found")
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[r["type"]][r["outcome"]] += 1
for t, c in sorted(by.items(), key=lambda x: -sum(x[1].values())):
    print(f"  {t}: {sum(c.values())} ({c['MET']} met, {c['NOT MET']} not met)")
