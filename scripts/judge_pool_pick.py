#!/usr/bin/env python
"""Is the judge's TOP pick in a real pool correct? The direct measurement behind a human-free bootstrap.

For every natural pool in results/pods/judge_gap/pools.jsonl: take the attempt with the highest judge confidence
(ties -> shortest trace; the grader is never consulted to choose), then read its label. Also reports, per pool,
the rank of the best-ranked correct attempt -- a bootstrap that may try the top few picks as demonstrations needs
that number, not just the top-1 outcome.

  scripts/judge_pool_pick.py [VERDICTS]   (default results/pods/judge_gap/pool_verdicts_qwen8_think.jsonl)
"""
import json, sys, collections
V = sys.argv[1] if len(sys.argv) > 1 else "results/pods/judge_gap/pool_verdicts_qwen8_think.jsonl"
P = {json.loads(l)["id"]: json.loads(l) for l in open("results/pods/judge_gap/pools.jsonl")}
J = {}
for l in open(V):
    v = json.loads(l)
    if v["id"] in P:
        J[v["id"]] = v
pools = collections.defaultdict(list)
for i, c in P.items():
    pools[c["pool"]].append(c)
fam_stats = collections.defaultdict(lambda: [0, 0, 0.0])
print(f"{'pool':54s}{'judged':>8s}{'right':>6s}{'base':>7s}{'top-1':>7s}{'conf':>6s}{'rank of 1st right':>19s}")
for pid, items in sorted(pools.items()):
    judged = [c for c in items if c["id"] in J and J[c["id"]]["confidence"] is not None]
    if len(judged) < 0.95 * len(items):
        print(f"{pid:54s}{len(judged):5d}/{len(items):<4d}  (incomplete, not scored)"); continue
    ranked = sorted(judged, key=lambda c: (-J[c["id"]]["confidence"], len(c["work"])))
    top = ranked[0]
    first_right = next((k + 1 for k, c in enumerate(ranked) if c["passed"]), None)
    nr = sum(c["passed"] for c in items)
    fam = items[0]["family"]
    fam_stats[fam][0] += 1; fam_stats[fam][1] += top["passed"]; fam_stats[fam][2] += nr / len(items)
    print(f"{pid:54s}{len(judged):5d}/{len(items):<4d}{nr:4d}{100*nr/len(items):6.1f}%"
          f"{'RIGHT' if top['passed'] else 'wrong':>7s}{J[top['id']]['confidence']:6d}{str(first_right):>19s}")
print(f"\n{'family':20s}{'pools':>6s}{'top-1 right':>13s}{'mean base':>11s}")
for fam, (n, r, b) in fam_stats.items():
    print(f"{fam:20s}{n:6d}{r:9d}/{n:<3d}{100*b/n:10.1f}%")
