#!/usr/bin/env python
"""DABench, read the way the family results are now read: CONTENT vs its OWN PLACEBO, paired by problem.

WHY THIS EXISTS. The goal has two halves -- do well in general science, and generalise to unseen domains. Almost
everything in this programme measures the second half, on families we generate. DABench is the first half: 258
real data-analysis problems nobody here wrote. Every arm has run all 258, so the comparison is PAIRED and no GPU
time is needed to ask the question.

It is the same question the 28/0 tally got wrong on the families. "The self-verify agent beats the plain agent on
DABench" is a statement about the LOOP, not about what the loop is told to do. The only comparison that separates
them is a content arm against the contentless scaffold that matches it clause for clause:

    v12, v21, v23   REPLICATE-stage CONTENT   -> their placebo is v20 (same scaffolding, causally empty)
    v7, v9          READ/FALSIFY-stage arms   -> their placebo is v16 (READ-stage placebo)

Paired data means McNemar, not Fisher: with 258 problems answered by both arms, the concordant pairs carry no
information about the difference and throwing them into a 2x2 of marginals discards most of the power. The exact
binomial on the discordant pairs is the test.

    venv/bin/python scripts/dabench_content.py
"""
import os, json, glob, collections, warnings
warnings.filterwarnings("ignore")
from scipy.stats import binomtest, wilcoxon

ROOT = "results/pods"
# arm directory -> (label, its matched placebo label or None)
PAIRS = [
    ("v23 (content; `:general` below 20B)", "dabench_v23/dabench_selfver23",     "v20 placebo"),
    ("v21 (content)",                       "dabench_v21/dabench_selfver21",     "v20 placebo"),
    ("v21 (content, seed 2)",               "dabench_v21_s2/dabench_selfver21",  "v20 placebo s2"),
    ("v12 (content)",                       "dabench_v12/dabench_selfver12",     "v20 placebo"),
    ("v12 (content, seed 2)",               "dabench_v12_s2/dabench_selfver12",  "v20 placebo s2"),
    ("v17 (content)",                       "dabench_v17/dabench_selfver17",     "v20 placebo"),
    ("v18 (content)",                       "dabench_v18/dabench_selfver18",     "v20 placebo"),
    ("v7  (READ-stage content)",            "dabench_v7_s2/dabench_selfver7",    "v16 placebo"),
    ("v9  (FALSIFY content)",               "dabench_v9_s2/dabench_selfver9",    "v16 placebo"),
]
PLACEBO = {"v20 placebo":    "dabench_v20/dabench_selfver20",
           "v20 placebo s2": "dabench_v20_s2/dabench_selfver20",
           "v16 placebo":    "dabench_v7abl/dabench_selfver16"}
CONTROLS = {"control":    "dabench_agent/dabench_control",
            "control s2": "dabench_agent_s2/dabench_control"}


def load(rel):
    out = {}
    for p in glob.glob(os.path.join(ROOT, rel, "*.json")):
        d = json.load(open(p))
        g = d.get("grade") or {}
        out[d.get("id") or os.path.basename(p)[:-5]] = (bool(g.get("passed")), float(g.get("score") or 0.0))
    return out


def mcnemar(a, b):
    """Exact McNemar on the problems both arms answered. Returns (n_pairs, a_only, b_only, p)."""
    ids = sorted(set(a) & set(b))
    ao = sum(1 for i in ids if a[i][0] and not b[i][0])
    bo = sum(1 for i in ids if b[i][0] and not a[i][0])
    if ao + bo == 0:
        return len(ids), 0, 0, 1.0
    return len(ids), ao, bo, binomtest(ao, ao + bo, 0.5).pvalue


def main():
    arms = {}
    for label, rel, _ in PAIRS:
        arms[label] = load(rel)
    for label, rel in {**PLACEBO, **CONTROLS}.items():
        arms[label] = load(rel)

    print("DABENCH -- 258 real data-analysis problems, every arm on all of them\n")
    print(f"{'arm':38s} {'n':>4s} {'pass':>7s} {'mean score':>11s}")
    for label in [l for l, _, _ in PAIRS] + list(PLACEBO) + list(CONTROLS):
        d = arms[label]
        if not d:
            print(f"{label:38s}  -- not on disk"); continue
        n = len(d)
        print(f"{label:38s} {n:4d} {100*sum(v[0] for v in d.values())/n:6.1f}% {sum(v[1] for v in d.values())/n:10.3f}")

    print("\n=== CONTENT vs ITS OWN PLACEBO (paired, exact McNemar on discordant problems) ===")
    print(f"{'contrast':46s} {'pairs':>6s} {'C only':>7s} {'P only':>7s} {'p':>9s}   reading")
    for label, _, plab in PAIRS:
        c, p_ = arms.get(label), arms.get(plab)
        if not c or not p_:
            continue
        n, co, po, p = mcnemar(c, p_)
        verdict = ("content SEPARATES" if p < 0.05 and co > po else
                   "content WORSE" if p < 0.05 else "indistinguishable from scaffolding")
        print(f"{label + ' vs ' + plab:46s} {n:6d} {co:7d} {po:7d} {p:9.3g}   {verdict}")

    print("\n=== the same arms against the PLAIN agent (what the old tally reported) ===")
    print(f"{'contrast':46s} {'pairs':>6s} {'A only':>7s} {'ctl only':>8s} {'p':>9s}")
    for label in [l for l, _, _ in PAIRS] + list(PLACEBO):
        a, c = arms.get(label), arms.get("control")
        if not a or not c:
            continue
        n, ao, co, p = mcnemar(a, c)
        print(f"{label + ' vs control':46s} {n:6d} {ao:7d} {co:8d} {p:9.3g}")

    print("\nRead the two blocks together. An arm that beats CONTROL but not its own PLACEBO has shown that the")
    print("scaffolding works, not that what the scaffolding says is true -- the exact error the 28/0 family tally")
    print("made. The placebo rows in the second block are the check on that: they beat control too.")


if __name__ == "__main__":
    main()
