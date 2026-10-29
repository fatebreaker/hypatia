#!/usr/bin/env python
"""Score the generator-verifier gap (experiments/judge_gap). Registered readings are in queue items 443/444.

Per family: parse rate, AUC of the judge's CONFIDENCE, TPR/FPR of its VERDICT and the precision that implies at
the NATURAL base rate, and the best precision over confidence thresholds that keep >= 10% recall (a bootstrap
needs ONE right pick, so high precision at low recall is enough). The bar is the best CONTENT-FREE cue on the
same family, sign-free -- a judge that only matches it is reading effort, not correctness.

  scripts/judge_gap_score.py results/pods/judge_gap/verdicts_qwen8_think.jsonl [...]
"""
import json, sys, collections
from sklearn.metrics import roc_auc_score, roc_curve

C = {json.loads(l)["id"]: json.loads(l) for l in open("results/pods/judge_gap/candidates.jsonl")}
# natural counts from build_set.py: (correct, wrong) in the full pool of plain attempts
NAT = {"hc_dx": (64, 1998), "osc_damped": (97, 2734), "arr_kinetics": (267, 3425), "enzyme_mm": (10, 2101),
       "pk_oral": (16, 1133), "epi_rt": (37, 2267), "epi_r0": (14, 2096), "causal_confounded": (1717, 8548)}
TRIV = {"length": lambda x: len(x["work"]), "turns": lambda x: x["work"].count("### ANALYST"),
        "errors": lambda x: -(x["work"].count("Traceback") + x["work"].count("Error")),
        "unforced": lambda x: 0 if "[forcing final answer]" in x["work"] else 1}

def auc(y, s):
    try: return roc_auc_score(y, s)
    except Exception: return float("nan")

for path in sys.argv[1:]:
    V = [json.loads(l) for l in open(path)]
    print(f"\n=== {path.split('/')[-1]}  ({len(V)} verdicts) ===")
    print(f"{'family':18s}{'parsed':>7s}{'AUC':>6s}{'trivial':>8s}{'TPR':>6s}{'FPR':>6s}{'base':>6s}"
          f"{'prec@VERDICT':>13s}{'best prec':>10s}   reading")
    beat = tot = 0
    for fam, (npos, nneg) in NAT.items():
        v = [x for x in V if x["family"] == fam]
        if not v: continue
        ok = [x for x in v if x["confidence"] is not None]
        y = [x["passed"] for x in ok]; s = [x["confidence"] for x in ok]
        a = auc(y, s)
        cands = [C[x["id"]] for x in ok]
        triv = max(max(t, 1 - t) for t in (auc(y, [f(c) for c in cands]) for f in TRIV.values()) if t == t)
        vv = [x for x in v if x["verdict"]]
        P = [x for x in vv if x["passed"]]; N = [x for x in vv if not x["passed"]]
        tpr = sum(x["verdict"] == "correct" for x in P) / max(len(P), 1)
        fpr = sum(x["verdict"] == "correct" for x in N) / max(len(N), 1)
        p = npos / (npos + nneg)
        prec = tpr * p / (tpr * p + fpr * (1 - p)) if (tpr * p + fpr * (1 - p)) else float("nan")
        best = float("nan")
        if len(set(y)) == 2:
            fp_, tp_, _ = roc_curve(y, s)
            best = max((t * p / (t * p + f * (1 - p)) for f, t in zip(fp_, tp_) if t >= 0.10 and (t * p + f * (1 - p)) > 0),
                       default=float("nan"))
        verdict = "VERIFIES" if a > triv + 0.05 and a >= 0.65 else ("~ effort cue" if abs(a - triv) <= 0.05 else "no")
        beat += verdict == "VERIFIES"; tot += 1
        print(f"{fam:18s}{100*len(ok)/len(v):6.0f}%{a:6.2f}{triv:8.2f}{tpr:6.2f}{fpr:6.2f}{100*p:5.1f}%"
              f"{100*prec:12.1f}%{100*best:9.1f}%   {verdict}")
    print(f"judge VERIFIES (AUC >= 0.65 and > trivial bar + 0.05) on {beat} of {tot} families")
