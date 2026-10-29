#!/usr/bin/env python
"""Is CONSENSUS a verifier on unseen families? Self-consistency tested against the grader, no GPU.

Within each instance, the plain agent's k rollouts give k answers. Self-consistency picks the answer most of them
agree on. On a TRAP family that should fail: the benchmark is built so that the natural, naive analysis lands on
the SAME wrong number every time, while right answers need a less obvious analysis. Measured here per family:
for each rollout, how many of its instance-mates gave a matching answer (every numeric field within the
family's tolerance, relative 5%), and whether agreement predicts correctness (AUC; 0.5 = uninformative,
< 0.5 = consensus points AWAY from the truth).
"""
import json, glob, os, collections, sys
from sklearn.metrics import roc_auc_score

FAMS = ["hc_dx", "osc_damped", "arr_kinetics", "enzyme_mm", "pk_oral", "epi_rt", "epi_r0", "causal_confounded",
        "causal_did", "causal_iv", "causal_rdd", "surv_cox"]
def nums(ans):
    if not isinstance(ans, dict): return None
    out = {}
    for k, v in ans.items():
        try: out[k] = float(v)
        except Exception: pass
    return out or None
def agree(a, b, tol=0.05):
    if not a or not b or set(a) != set(b): return False
    return all(abs(a[k] - b[k]) <= tol * max(abs(a[k]), abs(b[k]), 1e-9) for k in a)

print(f"{'family':18s}{'rollouts':>9s}{'base':>7s}{'AUC(agreement)':>16s}{'P(right|agrees w/ >=2)':>24s}{'P(right|alone)':>16s}")
for fam in FAMS:
    y, s = [], []
    for p in glob.glob(f"results/pods/general_agent*/control_{fam}.json"):
        try: d = json.load(open(p))
        except Exception: continue
        if d.get("agent") != "llm:Qwen3-8B:nothink": continue
        for g in d["groups"].values():
            A = [nums(r.get("answer")) if r.get("answered") else None for r in g]
            for i, r in enumerate(g):
                if not r.get("answered") or A[i] is None: continue
                y.append(bool(r.get("passed")))
                s.append(sum(agree(A[i], A[j]) for j in range(len(g)) if j != i))
    if len(set(y)) < 2: 
        print(f"{fam:18s}{len(y):9d}   (one class only)"); continue
    hi = [yy for yy, ss in zip(y, s) if ss >= 2]; lo = [yy for yy, ss in zip(y, s) if ss == 0]
    print(f"{fam:18s}{len(y):9d}{100*sum(y)/len(y):6.1f}%{roc_auc_score(y, s):16.2f}"
          f"{(100*sum(hi)/len(hi) if hi else float('nan')):23.1f}%{(100*sum(lo)/len(lo) if lo else float('nan')):15.1f}%")
