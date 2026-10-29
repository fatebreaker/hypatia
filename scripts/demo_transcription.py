#!/usr/bin/env python
"""Does the agent TRANSCRIBE the worked example's code, or write its own code for the same procedure?

This decides how interesting the library result is. "One example takes a family from 0% to 94%" means one
thing if the agent adapts a procedure to new data, and something much weaker if it is copying a program and
the new instance happens to accept it.

The answer-copying check is already done (24 distinct answers per 96 rollouts). This is the CODE question,
which is different and sharper: an agent can produce distinct numbers while running an identical program.

Measure, per rollout, over the code the agent actually wrote:
  copy    Jaccard overlap of code LINES with the demo's code -- 1.0 means transcription
  ident   Jaccard over identifiers/calls -- high with low line-overlap means "same procedure, own code"
and compare the demo arm against the control arm on the SAME family. Control has no demo to copy, so its
`copy` score is the floor that any demo-arm number must be read against.
"""
import json, re, sys, glob, os, itertools, statistics as st

FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.S)
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
NOISE = {"import","from","as","def","return","for","in","if","else","print","np","pd","self","the","x","i"}

def code_of(msgs):
    out=[]
    for m in msgs:
        if m.get("role")=="assistant":
            out += FENCE.findall(m.get("content") or "")
    return "\n".join(out)

def lines(c):
    return {l.strip() for l in c.splitlines() if len(l.strip())>12 and not l.strip().startswith("#")}

def idents(c):
    return {t for t in IDENT.findall(c) if t not in NOISE and len(t)>2}

def jac(a,b):
    return len(a&b)/len(a|b) if (a|b) else 0.0

def arm_scores(path, demo_code):
    dl, di = lines(demo_code), idents(demo_code)
    cps, ids, selfsim = [], [], []
    codes=[]
    d=json.load(open(path))
    for rolls in d["groups"].values():
        for r in rolls:
            if not r.get("passed"):      # only solutions that WORK -- a failed rollout's code is noise
                continue
            c=code_of(r.get("messages") or [])
            if not c.strip(): continue
            codes.append(c)
            cps.append(jac(lines(c), dl)); ids.append(jac(idents(c), di))
    for a,b in itertools.islice(itertools.combinations(codes,2), 400):
        selfsim.append(jac(lines(a), lines(b)))
    return cps, ids, selfsim, len(codes)

DEMOS = os.environ.get("DEMOS","/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad/icl_demos")
PAIRS = [("enzyme_mm","enzyme_mm"), ("osc_damped","osc_damped"), ("hc_dx","hc_dx"),
         ("epi_rt","epi_r0"), ("pk_oral","pk_oral")]
print(f"{'family':12s}{'arm':9s}{'n':>4s}{'copy vs demo':>14s}{'identifier':>12s}{'rollout-to-rollout':>20s}")
for fam, demosrc in PAIRS:
    dp=f"{DEMOS}/{demosrc}.json"
    if not os.path.exists(dp): continue
    demo_code=code_of(json.load(open(dp)))
    for arm in ("same","sibling","control"):
        p=f"results/pods/icl_transfer/{arm}_{fam}.json"
        if not os.path.exists(p): continue
        cps, ids, ss, n = arm_scores(p, demo_code)
        if n==0:
            print(f"{fam:12s}{arm:9s}{0:4d}{'-- no passing rollouts --':>46s}"); continue
        print(f"{fam:12s}{arm:9s}{n:4d}{st.mean(cps):13.3f}{st.mean(ids):12.3f}"
              f"{(st.mean(ss) if ss else float('nan')):20.3f}")
print()
print("READING: copy ~1.0 = transcription. copy LOW with identifier HIGH = same procedure, own code.")
print("Control has no demo to copy, so its `copy` column is the floor for chance overlap.")
