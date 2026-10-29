"""mixed_v05.py (10-28; the 10-27 mock review's concern 3; descriptive -- changes no registered verdict). A model-based check of every
Table 1 comparison that rigor_v05.py bootstraps: a linear mixed model of per-problem correctness with crossed random intercepts for
PROBLEM and RUN,
    y[problem, run] = b0 + b1 * [trained arm] + u[problem] + v[run] + e,    u ~ N(0, s_p^2), v ~ N(0, s_r^2),
fitted by REML (statsmodels MixedLM, one group, problem and run as variance components). b1 x 100 is the difference in accuracy points
(the same scale as Table 1's d); its 95% Wald interval is set beside the hierarchical bootstrap's (figdata_rigor_v05.csv), and each
verdict is re-taken on it: requirement 1 (b1 >= 3 and lower bound > 0) and level (b1 >= -3 and upper bound >= 0).
The reference system (untrained model, Claude Opus 5.5, GPT-6 Sol) is one read, so the run variance is estimated from the arm's runs.
    venv/bin/python docs/rl4sci/v05/mixed_v05.py     -> docs/rl4sci/v05/figdata_mixed_v05.csv  (repo root; CPU agent)
"""
import contextlib
import csv
import io
import os
import sys
import warnings

import pandas as pd
import statsmodels.formula.api as smf

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
with contextlib.redirect_stdout(io.StringIO()):
    import build_table1 as bt      # noqa: E402

warnings.simplefilter("ignore")


def fit(runs, ref, b):
    """runs: the arm's per-run {problem: correct}; ref: one read. Returns (b1, lo, hi, p, sd_problem, sd_run) in points."""
    ids = bt.IDS[b]; rows = []
    for k, r in enumerate(runs):
        rows += [(i, f"arm{k}", 1, float(r.get(i, 0))) for i in ids]
    rows += [(i, "ref", 0, float(ref.get(i, 0))) for i in ids]
    df = pd.DataFrame(rows, columns=["problem", "run", "arm", "y"]); df["g"] = 1
    m = smf.mixedlm("y ~ arm", df, groups="g", re_formula="0",
                    vc_formula={"problem": "0 + C(problem)", "run": "0 + C(run)"}).fit(reml=True)
    lo, hi = m.conf_int().loc["arm"]
    vc = dict(zip(m.model.exog_vc.names, m.vcomp))
    return (100 * m.params["arm"], 100 * lo, 100 * hi, float(m.pvalues["arm"]),
            100 * vc.get("problem", 0) ** 0.5, 100 * vc.get("run", 0) ** 0.5)


boot = {}
for r in csv.DictReader(open(os.path.join(D, "figdata_rigor_v05.csv"))):
    boot[(r["row"], r["bench"], r["ref"])] = r
claude = {b: bt.load("frontier_api/claude-opus-5-5-high", b, bt.API_SUB) for b in bt.BENCHES}
sol = {b: bt.load(bt.SOL_SETTINGS["high"], b, bt.API_SUB) for b in bt.BENCHES}
taught = {b: ([bt.TAUGHT9[b]] if b in bt.TAUGHT9 else bt.RL9_LIST[b]) for b in bt.BENCHES}
ROWS = [("9B + RL", bt.RL9_LIST, "Qwen3.5-9B"), ("9B distilled", taught, "Qwen3.5-9B"), ("4B + RL", bt.RL4, "Qwen3.5-4B")]
out = []
for row, spec, refname in ROWS:
    for b in bt.BENCHES:
        disp, runs, arm, mean = bt.cell(spec[b], b)
        if len(runs) < 3:
            print(f"{row} {bt.NAME[b]}: {arm} incomplete ({len(runs)}/3)"); continue
        comps = [("R1", refname, bt.base_rows[refname][b][1])]
        if row != "4B + RL":
            comps += [("level", "Claude Opus 5.5 (high)", claude[b]), ("level", "GPT-6 Sol (high)", sol[b])]
        for kind, lab, ref in comps:
            if ref is None:
                continue
            b1, lo, hi, p, sp, sr = fit(runs, ref, b)
            bo = boot.get((row, b, lab), {})
            blo, bhi = (float(bo["lo95"]), float(bo["hi95"])) if bo.get("lo95") else (None, None)
            if kind == "R1":
                v_lmm = b1 >= 3 and lo > 0
                v_boot = None if blo is None else (float(bo["d"]) >= 3 and blo > 0)
            else:
                v_lmm = b1 >= -3 and hi >= 0
                v_boot = None if bhi is None else (float(bo["d"]) >= -3 and bhi >= 0)
            out.append(dict(row=row, bench=b, arm=arm, kind=kind, ref=lab, runs=len(runs), d_lmm=round(b1, 2), lo_lmm=round(lo, 2),
                            hi_lmm=round(hi, 2), p_lmm=round(p, 5), sd_problem=round(sp, 1), sd_run=round(sr, 2),
                            d_boot=bo.get("d"), lo_boot=blo, hi_boot=bhi, verdict_lmm=v_lmm, verdict_boot=v_boot,
                            agree=(v_boot is None) or (v_lmm == v_boot)))
            r = out[-1]
            print(f"{kind:5s} {row:12s} {bt.NAME[b]:14s} {arm:10s} vs {lab:24s} LMM {b1:+6.1f} [{lo:+.1f},{hi:+.1f}] p {p:.4f} "
                  f"(sd problem {sp:.0f}, run {sr:.1f}) | boot [{blo},{bhi}] | {'MET' if v_lmm else 'not'} / "
                  f"{'MET' if v_boot else ('not' if v_boot is not None else '?')}{'' if r['agree'] else '  <-- DIFFERS'}")
with open(os.path.join(D, "figdata_mixed_v05.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
n_dis = sum(not r["agree"] for r in out)
print(f"{len(out)} comparisons; the mixed model and the bootstrap disagree on {n_dis} verdict(s); wrote figdata_mixed_v05.csv")
