"""Synthetic causal-discovery replay traces -- the MITIGATION corpus.

Specialising on ten estimand-given analyses cost -0.167 on QRData's causal
subset while leaving statistics untouched; the interpretation is that the
demonstrations never exercise deciding WHAT causes WHAT. This builds ~260
traces that exercise exactly that, to mix into the ten-family corpus at ~13%.

Two task types:
  pairwise : X,Y from a nonlinear additive-noise model; decide X->Y or Y->X by
             fitting both directions and testing residual-predictor dependence
             (identifiable because f is nonlinear; stated in the trace).
  parents  : which of five earlier-measured candidates DIRECTLY cause Z; the
             candidates contain a mediation chain (C_i -> C_j -> Z) so marginal
             correlation flags an indirect cause that partial regression must
             drop. Teaches conditioning, the skill QRData's causal items need.

Ground-truth hygiene: a trace is emitted ONLY if the scripted method's
conclusion matches the generating DAG -- demonstrations must be correct, and
with nonlinear ANM + |t|>4 thresholds the keep-rate is high. NO QRData content
is used anywhere here: replay is synthetic, the causal test set stays untouched.

Cells are actually EXECUTED (one subprocess, delimiter-split stdout), so the
outputs in the trace are genuine, matching how every other family's traces are
built.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

import numpy as np

SYSTEM = """You are an expert computational biologist and statistician.

You solve a data-analysis problem by writing Python. Rules:

1. To run code, emit exactly one ```python fenced block. You will receive its
   stdout and stderr. State persists across blocks like a notebook.
2. Inspect the data before modelling it. These datasets contain deliberate
   irregularities -- batch effects, ascertainment, winner's curse, ambient
   contamination, pseudogene mismapping, LD structure. Diagnose before you
   estimate, and let what you find change your estimator.
3. Available: numpy, scipy, pandas, statsmodels, scikit-learn. Files are under
   data_files/ in the working directory.
4. When confident, emit your final answer as ONE JSON object in a ```json
   block, with no code block in that same message.

Do not guess numbers. Compute them."""


def run_cells(workdir: Path, cells: list[str]) -> list[str]:
    """Execute cells in ONE interpreter; split stdout on delimiters."""
    delim = "===CELL-BREAK==="
    src = ""
    for i, c in enumerate(cells):
        src += c + f"\nprint({delim!r})\n"
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, cwd=workdir, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    outs = r.stdout.split(delim + "\n")
    return [o.rstrip() for o in outs[:len(cells)]]


def pairwise_instance(rng: np.random.Generator):
    n = int(rng.integers(400, 1200))
    a, b = rng.uniform(0.6, 1.4), rng.uniform(0.4, 1.2) * rng.choice([-1, 1])
    x = rng.uniform(-2, 2, n)
    f = a * x + b * np.tanh(2 * x) + 0.4 * b * x**2
    y = f + rng.normal(0, rng.uniform(0.3, 0.7), n)
    # Semantically NEUTRAL names, so the direction cannot be guessed from domain
    # priors -- and a real flip, so the answer cannot be read off column order.
    # The first version of this generator always made column 1 the cause, which
    # a shortcut learner solves with zero causal reasoning.
    names = [tuple(rng.choice(["assay_A","assay_B","marker_P","marker_Q",
                               "signal_U","signal_V","feature_K","feature_L"],
                              size=2, replace=False))]
    X, Y = names[0]
    if rng.integers(2):
        cols = {X: x, Y: y}; truth = f"{X}->{Y}"
    else:
        cols = {X: y, Y: x}; truth = f"{Y}->{X}"
    return n, cols, truth, (X, Y)


def build_pairwise(rng, idx):
    n, cols, truth, (X, Y) = pairwise_instance(rng)
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td); (wd / "data_files").mkdir()
        pd.DataFrame(cols).to_csv(wd / "data_files" / "pair_obs.csv", index=False)
        c1 = (f"import pandas as pd, numpy as np\n"
              f"df = pd.read_csv('data_files/pair_obs.csv')\n"
              f"print(df.shape); print(df.describe().round(2))")
        c2 = (f"from numpy.polynomial import polynomial as P\n"
              f"def resid_dep(cause, effect):\n"
              f"    C = np.polyfit(df[cause], df[effect], 3)\n"
              f"    r = df[effect] - np.polyval(C, df[cause])\n"
              f"    # OLS makes residuals orthogonal to the predictor's powers, so test the\n"
              f"    # residual MAGNITUDE instead: under the true direction |r| is flat in the\n"
              f"    # predictor, under the reverse it is not.\n"
              f"    d = max(abs(np.corrcoef(np.abs(r), df[cause])[0,1]), abs(np.corrcoef(np.abs(r), df[cause]**2)[0,1]))\n"
              f"    return d\n"
              f"d_xy = resid_dep('{X}','{Y}'); d_yx = resid_dep('{Y}','{X}')\n"
              f"print(f'residual dependence  {X}->{Y}: {{d_xy:.3f}}   {Y}->{X}: {{d_yx:.3f}}')")
        outs = run_cells(wd, [c1, c2])
    m = outs[1].split()
    d_xy, d_yx = float(m[-3].rstrip(':')) if False else None, None
    # parse robustly
    import re
    mm = re.search(r"->.*?: ([0-9.]+)\s+.*?: ([0-9.]+)", outs[1])
    d_xy, d_yx = float(mm.group(1)), float(mm.group(2))
    decided = f"{X}->{Y}" if d_xy < d_yx else f"{Y}->{X}"
    if decided != truth or abs(d_xy - d_yx) < 0.03:
        return None
    task = (f"The CSV file pair_obs.csv records two variables, {X} and {Y}, measured on the same "
            f"{n} experimental units. Exactly one of them causally drives the other through a "
            f"smooth (possibly nonlinear) mechanism with independent additive noise; there is no "
            f"confounding. Determine the causal direction from the data.\n\n"
            f'Return the JSON object in your final answer:\n{{\n"answer": {{\n"direction": "<{X}->{Y} or {Y}->{X}>"\n}},\n'
            f'"reasoning": "<method>"\n}}\n\nFiles in your working directory:\n  data_files/pair_obs.csv')
    r1 = ("Under an additive-noise model the direction is identifiable when the mechanism is "
          "nonlinear: regressing effect on cause leaves residuals independent of the cause, while "
          "the reverse fit leaves structure behind. I start by looking at the data.")
    r2 = ("I fit a cubic in both directions. Ordinary least squares makes the residuals "
          "orthogonal to the predictor's powers in-sample, so raw correlation cannot separate the "
          "directions; the residual MAGNITUDE can. Under the causal direction the noise is "
          "homoscedastic in the predictor, while the anti-causal fit shows structure in |residual|.")
    lo, hi = sorted([d_xy, d_yx])
    fin = (f'The dependence is {lo:.3f} in the {decided} direction against {hi:.3f} the other way, '
           f'so the asymmetry is clear.\n\n```json\n{{\n"answer": {{\n"direction": "{decided}"\n}},\n'
           f'"reasoning": "Fitted cubic regressions in both directions; the {decided} fit leaves residuals '
           f'independent of the predictor while the reverse does not, which under an additive-noise model '
           f'identifies {decided} as causal."\n}}\n```')
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task},
            {"role": "assistant", "content": r1 + "\n\n```python\n" + c1 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[0]},
            {"role": "assistant", "content": r2 + "\n\n```python\n" + c2 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[1]},
            {"role": "assistant", "content": fin}]
    return {"id": f"causal_pair_{idx:04d}", "messages": msgs, "reward": 1.0,
            "n_cells": 2, "plan": "ANM residual-independence, both directions", "_family": "causal_replay"}


def build_parents(rng, idx):
    import pandas as pd
    n = int(rng.integers(500, 1500))
    cand = [f"C{i}" for i in range(1, 6)]
    parents = sorted(rng.choice(cand, size=2, replace=False))
    # mediation: pick a non-parent M0 that drives one parent, so M0 correlates with Z but is indirect
    others = [c for c in cand if c not in parents]
    m0 = others[0]
    data = {}
    data[m0] = rng.normal(0, 1, n)
    for c in cand:
        if c == m0: continue
        if c == parents[0]:
            data[c] = 0.8 * data[m0] + rng.normal(0, 0.7, n)   # mediated path m0 -> parent0 -> Z
        else:
            data[c] = rng.normal(0, 1, n)
    b1, b2 = rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
    z = b1 * data[parents[0]] + b2 * data[parents[1]] + rng.normal(0, 1.0, n)
    df = pd.DataFrame({**data, "Z": z})
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td); (wd / "data_files").mkdir()
        df.to_csv(wd / "data_files" / "screen_obs.csv", index=False)
        c1 = ("import pandas as pd, numpy as np\n"
              "df = pd.read_csv('data_files/screen_obs.csv')\n"
              "print(df.shape)\n"
              "print(df.corr().round(2)['Z'].sort_values(ascending=False))")
        c2 = ("import statsmodels.api as sm\n"
              "X = sm.add_constant(df[['C1','C2','C3','C4','C5']])\n"
              "fit = sm.OLS(df['Z'], X).fit()\n"
              "print(fit.tvalues.round(2))")
        outs = run_cells(wd, [c1, c2])
    import re
    tv = dict(re.findall(r"(C\d)\s+(-?[0-9.]+)", outs[1]))
    picked = sorted([c for c in cand if abs(float(tv[c])) > 4.0])
    if picked != parents:
        return None
    task = (f"The CSV file screen_obs.csv contains five upstream factors C1..C5 and a downstream "
            f"response Z, measured on {n} samples. All five factors are measured before Z and none "
            f"is a descendant of Z, but the factors may influence one another. Determine which of "
            f"the five DIRECTLY cause Z -- a factor whose influence is entirely mediated by another "
            f"factor does not count as direct.\n\n"
            f'Return the JSON object in your final answer:\n{{\n"answer": {{\n"direct_causes": ["<C..>", "<C..>"]\n}},\n'
            f'"reasoning": "<method>"\n}}\n\nFiles in your working directory:\n  data_files/screen_obs.csv')
    r1 = ("Marginal correlation with Z cannot separate direct causes from factors whose effect is "
          "mediated by another factor, so it is only a screen. I look at the correlations first to "
          "see which candidates are entangled with each other.")
    r2 = ("Because every candidate precedes Z, regressing Z on all five simultaneously gives each "
          "coefficient the interpretation of a direct effect holding the others fixed; a mediated "
          "factor's coefficient collapses once its mediator is in the model. I use |t| > 4 as the "
          "cutoff.")
    fin = (f'The joint regression keeps {picked[0]} and {picked[1]} and drops the rest -- including the '
           f'factor whose marginal correlation was inflated through a mediator.\n\n'
           f'```json\n{{\n"answer": {{\n"direct_causes": ["{picked[0]}", "{picked[1]}"]\n}},\n'
           f'"reasoning": "Screened marginal correlations, then regressed Z on all candidates jointly; '
           f'with all factors pre-Z, surviving partial coefficients (|t|>4) identify the direct causes and '
           f'the mediated factor drops out."\n}}\n```')
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task},
            {"role": "assistant", "content": r1 + "\n\n```python\n" + c1 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[0]},
            {"role": "assistant", "content": r2 + "\n\n```python\n" + c2 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[1]},
            {"role": "assistant", "content": fin}]
    return {"id": f"causal_parents_{idx:04d}", "messages": msgs, "reward": 1.0,
            "n_cells": 2, "plan": "correlation screen, then joint partial regression", "_family": "causal_replay"}


def main():
    rng = np.random.default_rng(20260822)
    out = data_root() / "sft" / "causal_replay.jsonl"
    rows, tries = [], 0
    while len(rows) < 260 and tries < 1200:
        tries += 1
        try:
            r = (build_pairwise if tries % 2 == 0 else build_parents)(rng, tries)
        except Exception:
            continue
        if r: rows.append(r)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kinds = sum(1 for r in rows if "pair" in r["id"])
    print(f"wrote {len(rows)} traces ({kinds} pairwise, {len(rows)-kinds} parents) "
          f"from {tries} attempts -> {out}")


# --------------------------------------------------------------------------- #
# TEMPLATE 3: ESTIMAND SELECTION
#
# The first two templates still hand the model its target ("decide the
# direction", "name the direct causes"). Section 6.1 argues the lost capability
# is deciding WHAT TO ESTIMATE, and neither template exercises that. Here the
# question is fixed -- "report the causal effect of T on Y" -- but the correct
# estimator depends on a design fact the model must diagnose: whether treatment
# assignment is balanced across covariates. Randomised -> the crude difference
# is the answer and adjustment is unnecessary; observational -> the crude
# difference is confounded and the covariate-adjusted coefficient is the answer.
#
# Both arms appear with equal probability, the prompt is identical in both, and
# the two estimates differ by enough that answering with the wrong one fails the
# 3% tolerance. A model that has memorised "always adjust" scores ~50%.
# --------------------------------------------------------------------------- #

def build_estimand(rng, idx):
    """Adjust-or-not, where BOTH answers are wrong half the time.

    A corpus of only-confounded cases teaches "always adjust" -- a procedure,
    not judgment, and exactly the shortcut Section 6.1 predicts SFT will latch
    onto. So the two arms are made symmetric:

      confounded   : the extra variable is measured at BASELINE and predicts
                     treatment -> it is a confounder, adjust.
      mediated     : the extra variable is measured AFTER treatment and is
                     caused by it -> adjusting blocks part of the effect and
                     biases the estimate, do NOT adjust.

    In both arms the variable is strongly associated with treatment, so
    association alone cannot decide; the measurement timing stated in the prompt
    is what distinguishes them. Both arms are kept only when crude and adjusted
    differ enough that picking the wrong one fails the 3% tolerance, so "always
    adjust" and "never adjust" each score ~50%.
    """
    import pandas as pd
    import statsmodels.api as sm
    n = int(rng.integers(600, 1600))
    mediated = bool(rng.integers(2))
    eff = float(rng.uniform(1.0, 2.5)) * rng.choice([-1.0, 1.0])
    vname, vlabel = rng.choice([("plasma_marker", "plasma marker"), ("cell_count", "cell count"),
                                ("score_index", "composite score"), ("enzyme_level", "enzyme level")])
    W = rng.normal(0, 1, n)
    if mediated:
        T = rng.binomial(1, 0.5, n).astype(float)          # randomised
        a, b = 1.4, 1.3
        V = a * T + rng.normal(0, 0.8, n)                   # measured AFTER treatment
        Y = eff * T + b * V + 0.4 * W + rng.normal(0, 1.0, n)
        # crude estimates the TOTAL effect (direct + the path through V), which
        # is the causal quantity asked for -- not the direct coefficient `eff`.
        target = eff + a * b
        timing = f"{vname} was measured AFTER treatment was administered"
    else:
        V = rng.normal(0, 1, n)                             # measured at BASELINE
        T = rng.binomial(1, 1 / (1 + np.exp(-1.4 * V))).astype(float)
        Y = eff * T + 1.7 * V + 0.4 * W + rng.normal(0, 1.0, n)
        target = eff
        timing = f"{vname} was measured at BASELINE, before treatment assignment"
    df = pd.DataFrame({"T": T, "Y": Y, vname: V, "covar_W": W})
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td); (wd / "data_files").mkdir()
        df.to_csv(wd / "data_files" / "trial_obs.csv", index=False)
        c1 = ("import pandas as pd, numpy as np\n"
              "df = pd.read_csv('data_files/trial_obs.csv')\n"
              "print(df.shape); print(df.columns.tolist())\n"
              f"print(df.groupby('T')['{vname}'].mean().round(3))")
        c2 = ("import statsmodels.api as sm\n"
              "crude = sm.OLS(df['Y'], sm.add_constant(df[['T']])).fit()\n"
              f"adj = sm.OLS(df['Y'], sm.add_constant(df[['T','{vname}','covar_W']])).fit()\n"
              "print(f\"crude    T = {crude.params['T']:.4f}\")\n"
              "print(f\"adjusted T = {adj.params['T']:.4f}\")")
        outs = run_cells(wd, [c1, c2])
    import re
    crude = float(re.search(r"crude\s+T = (-?[0-9.]+)", outs[1]).group(1))
    adj = float(re.search(r"adjusted T = (-?[0-9.]+)", outs[1]).group(1))
    chosen, other = (crude, adj) if mediated else (adj, crude)
    if abs(crude - adj) < 0.20 or abs(chosen - target) > 0.30:
        return None
    task = (f"The CSV file trial_obs.csv records a binary treatment T, an outcome Y, a variable "
            f"{vname}, and a baseline covariate covar_W for {n} units. Note the study design: "
            f"{timing}. Report the causal effect of T on Y. Decide which estimator is appropriate "
            f"before computing it.\n\n"
            f'Return the JSON object in your final answer:\n{{\n"answer": {{\n"causal_effect": <float>\n}},\n'
            f'"reasoning": "<which estimator and why>"\n}}\n\nFiles in your working directory:\n  data_files/trial_obs.csv')
    r1 = (f"The target is fixed -- the causal effect of T on Y -- but the estimator is not, and it turns "
          f"on what {vlabel} is relative to the treatment. Conditioning on a baseline confounder removes "
          f"bias; conditioning on something the treatment itself caused removes part of the effect I am "
          f"trying to measure. The design note says {timing.lower()}, so I look at the data with that in "
          f"mind rather than adjusting reflexively.")
    if mediated:
        r2 = (f"{vlabel} differs sharply by treatment arm, which is expected because treatment caused it. "
              f"That makes it a mediator, not a confounder: adjusting for it would block part of the causal "
              f"path. I compute both so the difference is visible, then report the unadjusted contrast.")
        why = (f"{vname} was measured after treatment and is caused by it, so it is a mediator; adjusting "
               f"for it blocks part of the effect. The unadjusted contrast is the total causal effect")
        label = "unadjusted"
    else:
        r2 = (f"{vlabel} is a baseline variable that predicts treatment assignment, so it confounds the "
              f"crude comparison. I compute both and report the adjusted coefficient.")
        why = (f"{vname} was measured at baseline and predicts assignment, so it is a confounder; the "
               f"crude contrast is biased and the covariate-adjusted coefficient is the causal estimand")
        label = "covariate-adjusted"
    fin = (f"So the correct estimand is the {label} contrast, {chosen:.4f}; the other estimator gives "
           f"{other:.4f}, which would be wrong here.\n\n```json\n{{\n\"answer\": {{\n"
           f"\"causal_effect\": {chosen:.4f}\n}},\n\"reasoning\": \"{why}.\"\n}}\n```")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task},
            {"role": "assistant", "content": r1 + "\n\n```python\n" + c1 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[0]},
            {"role": "assistant", "content": r2 + "\n\n```python\n" + c2 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[1]},
            {"role": "assistant", "content": fin}]
    return {"id": f"causal_estimand_{idx:04d}", "messages": msgs, "reward": 1.0, "n_cells": 2,
            "plan": "classify the covariate as confounder vs mediator, then pick the estimator",
            "_family": "causal_replay", "_mediated": mediated}


def main_v2():
    """Larger, three-template corpus for the dose-response arm of Section 6.2."""
    rng = np.random.default_rng(20260823)
    out = data_root() / "sft" / "causal_replay_v2.jsonl"
    builders = [build_pairwise, build_parents, build_estimand]
    rows, tries = [], 0
    while len(rows) < 620 and tries < 4000:
        tries += 1
        try:
            r = builders[tries % 3](rng, tries)
        except Exception:
            continue
        if r: rows.append(r)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    import collections
    kinds = collections.Counter(r["id"].split("_")[1] for r in rows)
    rnd = sum(1 for r in rows if r.get("_mediated"))
    est = kinds.get("estimand", 0)
    print(f"wrote {len(rows)} traces {dict(kinds)} from {tries} attempts")
    print(f"  estimand arm balance: {rnd} mediated (do-not-adjust) / {est - rnd} confounded (adjust)")


# --------------------------------------------------------------------------- #
# BALANCED REPLAY (review 4, Q7)
#
# Every trace above ends in a relationship, so replay never shows the model
# concluding that two variables are unrelated -- the one answer fine-tuning took
# away on QRData's causal-discovery questions. Here half of the traces are
# pairwise tasks whose two variables are independent. Every pairwise prompt
# admits "none", and every pairwise trace tests for dependence (linear, rank and
# distance correlation) before it considers a direction, answering "none" when
# no test detects any. The other half keeps the original mix (pairwise with a
# direction, parents), so the null share is the only change.
# --------------------------------------------------------------------------- #

NAMES = ["assay_A", "assay_B", "marker_P", "marker_Q", "signal_U", "signal_V", "feature_K", "feature_L"]


def build_pairwise3(rng, idx, null: bool):
    """Pairwise task with three possible answers: X->Y, Y->X or none."""
    import re
    import pandas as pd
    n = int(rng.integers(400, 1200))

    def mechanism(u):
        a, b = rng.uniform(0.6, 1.4), rng.uniform(0.4, 1.2) * rng.choice([-1, 1])
        return a * u + b * np.tanh(2 * u) + 0.4 * b * u**2

    x = rng.uniform(-2, 2, n)
    # null: the same kind of marginal, driven by an independent draw, so the answer cannot be read off the shape
    y = mechanism(rng.uniform(-2, 2, n) if null else x) + rng.normal(0, rng.uniform(0.3, 0.7), n)
    X, Y = rng.choice(NAMES, size=2, replace=False)
    if rng.integers(2):
        cols = {X: x, Y: y}; truth = "none" if null else f"{X}->{Y}"
    else:
        cols = {X: y, Y: x}; truth = "none" if null else f"{Y}->{X}"
    c1 = ("import pandas as pd, numpy as np\n"
          "df = pd.read_csv('data_files/pair_obs.csv')\n"
          "print(df.shape); print(df.describe().round(2))")
    c2 = ("from scipy import stats\n"
          f"a, b = df['{X}'].to_numpy(), df['{Y}'].to_numpy()\n"
          "r, p_r = stats.pearsonr(a, b); rho, p_s = stats.spearmanr(a, b)\n"
          "def centred(v):\n"
          "    d = np.abs(v[:, None] - v[None, :])\n"
          "    return d - d.mean(0) - d.mean(1)[:, None] + d.mean()\n"
          "A, B = centred(a), centred(b)\n"
          "dcov = (A * B).mean(); dcor = np.sqrt(dcov / np.sqrt((A * A).mean() * (B * B).mean()))\n"
          "g = np.random.default_rng(0); perm = []\n"
          "for _ in range(199):\n"
          "    i = g.permutation(len(b)); perm.append((A * B[i][:, i]).mean())\n"
          "p_d = (1 + sum(v >= dcov for v in perm)) / 200\n"
          "print(f'pearson r = {r:.3f} (p = {p_r:.2g}); spearman rho = {rho:.3f} (p = {p_s:.2g}); '\n"
          "      f'distance correlation = {dcor:.3f} (permutation p = {p_d:.3f})')")
    c3 = ("def resid_dep(cause, effect):\n"
          "    C = np.polyfit(df[cause], df[effect], 3)\n"
          "    r = df[effect] - np.polyval(C, df[cause])\n"
          "    # OLS makes residuals orthogonal to the predictor's powers, so test the\n"
          "    # residual MAGNITUDE instead: under the true direction |r| is flat in the\n"
          "    # predictor, under the reverse it is not.\n"
          "    return max(abs(np.corrcoef(np.abs(r), df[cause])[0,1]), abs(np.corrcoef(np.abs(r), df[cause]**2)[0,1]))\n"
          f"d_xy = resid_dep('{X}','{Y}'); d_yx = resid_dep('{Y}','{X}')\n"
          f"print(f'residual dependence  {X}->{Y}: {{d_xy:.3f}}   {Y}->{X}: {{d_yx:.3f}}')")
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td); (wd / "data_files").mkdir()
        pd.DataFrame(cols).to_csv(wd / "data_files" / "pair_obs.csv", index=False)
        outs = run_cells(wd, [c1, c2, c3])
    ps = [float(v) for v in re.findall(r"p = ([0-9.e+-]+)\)", outs[1])]
    assert len(ps) == 3, outs[1]
    dependent = min(ps) < 0.01
    if dependent:
        mm = re.search(r"->.*?: ([0-9.]+)\s+.*?: ([0-9.]+)", outs[2])
        d_xy, d_yx = float(mm.group(1)), float(mm.group(2))
        decided = f"{X}->{Y}" if d_xy < d_yx else f"{Y}->{X}"
        if abs(d_xy - d_yx) < 0.03:
            return None
    else:
        decided = "none"
    if decided != truth:          # a demonstration must reach the generating truth
        return None
    task = (f"The CSV file pair_obs.csv records two variables, {X} and {Y}, measured on the same "
            f"{n} experimental units. Either one of them causally drives the other through a smooth "
            f"(possibly nonlinear) mechanism with independent additive noise, or the two are causally "
            f"unrelated; there is no confounding. Determine the causal relationship from the data.\n\n"
            f'Return the JSON object in your final answer:\n{{\n"answer": {{\n"relationship": "<{X}->{Y}, {Y}->{X} or none>"\n}},\n'
            f'"reasoning": "<method>"\n}}\n\nFiles in your working directory:\n  data_files/pair_obs.csv')
    r1 = ("A causal link through a smooth mechanism makes the two variables dependent, so the first "
          "question is whether they are dependent at all; only then is a direction worth deciding. "
          "I start by looking at the data.")
    r2 = ("I test for dependence three ways: Pearson for a linear trend, Spearman for a monotone one, "
          "and distance correlation with a permutation test for any other shape. If none detects "
          "dependence, the data give no reason to posit a causal relationship.")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task},
            {"role": "assistant", "content": r1 + "\n\n```python\n" + c1 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[0]},
            {"role": "assistant", "content": r2 + "\n\n```python\n" + c2 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[1]}]
    if dependent:
        r3 = ("The dependence is clear, so the question is the direction. Under an additive-noise model "
              "it is identifiable when the mechanism is nonlinear. I fit a cubic in both directions; "
              "least squares makes the residuals orthogonal to the predictor's powers in-sample, so I "
              "test the residual MAGNITUDE, which is flat in the predictor only in the causal direction.")
        lo, hi = sorted([d_xy, d_yx])
        fin = (f"The residual dependence is {lo:.3f} in the {decided} direction against {hi:.3f} the other "
               f"way, so the asymmetry is clear.\n\n```json\n{{\n\"answer\": {{\n\"relationship\": \"{decided}\"\n}},\n"
               f"\"reasoning\": \"The variables are clearly dependent; cubic fits in both directions leave residuals "
               f"independent of the predictor only for {decided}, which under an additive-noise model identifies "
               f"it as causal.\"\n}}\n```")
        msgs += [{"role": "assistant", "content": r3 + "\n\n```python\n" + c3 + "\n```"},
                 {"role": "user", "content": "STDOUT:\n" + outs[2]}]
    else:
        fin = (f"None of the three tests detects dependence (smallest p = {min(ps):.2g}), so the data give no "
               f"evidence that either variable drives the other.\n\n```json\n{{\n\"answer\": {{\n"
               f"\"relationship\": \"none\"\n}},\n\"reasoning\": \"Pearson, Spearman and a distance-correlation "
               f"permutation test all fail to detect dependence, so there is no evidence of a causal "
               f"relationship.\"\n}}\n```")
    msgs.append({"role": "assistant", "content": fin})
    return {"id": f"causal_{'null' if null else 'pair3'}_{idx:04d}", "messages": msgs, "reward": 1.0,
            "n_cells": 3 if dependent else 2, "plan": "test dependence, then (if dependent) ANM direction",
            "_family": "causal_replay"}


def main_balanced():
    """Balanced replay: 130 null pairwise, 65 pairwise with a direction, 65 parents (review 4, Q7).

    Writes sft/causal_replay_bal.jsonl and two corpora for rl/sft_train.py --corpus: replaybal (the balanced
    traces alone) and tenfam_replaybal (the ten-family corpus plus them, 13% as in tenfam_replay); 12 traces
    are held out for validation, as there.
    """
    rng = np.random.default_rng(20260914)
    sft = data_root() / "sft"
    want = {"null": 130, "pair3": 65, "parents": 65}
    rows = {k: [] for k in want}
    tries = 0
    while any(len(rows[k]) < want[k] for k in want) and tries < 4000:
        tries += 1
        k = [k for k in want if len(rows[k]) < want[k]][tries % sum(len(rows[k]) < want[k] for k in want)]
        try:
            r = build_parents(rng, tries) if k == "parents" else build_pairwise3(rng, tries, null=(k == "null"))
        except Exception:
            continue
        if r:
            rows[k].append(r)
    got = [r for k in want for r in rows[k]]
    got = [got[i] for i in rng.permutation(len(got))]
    dump = lambda rs: "".join(json.dumps(r) + "\n" for r in rs)   # noqa: E731
    (sft / "causal_replay_bal.jsonl").write_text(dump(got))
    val, train = got[:12], got[12:]
    (sft / "replaybal_train.jsonl").write_text(dump(train))
    (sft / "replaybal_val.jsonl").write_text(dump(val))
    (sft / "tenfam_replaybal_train.jsonl").write_text((sft / "tenfam_train.jsonl").read_text() + dump(train))
    (sft / "tenfam_replaybal_val.jsonl").write_text((sft / "tenfam_val.jsonl").read_text() + dump(val))
    print(f"wrote {len(got)} traces {({k: len(v) for k, v in rows.items()})} from {tries} attempts "
          f"-> causal_replay_bal.jsonl, replaybal_*, tenfam_replaybal_* ({len(train)} + {len(val)} replay)")


if __name__ == "__main__":
    {"--v2": main_v2, "--balanced": main_balanced}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
