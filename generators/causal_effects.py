#!/usr/bin/env python
"""Second scientific domain: causal-effect estimation, five generated families.

The genomics families transcribe GeneBench-Pro's data-generating processes; these
transcribe the textbook designs behind CauSciBench's real split (applied economics
and social science; experiments/second_domain/). Each family has a reference
analysis and documented wrong analyses, and every instance is validated the way the
genomics generators are:

  * the ground truth is the REFERENCE ESTIMATOR APPLIED TO THE SAMPLED DATA, not the
    latent parameter, so a correct analysis reproduces it to floating-point precision;
  * every documented wrong analysis (a trap) must land outside tolerance -- a draw
    whose trap is dead is strengthened and re-drawn, as generators/emit.py does;
  * the ablation table (value, the pipeline step the wrong analysis skips, and its
    deviation) is stored in `_provenance.ablations`, so rl/reward.py's step-attributed
    partial credit works unchanged.

Tolerance per instance: max(0.10 |gold|, 0.5 SE_gold), the rule used to validate
CauSciBench's real split. The task text is CauSciBench's own prompt content
(TASK_TEMPLATE, shared with experiments/second_domain/build_causcibench.py) so
generated and real instances ask in the same words.

  family         reference analysis                         traps (fails_at_step)
  confounded     OLS on treatment + pre-treatment confounders  no adjustment (1), post-treatment
                                                               bad control (2), a confounder omitted (2)
  iv             2SLS, instrument + covariates               OLS ignoring the instrument (1), the reduced
                                                               form (2), 2SLS without covariates (2)
  did            two-way fixed effects, single adoption      post-period treated-control gap (1),
                                                               treated before/after (1)
  rdd            local linear, triangular kernel, each side  global linear fit (1), whole-sample
                                                               difference in means (1)
  att            ATT by outcome-model imputation from controls  no adjustment (1), additive OLS (2),
                                                               the ATE instead of the ATT (3)

  venv/bin/python generators/causal_effects.py --family iv --split train --n 400
  venv/bin/python generators/causal_effects.py --family all --split train --mix   # RL pool causal_mix
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METHOD_MENU = """Here are some example methods; you can choose one from them: [
    IPW (Inverse Probability Weighting): Choose the right estimand (ATE/ATT/ATC), and compute the causal effect,
    Linear regression with control variables: Build a regression model with the treatment, outcome, and confounders/control variables, and compute the causal effects,
    Instrumental variable: Build an instrumental variable model, and compute the causal effects associated with the treatment variable,
    Matching: Choose the correct estimand (ATE/ATT/ATC), and match accordingly, and then compute the causal effects,
    Difference-in-differences: Build a difference-in-differences model, and output the coefficient of the variable of interest,
    Regression discontinuity design: Build a regression discontinuity design model, and output the coefficient of the variable of interest,
    Linear regression / difference-in-means: Either build a regression model consisting of the treatment and outcome variables, and compute the coefficient associated with the treatment variable or compute the difference in means across treatment and control groups,
    Generalized linear models / GLM: Build a GLM model, and output the coefficient of the variable of interest,
    Frontdoor adjustment: Build a causal graph, identify a mediator variable between the treatment and outcome, check for frontdoor criterion, and compute the causal effect using the frontdoor adjustment formula]"""


def task_text(desc: str, query: str, csv: str) -> str:
    """CauSciBench's prompt content, with the harness's JSON answer in place of its
    print-14-fields instructions. Shared by the generated and the real instances."""
    return (f"You are an expert in statistics and causal reasoning. You will answer a causal question on a tabular dataset.\n\n"
            f"The dataset is located at: data_files/{csv}.\n\n"
            f"The dataset has the following description:\n```\n{desc.strip()}\n```\n\n"
            f"The causal question I would like you to answer is:\n```\n{query.strip()}\n```\n\n"
            f"{METHOD_MENU}\n\n"
            f"Estimate the causal effect that answers the question from the data. "
            f'Return your answer as JSON: {{"answer": {{"effect": <value>}}}}')


# ----------------------------------------------------------------------------- estimators
def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and HC1 standard errors; X must include the intercept."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    meat = (X * e[:, None] ** 2).T @ X
    V = bread @ meat @ bread * n / (n - k)
    return beta, np.sqrt(np.diag(V))


def tsls(y, T, Z, W) -> tuple[float, float]:
    """2SLS coefficient on T with instrument Z and exogenous W (intercept included in W)."""
    Xz = np.column_stack([Z, W])
    that = Xz @ np.linalg.lstsq(Xz, T, rcond=None)[0]
    X2 = np.column_stack([that, W])
    b = np.linalg.lstsq(X2, y, rcond=None)[0]
    e = y - np.column_stack([T, W]) @ b                   # structural residuals
    n, k = X2.shape
    bread = np.linalg.pinv(X2.T @ X2)
    V = bread @ ((X2 * e[:, None] ** 2).T @ X2) @ bread * n / (n - k)
    return float(b[0]), float(np.sqrt(V[0, 0]))


def c1(*cols) -> np.ndarray:
    return np.column_stack([np.ones(len(cols[0]))] + list(cols))


def local_linear(y, r, cut, h, kernel="triangular") -> tuple[float, float]:
    """Sharp-RD jump: weighted linear fit on each side of the cutoff within h."""
    x = r - cut
    keep = np.abs(x) <= h
    w = (1 - np.abs(x[keep]) / h) if kernel == "triangular" else np.ones(keep.sum())
    d = (x[keep] >= 0).astype(float)
    X = c1(d, x[keep], d * x[keep])
    sw = np.sqrt(w)
    b, se = ols(y[keep] * sw, X * sw[:, None])
    return float(b[1]), float(se[1])


# ----------------------------------------------------------------------------- contexts
CONFOUNDED = [
    dict(unit="adults in a regional labour survey", T="training", Tdesc="1 if the person completed a job-training programme",
         Y="earnings", Ydesc="annual earnings in thousands of dollars one year later",
         X=[("age", "age in years"), ("educ", "years of schooling"), ("prior_earn", "earnings the year before the programme, thousands"),
            ("married", "1 if married")],
         M=("hours_worked", "weekly hours worked, measured after the programme"), scale=8.0),
    dict(unit="patients admitted to a network of hospitals", T="new_protocol", Tdesc="1 if the patient was treated under the new care protocol",
         Y="recovery_score", Ydesc="functional recovery score at discharge (0-100)",
         X=[("age", "age in years"), ("severity", "admission severity index"), ("comorbid", "number of chronic conditions"),
            ("female", "1 if female")],
         M=("length_of_stay", "days in hospital after the protocol decision"), scale=6.0),
    dict(unit="small firms in a manufacturing census", T="subsidy", Tdesc="1 if the firm received an investment subsidy",
         Y="output", Ydesc="value added the following year, millions",
         X=[("employees", "employees before the subsidy"), ("capital", "capital stock before the subsidy, millions"),
            ("age_firm", "firm age in years"), ("exporter", "1 if the firm exported before the subsidy")],
         M=("new_machines", "machines purchased after the subsidy decision"), scale=2.0),
    dict(unit="students in a school district", T="tutoring", Tdesc="1 if the student enrolled in after-school tutoring",
         Y="math_score", Ydesc="end-of-year standardised maths score",
         X=[("prior_score", "maths score the previous year"), ("low_income", "1 if eligible for free lunch"),
            ("attendance", "attendance rate the previous year"), ("grade", "grade level")],
         M=("homework_hours", "weekly homework hours reported after enrolment"), scale=5.0),
]
IV = [
    dict(unit="workers in a national household survey", T="educ", Tdesc="years of schooling",
         Y="log_wage", Ydesc="log hourly wage", Z="college_nearby", Zdesc="1 if a four-year college was within commuting distance of the person's childhood home",
         W=[("age", "age in years"), ("urban", "1 if the person grew up in an urban area"), ("parent_educ", "parents' mean years of schooling")],
         binary_z=True, scale=0.08),
    dict(unit="households offered a savings product", T="deposits", Tdesc="monthly deposits into the savings account, dollars",
         Y="assets", Ydesc="household assets after two years, hundreds of dollars", Z="sms_reminder", Zdesc="1 if the household was randomly assigned to receive SMS reminders",
         W=[("income", "monthly income, hundreds of dollars"), ("hh_size", "household size"), ("rural", "1 if rural")],
         binary_z=True, scale=0.9),
    dict(unit="counties in an agricultural panel", T="irrigation", Tdesc="share of farmland irrigated",
         Y="yield", Ydesc="crop yield, tonnes per hectare", Z="aquifer_depth", Zdesc="depth to the aquifer in metres (deeper aquifers make irrigation costlier)",
         W=[("rainfall", "average annual rainfall, hundreds of mm"), ("soil_quality", "soil quality index"), ("farm_size", "average farm size, hectares")],
         binary_z=False, scale=3.0),
    dict(unit="patients in an observational cohort", T="dose", Tdesc="daily dose of the medication, mg",
         Y="blood_pressure", Ydesc="systolic blood pressure after six months, mmHg", Z="physician_pref", Zdesc="the prescribing physician's preference for the drug (share of their patients prescribed it)",
         W=[("age", "age in years"), ("bmi", "body-mass index"), ("baseline_bp", "systolic blood pressure at enrolment")],
         binary_z=False, scale=-0.25),
]
DID = [
    dict(unit="counties", T="minimum wage increase", Y="teen_employment", Ydesc="teen employment rate (percent)",
         treat="treated_state", tdesc="1 for counties in the state that raised its minimum wage", period="quarter", scale=-1.5),
    dict(unit="hospitals", T="price-transparency rule", Y="avg_price", Ydesc="average procedure price, hundreds of dollars",
         treat="covered", tdesc="1 for hospitals covered by the transparency rule", period="year", scale=-2.0),
    dict(unit="school districts", T="class-size reduction", Y="reading_score", Ydesc="average reading score",
         treat="reform_district", tdesc="1 for districts that adopted the class-size reduction", period="year", scale=4.0),
    dict(unit="municipalities", T="bike-lane programme", Y="commute_minutes", Ydesc="average commute time in minutes",
         treat="program_city", tdesc="1 for municipalities that built the bike-lane network", period="month", scale=-3.0),
]
RDD = [
    dict(unit="students applying for a merit scholarship", R="entrance_score", Rdesc="entrance exam score; students scoring at or above the cutoff received the scholarship",
         Y="graduated_gpa", Ydesc="GPA at graduation", cut=70.0, span=30.0, scale=0.3),
    dict(unit="municipal elections", R="vote_margin", Rdesc="the incumbent party's vote margin in percentage points; the party wins the seat when the margin is at or above zero",
         Y="next_turnout", Ydesc="turnout in the next election (percent)", cut=0.0, span=20.0, scale=3.0),
    dict(unit="firms applying for a small-business grant", R="application_score", Rdesc="application review score; firms scoring at or above the threshold were funded",
         Y="revenue_growth", Ydesc="two-year revenue growth (percent)", cut=60.0, span=25.0, scale=6.0),
    dict(unit="older adults near a pension eligibility age", R="age_months", Rdesc="age in months relative to the eligibility date (eligible at zero and above)",
         Y="hours_worked", Ydesc="weekly hours worked", cut=0.0, span=36.0, scale=-4.0),
]
ATT = [
    dict(unit="unemployed workers", T="enrolled", Tdesc="1 if the worker enrolled in the wage-subsidy programme",
         Y="months_employed", Ydesc="months employed during the following year",
         X=[("age", "age in years"), ("skill", "pre-programme skills test score"), ("tenure", "years of previous job tenure")],
         query="What was the effect of the wage-subsidy programme on months employed for the workers who enrolled in it?", scale=2.0),
    dict(unit="farmers in a development project", T="adopted", Tdesc="1 if the farmer adopted the improved seed variety",
         Y="harvest", Ydesc="harvest value, hundreds of dollars",
         X=[("land", "farm size, hectares"), ("experience", "years farming"), ("credit", "access to credit score")],
         query="What was the effect of adopting the improved seed on harvest value among the farmers who adopted it?", scale=4.0),
    dict(unit="customers of an online retailer", T="loyalty", Tdesc="1 if the customer joined the loyalty programme",
         Y="annual_spend", Ydesc="annual spending, hundreds of dollars",
         X=[("tenure_years", "years as a customer"), ("prior_spend", "spending the year before, hundreds"), ("app_user", "1 if the customer uses the app")],
         query="How much did joining the loyalty programme change annual spending for the customers who joined?", scale=3.0),
]


def _vars(lines) -> str:
    return "\n".join(f"{n}: {d}" for n, d in lines)


# ----------------------------------------------------------------------------- families
def gen_confounded(rng, k: float):
    c = CONFOUNDED[rng.integers(len(CONFOUNDED))]
    n = int(rng.integers(1500, 4000))
    p = len(c["X"])
    X = rng.normal(size=(n, p))
    X[:, -1] = (X[:, -1] > rng.normal(0, 0.5)).astype(float)
    a = rng.normal(0, 0.6 * k, p) + np.sign(rng.normal(size=p)) * 0.4 * k
    T = (rng.random(n) < 1 / (1 + np.exp(-(X @ a - 0.2)))).astype(float)
    tau_d = c["scale"] * rng.uniform(0.4, 1.0) * rng.choice([-1, 1])
    gam, dlt = rng.uniform(0.8, 1.6) * k, c["scale"] * rng.uniform(0.4, 0.9) * rng.choice([-1, 1])
    M = gam * T + 0.5 * X[:, 0] + rng.normal(size=n)
    beta = c["scale"] * rng.normal(0, 0.9, p) * k
    Y = tau_d * T + dlt * M + X @ beta + rng.normal(0, abs(c["scale"]) * 1.2, n)
    # rescale covariates to plausible units
    cols = {name: np.round(10 + 3 * X[:, j], 2) if j < p - 1 else X[:, j].astype(int) for j, (name, _) in enumerate(c["X"])}
    df = pd.DataFrame({c["T"]: T.astype(int), **cols, c["M"][0]: np.round(M + 5, 3), c["Y"]: np.round(Y + 50, 3)})
    Xa = np.column_stack([df[nm].to_numpy(float) for nm, _ in c["X"]])
    y, t, m = df[c["Y"]].to_numpy(float), df[c["T"]].to_numpy(float), df[c["M"][0]].to_numpy(float)
    b, se = ols(y, c1(t, Xa))
    gold, gse = float(b[1]), float(se[1])
    # the confounder the outcome loads on most: omitting it is the named trap
    j = int(np.argmax(np.abs(ols(y, c1(t, Xa))[0][2:]) * Xa.std(0)))
    abl = {"reference": (gold, None),
           "no_adjustment": (float(y[t == 1].mean() - y[t == 0].mean()), 1),
           "post_treatment_control": (float(ols(y, c1(t, Xa, m))[0][1]), 2),
           "confounder_omitted": (float(ols(y, c1(t, np.delete(Xa, j, 1)))[0][1]), 2)}
    desc = (f"Observational data on {n} {c['unit']}. Variables in the Dataset:\n"
            + _vars([(c["T"], c["Tdesc"])] + c["X"] + [c["M"], (c["Y"], c["Ydesc"])])
            + "\nAll covariates except the ones described as measured after treatment were recorded before treatment.")
    query = f"What is the effect of {c['T'].replace('_', ' ')} on {c['Y'].replace('_', ' ')}?"
    roles = {"treatment": c["T"], "outcome": c["Y"], "confounders": [nm for nm, _ in c["X"]], "post_treatment": c["M"][0]}
    return df, desc, query, gold, gse, abl, roles


def gen_iv(rng, k: float):
    c = IV[rng.integers(len(IV))]
    n = int(rng.integers(2000, 5000))
    W = rng.normal(size=(n, len(c["W"])))
    U = rng.normal(size=n)
    zlin = 0.6 * k * W[:, 0] + rng.normal(size=n)
    Z = (zlin > 0).astype(float) if c["binary_z"] else zlin
    pi = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
    T = pi * Z + 0.9 * k * U + W @ rng.normal(0, 0.4, W.shape[1]) + rng.normal(size=n)
    tau = c["scale"] * rng.uniform(0.6, 1.4)
    lam = abs(c["scale"]) * rng.uniform(1.0, 2.0) * k * rng.choice([-1, 1])
    beta = abs(c["scale"]) * rng.normal(0, 1.2, W.shape[1]) * k
    beta[0] = abs(beta[0]) + abs(c["scale"]) * k            # the covariate the instrument tracks moves Y
    Y = tau * T + lam * U + W @ beta + rng.normal(0, abs(c["scale"]) * 1.5, n)
    wcols = {nm: np.round(10 + 2 * W[:, j], 3) for j, (nm, _) in enumerate(c["W"])}
    df = pd.DataFrame({c["Z"]: (Z.astype(int) if c["binary_z"] else np.round(Z, 4)), c["T"]: np.round(T + 5, 4),
                       **wcols, c["Y"]: np.round(Y + 20, 4)})
    y, t, z = df[c["Y"]].to_numpy(float), df[c["T"]].to_numpy(float), df[c["Z"]].to_numpy(float)
    Wd = c1(*[df[nm].to_numpy(float) for nm, _ in c["W"]])
    gold, gse = tsls(y, t, z, Wd)
    rf = ols(y, np.column_stack([z, Wd]))[0][0]
    abl = {"reference": (gold, None),
           "ols_ignores_instrument": (float(ols(y, np.column_stack([t, Wd]))[0][0]), 1),
           "reduced_form": (float(rf), 2),
           "iv_without_covariates": (tsls(y, t, z, c1(np.zeros(n))[:, :1])[0], 2)}
    desc = (f"Data on {n} {c['unit']}. Unobserved factors may affect both {c['T']} and {c['Y']}. Variables in the Dataset:\n"
            + _vars([(c["T"], c["Tdesc"]), (c["Z"], c["Zdesc"])] + c["W"] + [(c["Y"], c["Ydesc"])]))
    query = f"What is the causal effect of {c['T'].replace('_', ' ')} on {c['Y'].replace('_', ' ')}?"
    roles = {"treatment": c["T"], "outcome": c["Y"], "instrument": c["Z"], "covariates": [nm for nm, _ in c["W"]]}
    return df, desc, query, gold, gse, abl, roles


def gen_did(rng, k: float):
    c = DID[rng.integers(len(DID))]
    nu = int(rng.integers(60, 160)); P = int(rng.integers(4, 9)); t0 = int(rng.integers(2, P - 1))
    treat = (rng.random(nu) < 0.45).astype(float)
    alpha = rng.normal(0, 3, nu) + treat * abs(c["scale"]) * rng.uniform(1.5, 3.0) * k * rng.choice([-1, 1])
    gamma = np.cumsum(rng.normal(abs(c["scale"]) * 0.6 * k * rng.choice([-1, 1]), 0.3, P))
    tau = c["scale"] * rng.uniform(0.6, 1.4)
    rows = []
    for i in range(nu):
        for t in range(P):
            post = float(t >= t0)
            yv = 50 + alpha[i] + gamma[t] + tau * treat[i] * post + rng.normal(0, abs(c["scale"]) * 0.8)
            rows.append((f"{c['unit'][:3].upper()}{i:03d}", t + 1, int(treat[i]), int(post), round(yv, 3)))
    df = pd.DataFrame(rows, columns=["unit_id", c["period"], c["treat"], "post", c["Y"]])
    y = df[c["Y"]].to_numpy(float); g = df[c["treat"]].to_numpy(float); po = df["post"].to_numpy(float)
    U = pd.get_dummies(df["unit_id"], drop_first=True).to_numpy(float)
    Tt = pd.get_dummies(df[c["period"]], drop_first=True).to_numpy(float)
    b, se = ols(y, c1(g * po, U, Tt))
    gold, gse = float(b[1]), float(se[1])
    post_gap = y[(g == 1) & (po == 1)].mean() - y[(g == 0) & (po == 1)].mean()
    prepost = y[(g == 1) & (po == 1)].mean() - y[(g == 1) & (po == 0)].mean()
    abl = {"reference": (gold, None), "post_period_gap": (float(post_gap), 1), "treated_before_after": (float(prepost), 1)}
    desc = (f"Panel data on {nu} {c['unit']} observed over {P} {c['period']}s. The {c['T']} took effect from {c['period']} {t0 + 1} "
            f"onwards in the treated {c['unit']} only. Variables in the Dataset:\n"
            + _vars([("unit_id", "identifier of the unit"), (c["period"], f"{c['period']} index (1-{P})"), (c["treat"], c["tdesc"]),
                     ("post", f"1 from {c['period']} {t0 + 1} onwards"), (c["Y"], c["Ydesc"])]))
    query = f"What was the effect of the {c['T']} on {c['Y'].replace('_', ' ')}?"
    roles = {"outcome": c["Y"], "unit": "unit_id", "period": c["period"], "group": c["treat"], "post": "post"}
    return df, desc, query, gold, gse, abl, roles


def gen_rdd(rng, k: float):
    c = RDD[rng.integers(len(RDD))]
    n = int(rng.integers(2000, 5000))
    r = c["cut"] + rng.uniform(-c["span"], c["span"], n)
    x = (r - c["cut"]) / c["span"]
    tau = c["scale"] * rng.uniform(0.7, 1.3)
    curv = abs(c["scale"]) * rng.uniform(1.5, 3.0) * k
    f = curv * (1.2 * x + 1.5 * np.sign(rng.normal()) * x ** 2 + 0.8 * x ** 3)
    y = 10 + f + tau * (r >= c["cut"]) + rng.normal(0, abs(c["scale"]) * 0.9, n)
    df = pd.DataFrame({c["R"]: np.round(r, 3), c["Y"]: np.round(y, 4)})
    rr, yy = df[c["R"]].to_numpy(float), df[c["Y"]].to_numpy(float)
    h = 0.4 * (rr.max() - rr.min()) / 2                    # one rule, computable from the data
    gold, gse = local_linear(yy, rr, c["cut"], h)
    d = (rr >= c["cut"]).astype(float)
    abl = {"reference": (gold, None),
           "global_linear": (float(ols(yy, c1(d, rr - c["cut"]))[0][1]), 1),
           "difference_in_means": (float(yy[d == 1].mean() - yy[d == 0].mean()), 1),
           # accepted variants of the reference: not traps, recorded for the audit
           "local_linear_half_bandwidth": (local_linear(yy, rr, c["cut"], h / 2)[0], None),
           "local_linear_wide_bandwidth": (local_linear(yy, rr, c["cut"], 1.5 * h)[0], None)}
    desc = (f"Data on {n} {c['unit']}. Variables in the Dataset:\n"
            + _vars([(c["R"], c["Rdesc"]), (c["Y"], c["Ydesc"])]) + f"\nThe cutoff is {c['cut']:g}.")
    query = f"What is the effect of crossing the cutoff on {c['Y'].replace('_', ' ')}?"
    roles = {"outcome": c["Y"], "running": c["R"], "cutoff": c["cut"], "bandwidth_rule": "0.4 x half the running-variable range"}
    return df, desc, query, gold, gse, abl, roles


def gen_att(rng, k: float):
    c = ATT[rng.integers(len(ATT))]
    n = int(rng.integers(2000, 5000))
    X = rng.normal(size=(n, 3))
    T = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * k * X[:, 0] + 0.5 * X[:, 1] - 0.3)))).astype(float)
    tau0, tau1 = c["scale"] * rng.uniform(0.5, 1.0), c["scale"] * rng.uniform(0.6, 1.2) * k
    Y = 20 + X @ (abs(c["scale"]) * rng.normal(0, 0.8, 3) + np.array([abs(c["scale"]) * k, 0, 0])) \
        + T * (tau0 + tau1 * X[:, 0]) + rng.normal(0, abs(c["scale"]), n)
    cols = {nm: np.round(10 + 3 * X[:, j], 3) for j, (nm, _) in enumerate(c["X"])}
    df = pd.DataFrame({c["T"]: T.astype(int), **cols, c["Y"]: np.round(Y, 4)})
    y, t = df[c["Y"]].to_numpy(float), df[c["T"]].to_numpy(float)
    Xa = np.column_stack([df[nm].to_numpy(float) for nm, _ in c["X"]])
    b0 = ols(y[t == 0], c1(Xa[t == 0]))[0]
    b1 = ols(y[t == 1], c1(Xa[t == 1]))[0]
    att = float(np.mean(y[t == 1] - c1(Xa[t == 1]) @ b0))
    ate = float(np.mean(c1(Xa) @ b1 - c1(Xa) @ b0))
    boots = []
    for _ in range(60):                                      # SE for the tolerance rule
        ix = rng.integers(0, n, n); yb, tb, Xb = y[ix], t[ix], Xa[ix]
        bb = ols(yb[tb == 0], c1(Xb[tb == 0]))[0]
        boots.append(np.mean(yb[tb == 1] - c1(Xb[tb == 1]) @ bb))
    abl = {"reference": (att, None),
           "no_adjustment": (float(y[t == 1].mean() - y[t == 0].mean()), 1),
           "additive_ols": (float(ols(y, c1(t, Xa))[0][1]), 2),
           "ate_not_att": (ate, 3)}
    desc = (f"Observational data on {n} {c['unit']}; the covariates were recorded before enrolment. Variables in the Dataset:\n"
            + _vars([(c["T"], c["Tdesc"])] + c["X"] + [(c["Y"], c["Ydesc"])]))
    roles = {"treatment": c["T"], "outcome": c["Y"], "covariates": [nm for nm, _ in c["X"]]}
    return df, desc, c["query"], att, float(np.std(boots)), abl, roles


FAMILIES = {"confounded": gen_confounded, "iv": gen_iv, "did": gen_did, "rdd": gen_rdd, "att": gen_att}


def make(family: str, seed: int, max_tries: int = 12):
    """Draw until every trap is live (outside tolerance) AND every accepted variant of the
    reference (step None, e.g. a half or 1.5x RD bandwidth) is inside it: an instance whose
    answer hinges on an unstated choice would grade judgment on a coin flip, which is what
    makes half of CauSciBench's real RD queries unusable. A dead trap strengthens the bias
    knob; a failed variant just re-draws."""
    rng = np.random.default_rng(seed)
    k = 1.0
    for tries in range(1, max_tries + 1):
        df, desc, query, gold, gse, abl, roles = FAMILIES[family](rng, k)
        tol = max(0.10 * abs(gold), 0.5 * gse)
        weak = [nm for nm, (v, step) in abl.items() if step is not None and abs(v - gold) <= tol]
        loose = [nm for nm, (v, step) in abl.items() if step is None and nm != "reference" and abs(v - gold) > tol]
        if not weak and not loose:
            break
        if weak:
            k *= 1.4
    return df, desc, query, gold, gse, tol, abl, weak + loose, tries, roles


def emit(family: str, split: str, n: int, seed0: int, out_root: Path) -> dict:
    out = out_root / f"causal_{family}" / split
    out.mkdir(parents=True, exist_ok=True)
    weak_ids, tries_total = [], 0
    for i in range(n):
        seed = seed0 + i
        df, desc, query, gold, gse, tol, abl, weak, tries, roles = make(family, seed)
        tries_total += tries
        pid = f"causal_{family}_{i:05d}"
        d = out / pid
        (d / "data_files").mkdir(parents=True, exist_ok=True)
        csv = f"{family}_data.csv"
        df.to_csv(d / "data_files" / csv, index=False)
        cfg = {"id": pid, "task": task_text(desc, query, csv), "data_files": [f"data_files/{csv}"],
               "ground_truth": {"effect": gold},
               "grader": {"type": "multi_numeric_tolerance",
                          "config": {"keys": {"effect": {"absolute_tolerance": tol}}}},
               "_provenance": {"family": f"causal_{family}", "seed": seed, "gold_se": gse, "gen_tries": tries, "roles": roles,
                               "tol_rule": "max(0.10*|gold|, 0.5*SE)",
                               "note": "ground truth is the reference estimator on the sampled data, not the latent effect",
                               "ablations": {nm: {"effect": v, "fails_at_step": step, "deviation": abs(v - gold)}
                                             for nm, (v, step) in abl.items()}}}
        (d / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        if weak:
            weak_ids.append(pid)
    man = {"family": f"causal_{family}", "split": split, "n": n, "seed0": seed0,
           "n_all_traps_valid": n - len(weak_ids), "n_weak": len(weak_ids), "weak_ids": weak_ids,
           "mean_tries": tries_total / n}
    (out / "_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def mix(split: str, out_root: Path) -> int:
    """generated/causal_mix/SPLIT: symlinks to every family's instances, one RL training pool."""
    out = out_root / "causal_mix" / split
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(FAMILIES):
        for d in sorted((out_root / f"causal_{f}" / split).glob(f"causal_{f}_*")):
            link = out / d.name
            if not link.exists():
                link.symlink_to(d.resolve(), target_is_directory=True)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES) + ["all"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--out", default=None, help="default: $DEEP_DATA_ROOT/generated")
    ap.add_argument("--mix", action="store_true", help="only link the emitted families into causal_mix/SPLIT")
    args = ap.parse_args()
    from paths import data_root
    out = Path(args.out) if args.out else data_root() / "generated"
    if args.mix:
        print(json.dumps({"causal_mix": args.split, "n": mix(args.split, out)}))
        return 0
    fams = sorted(FAMILIES) if args.family == "all" else [args.family]
    for j, f in enumerate(fams):
        seed0 = args.seed0 if args.seed0 is not None else (7_000_000 + 100_000 * j + (0 if args.split == "train" else 50_000))
        man = emit(f, args.split, args.n, seed0, out)
        print(json.dumps({k: v for k, v in man.items() if k != "weak_ids"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
