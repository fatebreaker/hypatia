"""Pharmacokinetics: apparent clearance and terminal half-life after an oral dose (a SAGE plug-in).

The data are a single-dose oral PK study as a clinical pharmacologist receives it: plasma
concentrations for 6-12 subjects at scheduled times (recorded actual times), from a
two-compartment model with first-order absorption and an absorption lag, log-normal
between-subject variability, a combined proportional + additive assay error, and every value
below the assay's lower limit of quantification (LLOQ) reported as "BLQ". The irregularities a
careful analysis has to handle are the ones real bioanalytical data carry: a biphasic decline
(distribution, then elimination), heteroscedastic error over two to three decades of
concentration, and censored late samples.

Reference analysis (the gold, on the sampled data), per subject, then summarised as geometric
means across subjects (the standard summary for log-normally distributed PK parameters):

  model     C(t) = A e^{-alpha t'} + B e^{-beta t'} - (A + B) e^{-ka t'}, t' = t - tlag > 0
            (the macro-constant form of the two-compartment oral model; ka > alpha > beta), and
            its one-compartment counterpart A (e^{-k t'} - e^{-ka t'}); the structural model is
            CHOSEN from the data, by AIC summed over subjects (a generated draw always has a
            biphasic decline, and `sanity` rejects a draw on which the choice misses it)
  fit       weighted by the stated assay error model, var = sd_add^2 + (cv f)^2, with BLQ values
            as left-censored observations (Beal's M3: their likelihood is P(value < LLOQ)),
            solved as iteratively reweighted least squares (scipy.optimize.least_squares) from
            curve-stripping starts
  answers   CL/F = dose / AUC(0-inf) of the fitted curve;  t1/2 = ln 2 / (slowest rate)

Traps (the step each skips) and accepted variants:

  1 qc            blq_as_zero            BLQ entered as 0 mg/L and fitted like a measurement:
                                         the elimination phase is dragged down, t1/2 shortens
  2 model-choice  one_compartment        a monoexponential decline: t1/2 is a compromise between
                  (core where the data     the distribution and elimination phases
                  are biphasic)
  2 model-choice  two_compartment_forced two compartments without asking the data: coincides with
                  (conditional)          the reference on every generated draw, live on the real
                                         theophylline study (a one-compartment drug)
  3 estimate      unweighted_linear      ordinary least squares on the concentration scale; the
                  (conditional)          peak dominates and the terminal slope is only loosely
                                         constrained -- unbiased on average but erratic, so it is
                                         live only on some draws (step None where it coincides)
  4 estimate      half_life_from_early_decline   t1/2 from a log-linear fit to the whole
                                         post-peak decline, which starts in the distribution phase
  -  variant      blq_discarded          Beal's M1 (BLQ values dropped): accepted, must agree

Real data: `TheophSource` (sage real --source theoph) asks the same question of R's classic
theophylline study, with the published population estimates as gold (see its docstring).

The analysis functions are self-contained (numpy/scipy/pandas) so a demonstration embeds their
source verbatim: the trace and the ground truth are the same code.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sage.core import Cell, Family, RealSource, RealTarget, Sample, Trap

CONC = "concentrations.csv"
NAN = float("nan")
SCHEDULE = (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 24, 36, 48, 72, 96, 120, 144)


# ----------------------------------------------------------------------------- analysis
def exponentials(p, n_cmt=2):
    """The oral-dose curve as sum_j c_j exp(-rate_j (t - tlag)): (c, rates, tlag). Parameters are
    2 cmt: log A, log B, log beta, log(alpha - beta), log(ka - alpha), tlag; 1 cmt: log A, log k, log(ka - k), tlag."""
    if n_cmt == 1:
        A, k = np.exp(p[0]), np.exp(p[1])
        return np.array([A, -A]), np.array([k, k + np.exp(p[2])]), p[3]
    A, B, beta = np.exp(p[:3])
    alpha = beta + np.exp(p[3])
    return np.array([A, B, -(A + B)]), np.array([alpha, beta, alpha + np.exp(p[4])]), p[5]


def pk_curve(p, t, n_cmt=2):
    c, rates, lag = exponentials(p, n_cmt)
    return np.exp(-np.outer(np.clip(np.asarray(t, float) - lag, 0, None), rates)) @ c


def auc_half_life(p, n_cmt=2):
    """AUC(0-inf) of the fitted curve, and the terminal half-life ln 2 / (slowest disposition rate)."""
    c, rates, _ = exponentials(p, n_cmt)
    return float(np.sum(c / rates)), float(np.log(2) / rates[n_cmt - 1])


def stripping_starts(t, c, n_cmt=2):
    """Curve stripping: terminal slope from the last 3 measured points, alpha from the residual."""
    i = int(np.argmax(c))
    b1, b0 = np.polyfit(t[-3:], np.log(c[-3:]), 1)
    beta, B = max(-b1, 1e-3), np.exp(b0)
    if n_cmt == 1:
        return [[np.log(c[i] * 2), np.log(beta), np.log(beta * m), t[0] / 2] for m in (3, 10)]
    rest = c[i:-3] - B * np.exp(-beta * t[i:-3])
    ok = rest > 0
    alpha, A = (max(-np.polyfit(t[i:-3][ok], np.log(rest[ok]), 1)[0], 2 * beta), rest[ok].max()) if ok.sum() >= 2 \
        else (5 * beta, c[i])
    return [[np.log(A), np.log(B), np.log(beta), np.log(max(alpha * m - beta, 0.1 * beta)), np.log(alpha * m * r),
             t[0] / 2] for m in (0.5, 1, 2) for r in (1, 4)]


def fit_subject(t, conc, blq, lloq, cv, sd_add, n_cmt=2, blq_mode="censored", weighted=True):
    """One subject: a proportional-error start, then IRLS with the assay error model (OLS if not
    weighted); BLQ "censored" (M3 likelihood), "dropped" (M1) or "zero". Returns (parameters, -2 log L)."""
    from scipy.optimize import least_squares
    from scipy.stats import norm
    if blq_mode == "zero":
        conc, blq = np.where(blq, 0.0, conc), np.zeros_like(blq)
    elif blq_mode == "dropped":
        t, conc, blq = t[~blq], conc[~blq], blq[~blq]
    q, pos = ~blq, ~blq & (t > 0) & (conc > 0)
    tp, cp = t[pos], conc[pos]
    lo = [np.log(cp.max()) - 12] * n_cmt + [np.log(np.log(2) / 2000)] + [-12] * n_cmt + [0]
    hi = [np.log(cp.max()) + 8] * n_cmt + [np.log(np.log(2) / 0.2)] + [4] * (n_cmt - 1) + [5, tp[0]]
    starts = [np.clip(x, np.add(lo, 1e-9), np.subtract(hi, 1e-9)) for x in stripping_starts(tp, cp, n_cmt)]
    logfit = lambda p: np.log(np.maximum(pk_curve(p, tp, n_cmt), 1e-30)) - np.log(cp)
    p = min((least_squares(logfit, x0, bounds=(lo, hi)) for x0 in starts), key=lambda r: r.cost).x
    for _ in range(4 if weighted else 1):                          # IRLS: variances from the current curve
        sd = np.sqrt(sd_add ** 2 + (cv * pk_curve(p, t, n_cmt)) ** 2) if weighted else np.ones_like(t)
        cens = lambda f: np.sqrt(-2 * norm.logcdf((lloq - f[blq]) / sd[blq])) if blq_mode == "censored" else []
        p = least_squares(lambda p: np.r_[(conc[q] - pk_curve(p, t, n_cmt)[q]) / sd[q], cens(pk_curve(p, t, n_cmt))],
                          p, bounds=(lo, hi)).x
    f = pk_curve(p, t, n_cmt)
    if not weighted:                                               # OLS: normal likelihood, variance = RSS / n
        return p, q.sum() * (np.log(2 * np.pi * np.mean((conc[q] - f[q]) ** 2)) + 1)
    v = sd_add ** 2 + (cv * f) ** 2
    return p, float(np.sum((conc[q] - f[q]) ** 2 / v[q] + np.log(2 * np.pi * v[q]))
                    - 2 * np.sum(norm.logcdf((lloq - f[blq]) / np.sqrt(v[blq]))) * (blq_mode == "censored"))


def fit_study(table, lloq, cv, sd_add, models=(1, 2), **kw):
    """Fit every subject with each structural model; the one with the lower AIC summed over subjects
    wins. Returns (number of compartments, {subject: parameters}, {compartments: AIC})."""
    fits = {m: {sid: fit_subject(t, c, b, lloq, cv, sd_add, m, **kw) for sid, (_, t, c, b) in table.items()}
            for m in models}
    aic = {m: sum(ll for _, ll in f.values()) + 2 * len(table) * (2 * m + 2) for m, f in fits.items()}
    best = min(aic, key=aic.get)
    return best, {sid: p for sid, (p, _) in fits[best].items()}, aic


def subject_table(df):
    """{subject: (dose, times, concentrations (BLQ -> nan), BLQ flags)}; dose in the file's unit."""
    dose = [c for c in df.columns if c.startswith("dose")][0]
    return {sid: (float(s[dose].iloc[0]), s["time_h"].to_numpy(float), pd.to_numeric(s["conc_mg_L"], errors="coerce")
                  .to_numpy(float), s["conc_mg_L"].astype(str).eq("BLQ").to_numpy()) for sid, s in df.groupby("subject")}


def summarise(table, n_cmt, fits):
    """Geometric means over subjects of CL/F = dose / AUC(0-inf) and of the terminal half-life."""
    cl, th = zip(*[(table[s][0] / a, h) for s, (a, h) in ((s, auc_half_life(p, n_cmt)) for s, p in fits.items())])
    return float(np.exp(np.mean(np.log(cl)))), float(np.exp(np.mean(np.log(th))))


def early_decline_half_life(t, conc, blq):
    """TRAP (step 4): ln 2 / slope of a log-linear fit to every measured point after the peak."""
    q = ~blq & (t > 0)
    tq, cq = t[q], conc[q]
    i = int(np.argmax(cq))
    return float(-np.log(2) / np.polyfit(tq[i + 1:], np.log(cq[i + 1:]), 1)[0])


# ----------------------------------------------------------------------------- simulator helpers
def _micro(alpha, beta, f_b, cl):
    """Micro-constants (CL, V1, Q, V2) with disposition rates alpha > beta and terminal
    intercept fraction f_b = B/(A+B) of an IV bolus."""
    k21 = (1 - f_b) * beta + f_b * alpha
    k10 = alpha * beta / k21
    k12 = alpha + beta - k21 - k10
    v1 = cl / k10
    return {"cl": cl, "v1": v1, "q": k12 * v1, "v2": k12 * v1 / k21}


def _macro(cl, v1, q, v2):
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    d = np.sqrt(s * s - 4 * k10 * k21)
    return (s + d) / 2, (s - d) / 2, k21


def _conc(t, dose, cl, v1, q, v2, ka, tlag):
    a, b, k21 = _macro(cl, v1, q, v2)
    tt = np.clip(t - tlag, 0, None)
    A = dose * ka / v1 * (k21 - a) / ((ka - a) * (b - a))
    B = dose * ka / v1 * (k21 - b) / ((ka - b) * (a - b))
    return A * np.exp(-a * tt) + B * np.exp(-b * tt) - (A + B) * np.exp(-ka * tt)


PAR = ("cl", "v1", "q", "v2", "ka", "tlag")


def geo_mean(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))
OMEGA = {"cl": 0.3, "v1": 0.2, "q": 0.2, "v2": 0.2, "ka": 0.3, "tlag": 0.2}


# ----------------------------------------------------------------------------- the family
class OralPK(Family):
    name = "pk_oral"
    domain = "pharmacokinetics"
    keys = ("cl_f", "t_half")
    steps = {1: "treat BLQ samples as censored below the LLOQ, not as zero",
             2: "choose the structural model from the data (AIC over subjects): here a biphasic decline",
             3: "weight the fit by the assay error model (proportional + additive)",
             4: "take the half-life from the terminal (elimination) phase, not the early decline"}
    skills = {1: "qc", 2: "model-choice", 3: "estimate", 4: "estimate"}
    tol_rule = "cl_f 10% of gold, t_half 12% of gold"
    note = ("ground truth is the reference analysis (per-subject one- and two-compartment oral fits, the structural "
            "model chosen by AIC summed over subjects, assay-error weighted, BLQ censored; geometric means across "
            "subjects) on the sampled concentrations, not the latent parameters; "
            "deviation is in units of tolerance, maxed over the answer keys")
    seed_blocks = {"train": 14_000_000, "test": 14_050_000}
    max_tries = 12
    rng_policy = "stream"
    demo_timeout = 240

    def knobs(self, rng):
        return {"blq": 1.0, "dist": 1.0}

    def draw(self, rng, k) -> Sample:
        n = int(rng.choice([6, 8, 10, 12]))
        dose = float(rng.choice([50, 100, 150, 200, 250, 300, 400, 500]))
        th = float(np.exp(rng.uniform(np.log(4), np.log(30))))          # typical terminal half-life, h
        beta = np.log(2) / th
        cl = float(np.exp(rng.uniform(np.log(2), np.log(30))))
        for _ in range(100):                     # a biphasic decline: alpha >> beta, a real distribution phase
            alpha = beta * rng.uniform(6, 15)
            f_b = float(np.clip(rng.uniform(0.15, 0.4) / k["dist"], 0.05, 0.9))
            typ = _micro(alpha, beta, f_b, cl)
            if min(typ.values()) > 0:
                break
        typ.update(ka=alpha * rng.uniform(1.5, 4), tlag=rng.uniform(0.1, 0.6))
        t_end = next((s for s in SCHEDULE if s >= 4.5 * th), SCHEDULE[-1])
        times = np.array([s for s in SCHEDULE if s <= t_end], float)
        cv = float(np.round(rng.uniform(0.08, 0.15), 3))
        c_end = _conc(np.array([t_end]), dose, *[typ[p] for p in PAR])[0]
        lloq = float(f"{c_end * 2 ** rng.uniform(1.0, 2.5) * k['blq']:.2g}")
        sd_add = float(f"{lloq * rng.uniform(0.05, 0.15):.2g}")
        rows, ind = [], []
        for i in range(n):
            for _ in range(200):                 # absorption faster than distribution (no flip-flop)
                p = {q: typ[q] * np.exp(OMEGA[q] * rng.normal()) for q in PAR}
                a, b, _ = _macro(p["cl"], p["v1"], p["q"], p["v2"])
                if p["ka"] > 1.2 * a and a > 3 * b:
                    break
            ind.append({"cl": p["cl"], "t_half": np.log(2) / b, "alpha": a, "ka": p["ka"], "tlag": p["tlag"]})
            tt = np.round(times * np.exp(rng.normal(0, 0.03, len(times))), 2)
            f = _conc(tt, dose, *[p[q] for q in PAR])
            y = f * (1 + cv * rng.normal(size=len(tt))) + sd_add * rng.normal(size=len(tt))
            for tj, yj in zip(tt, y):
                rows.append((i + 1, dose, float(tj), "BLQ" if yj < lloq else f"{yj:.4g}"))
        df = pd.DataFrame(rows, columns=["subject", "dose_mg", "time_h", "conc_mg_L"])
        ind = pd.DataFrame(ind)
        latent = {"cl_f": geo_mean(ind.cl), "t_half": geo_mean(ind.t_half), "n_cmt": 2,
                  "typical": {q: float(v) for q, v in typ.items()}, "f_b": float(f_b),
                  "subjects": ind.round(5).to_dict(orient="list")}
        meta = {"n": n, "dose": dose, "lloq": lloq, "cv": cv, "sd_add": sd_add, "t_end": float(t_end),
                "drug": ("an investigational small molecule", "a new anti-infective", "a kinase inhibitor",
                         "an oral anticoagulant candidate")[int(rng.integers(4))]}
        return Sample({"df": df, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / CONC, index=False)
        return [CONC]

    # ------------------------------------------------------------------ analysis
    def _study(self, data, models=(1, 2), **kw):
        """fit_study on the draw, cached: the reference's two structural models are also the
        one_compartment trap and the two_compartment_forced trap, so they are fitted once."""
        m = data["meta"]
        key = ("_fit", tuple(models), tuple(sorted(kw.items())))
        cache = data.setdefault("_cache", {})
        if key not in cache:
            table = subject_table(data["df"])
            n_cmt, fits, aic = fit_study(table, m["lloq"], m["cv"], m["sd_add"], models=models, **kw)
            cache[key] = (dict(zip(self.keys, summarise(table, n_cmt, fits))), n_cmt, aic)
        return cache[key]

    def reference(self, data) -> dict:
        ans, n_cmt, aic = self._study(data)
        return {**ans, "_n_cmt": n_cmt, "_aic": {str(k): float(v) for k, v in aic.items()}}

    def tolerance(self, gold) -> dict:
        return {"cl_f": 0.10 * abs(gold["cl_f"]), "t_half": 0.12 * abs(gold["t_half"])}

    def sanity(self, gold, tol, latent) -> list[str]:
        return [] if "n_cmt" not in latent or gold["_n_cmt"] == latent["n_cmt"] else ["structural_model_misidentified"]

    def traps(self, data) -> dict:
        def safe(fn):
            try:
                a = fn()
                return a if all(np.isfinite(v) and v > 0 for v in a.values()) else {k: NAN for k in self.keys}
            except Exception:                                            # noqa: BLE001
                return {k: NAN for k in self.keys}
        ref, n_cmt, _ = self._study(data)
        chosen = (n_cmt,)

        def early():
            th = [early_decline_half_life(t, c, b) for _, t, c, b in subject_table(data["df"]).values()]
            return {"cl_f": ref["cl_f"], "t_half": float(np.exp(np.mean(np.log(th))))}
        return {"blq_as_zero": Trap(safe(lambda: self._study(data, chosen, blq_mode="zero")[0]), 1),
                "one_compartment": Trap(safe(lambda: self._study(data, (1,))[0]), 2, n_cmt == 2),
                "two_compartment_forced": Trap(safe(lambda: self._study(data, (2,))[0]), 2, n_cmt == 1),
                "unweighted_linear": Trap(safe(lambda: self._study(data, chosen, blq_mode="dropped",
                                                                   weighted=False)[0]), 3, False),
                "half_life_from_early_decline": Trap(safe(early), 4),
                "blq_discarded": Trap(safe(lambda: self._study(data, chosen, blq_mode="dropped")[0]), None)}

    def strengthen(self, k, dead):
        k = dict(k)
        if "blq_as_zero" in dead:
            k["blq"] *= 1.5
        if "one_compartment" in dead or "half_life_from_early_decline" in dead:
            k["dist"] *= 1.4
        return k

    def shortcuts(self, data) -> dict:
        """By eye from the table: CL/F from the dose over a trapezoid AUC of the measured
        points, half-life from the last two measured points of each subject."""
        cl, th = [], []
        for dose, t, c, blq in subject_table(data["df"]).values():
            q = ~blq
            cl.append(dose / np.trapezoid(np.nan_to_num(c[q]), t[q]))
            tq, cq = t[q], c[q]
            th.append(np.log(2) * (tq[-1] - tq[-2]) / max(np.log(cq[-2] / cq[-1]), 1e-3))
        return {"trapezoid_and_last_two": {"cl_f": geo_mean(cl), "t_half": geo_mean(th)}}

    def extra_provenance(self, inst) -> dict:
        m = inst.sample.data["meta"]
        return {"knobs": inst.knobs, "design": {k: m[k] for k in ("n", "dose", "lloq", "cv", "sd_add", "t_end")},
                "skills": {str(s): v for s, v in self.skills.items()}}

    # ------------------------------------------------------------------ prompt
    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        unit = m.get("dose_unit", "mg")
        cl_unit = "L/h/kg" if unit == "mg/kg" else "L/h"
        dose_col = "dose_mg_per_kg" if unit == "mg/kg" else "dose_mg"
        dose = m.get("dose_text") or f"{m['dose']:g} {unit}"
        study = m.get("study") or f"In a first-in-human study of {m['drug']}"
        who = m.get("population", "healthy volunteers")
        head = (f"You are a clinical pharmacologist. {study}, {m['n']} {who} each received a single oral dose of "
                f"{dose} at time 0, and venous blood was sampled up to {m['t_end']:g} h after the dose.\n\n"
                "Data files:\n"
                f"- data_files/{CONC}: one row per plasma sample. Columns: subject, {dose_col}, time_h (actual sampling "
                "time after the dose, h), conc_mg_L (plasma concentration, mg/L). The pre-dose sample is at time 0.\n\n")
        if m.get("no_blq"):
            assay = ("Bioanalytical assay: no sample was below the limit of quantification; pre-dose samples without "
                     "detectable drug are recorded as 0. ")
        else:
            assay = (f"Bioanalytical assay: the lower limit of quantification (LLOQ) is {m['lloq']:g} mg/L; "
                     "concentrations below it are reported as \"BLQ\". ")
        if m["cv"] > 0:
            assay += ("From the assay validation, a measured concentration has a standard deviation of "
                      f"sqrt({m['sd_add']:g}^2 + ({m['cv']:g} x concentration)^2) mg/L (an additive component of "
                      f"{m['sd_add']:g} mg/L plus {100 * m['cv']:.1f}% proportional error).\n\n")
        else:
            assay += (m.get("error_text") or f"Measurement error is additive, with a standard deviation of "
                      f"{m['sd_add']:g} mg/L.") + "\n\n"
        q = (f"Question: what are the drug's apparent oral clearance CL/F ({cl_unit}) and its terminal elimination "
             "half-life (h)? Report the geometric mean across subjects of each subject's CL/F and of each subject's "
             "terminal half-life.\n\n")
        method = ("" if blind else
                  "Approach: fit every subject with one- and two-compartment models with first-order absorption and an "
                  "absorption lag time, and keep the structural model with the lower AIC summed over subjects (a "
                  "biphasic log-concentration decline needs two compartments). Weight the fit with the assay's error "
                  "model above, and treat BLQ samples as censored observations (their likelihood is the probability "
                  "of a value below the LLOQ), not as zero. CL/F = dose / AUC(0-inf) of the fitted curve; the "
                  "terminal half-life is ln 2 divided by the slowest (terminal) disposition rate constant.\n\n")
        return head + assay + q + method + \
            f'Return your answer as JSON: {{"answer": {{"cl_f": <{cl_unit}>, "t_half": <h>}}}}'

    def demo(self, cfg) -> list[Cell]:
        m = re.search(r"LLOQ\) is ([\d.eE+-]+) mg/L", cfg["task"])
        e = re.search(r"sqrt\(([\d.eE+-]+)\^2 \+ \(([\d.eE+-]+) x concentration", cfg["task"])
        lloq, sd_add, cv = m.group(1), e.group(1), e.group(2)
        inspect = (f'import numpy as np, pandas as pd\ndf = pd.read_csv("data_files/{CONC}")\n'
                   'print(df.head(4).to_string(index=False))\n'
                   'df["blq"], df["c"] = df.conc_mg_L.eq("BLQ"), pd.to_numeric(df.conc_mg_L, errors="coerce")\n'
                   'df["slot"] = df.groupby("subject").cumcount()                 # k-th scheduled sample\n'
                   'post = df[df.time_h > 0]\n'
                   'print("subjects:", df.subject.nunique(), "| samples each:", df.groupby("subject").size().iloc[0],\n'
                   '      "| BLQ after the dose:", int(post.blq.sum()), "of", len(post))\n'
                   'print("last measured time per subject (h):", post[~post.blq].groupby("subject").time_h.max().tolist())\n'
                   'prof = post[~post.blq].groupby("slot").agg(t=("time_h", "median"), gm=("c", lambda x: np.exp(np.log(x).mean())))\n'
                   'prof["log_slope"] = np.r_[np.nan, np.diff(np.log(prof.gm)) / np.diff(prof.t)]\n'
                   'print("geometric-mean profile (median time, h):")\n'
                   'print(prof.round(4).to_string())')

        def prose2(outs):
            o = outs[0]
            nb = re.search(r"BLQ after the dose: (\d+) of (\d+)", o)
            blq = f"{nb.group(1)} of the {nb.group(2)} post-dose samples are BLQ" if nb else "several late samples are BLQ"
            rows = [(float(a), float(b)) for a, b in re.findall(r"^\s*\d+\s+([\d.]+)\s+[\d.]+\s+(-[\d.]+)\s*$", o, re.M)]
            shape = "The mean log-concentration falls steeply after the peak and more slowly later"
            if len(rows) >= 4:
                (t1, s1), (t2, s2) = min(rows, key=lambda r: r[1]), rows[-1]
                shape = (f"The mean log-concentration falls at {-s1:.3f}/h around {t1:.3g} h but only {-s2:.3f}/h by "
                         f"{t2:.3g} h")
            return (f"{shape}, which looks biphasic: a distribution phase and then elimination. A one-compartment model "
                    "(or a log-linear fit to the whole post-peak decline) would then give a half-life that mixes the "
                    "two phases, so I let the data choose: every subject is fitted with one- and two-compartment oral "
                    "models with a lag time (sums of exponentials, started from curve stripping), and the model with "
                    f"the lower AIC summed over subjects is kept. {blq}. A BLQ sample says only that the concentration "
                    f"was below {lloq} mg/L; entering it as 0 would drag the elimination phase down, so it enters the "
                    "likelihood as P(value < LLOQ) (censoring, Beal's M3). The assay error is partly proportional, so "
                    "an unweighted fit on the concentration scale would let the peak dominate; I weight each point by "
                    f"the stated error model (SD = sqrt({sd_add}^2 + ({cv} x C)^2)), iterating the weights from the "
                    "current fit. CL/F is the dose over the fitted AUC(0-inf), the terminal half-life is ln 2 over the "
                    "slowest rate, and both are summarised as geometric means across subjects.")
        fit_code = (f'lloq, cv, sd_add = {lloq}, {cv}, {sd_add}\n'
                    'table = subject_table(df)\n'
                    'n_cmt, fits, aic = fit_study(table, lloq, cv, sd_add)\n'
                    'print("AIC summed over subjects: 1 compartment %.1f, 2 compartments %.1f -> %d compartments"\n'
                    '      % (aic[1], aic[2], n_cmt))\n'
                    'for sid, p in fits.items():\n'
                    '    auc, h = auc_half_life(p, n_cmt)\n'
                    '    print(f"subject {sid}: CL/F {table[sid][0] / auc:.3f}, terminal t1/2 {h:.2f} h, lag {p[-1]:.2f} h")\n'
                    'cl_f, t_half = summarise(table, n_cmt, fits)\n'
                    'print("FINAL cl_f = %.6f t_half = %.6f" % (cl_f, t_half))')
        return [Cell("I load the concentrations and look at the design: how many samples are BLQ, how far out each "
                     "subject stays measurable, and the shape of the mean log-concentration curve.", inspect),
                Cell(prose2, fit_code, uses=(exponentials, pk_curve, auc_half_life, stripping_starts, fit_subject,
                                             fit_study, subject_table, summarise))]


# ----------------------------------------------------------------------------- real data: theophylline
THEOPH_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv"
THEOPH_PUBLISHED = {"cl_f": float(np.exp(-3.2145)), "t_half": float(np.log(2) / np.exp(-2.4327))}


class TheophSource(RealSource):
    """The classic theophylline study (R's datasets::Theoph, from Boeckmann, Sheiner & Beal 1994,
    NONMEM Users Guide V): 12 subjects, one oral dose each (mg per kg), 11 samples over 25 h, no
    value below the limit of quantification. Published answer: Pinheiro & Bates (2000), Mixed-
    Effects Models in S and S-PLUS, section 6.4, model fm1Theo.nlme (first-order open
    one-compartment, random effects on all three parameters): lCl = -3.2145, lKe = -2.4327, so
    CL/F = 0.04018 L/h/kg and t1/2 = ln 2 / exp(lKe) = 7.894 h. Population typical values of
    log-normal parameters estimate the geometric means over subjects that the question asks
    for (the subjects' own nlsList fits give 0.04023 L/h/kg and 7.880 h). The residual SD of
    that fit, 0.68 mg/L, is stated as the additive error model, because the assay's precision
    was not published with the data.

    The question is the family's own; the family's reference -- which chooses the structural
    model by AIC -- selects one compartment on these data and lands within tolerance of both
    published values, so `two_compartment_forced` is the trap that is live here, where
    `one_compartment` is live on every generated instance.

      venv/bin/python -m sage real --source theoph --families pk_oral --out ROOT [--raw DIR]
    """
    label = "pk_real_theoph"
    keep_controls = False

    def __init__(self, fam: str = "pk_oral", raw: Path | None = None, limit: int = 0):
        from sage.registry import get
        self.family = get(fam)
        if raw is None:
            from paths import data_root
            raw = data_root() / "raw" / "theoph"
        self.raw = Path(raw)
        self.note = ("ground truth is the published population estimate (Pinheiro & Bates 2000, fm1Theo.nlme), not an "
                     "analysis output; the family's reference analysis of these data (one compartment chosen by AIC) "
                     "reproduces it within tolerance on both keys. Ablations are scored against the published value: "
                     "a trap inside tolerance on these data has fails_at_step None")

    def keys(self) -> list:
        return ["theoph"]

    def fetch(self) -> Path:
        """Theoph.csv by plain HTTPS from Rdatasets (the same table as R's datasets::Theoph), once,
        converted to the family's file format."""
        import urllib.request
        src, out = self.raw / "Theoph.csv", self.raw / CONC
        if not out.exists():
            self.raw.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                urllib.request.urlretrieve(THEOPH_URL, src)
            th = pd.read_csv(src)
            df = pd.DataFrame({"subject": th["Subject"].astype(int), "dose_mg_per_kg": th["Dose"],
                               "time_h": th["Time"], "conc_mg_L": th["conc"]})
            df.sort_values(["subject", "time_h"], kind="stable").to_csv(out, index=False)
        return out

    def load(self, key) -> RealTarget:
        csv = self.fetch()
        df = pd.read_csv(csv)
        lo, hi = df["dose_mg_per_kg"].min(), df["dose_mg_per_kg"].max()
        meta = {"n": int(df["subject"].nunique()), "dose_unit": "mg/kg", "t_end": float(df["time_h"].max()),
                "lloq": 0.0, "cv": 0.0, "sd_add": 0.68, "no_blq": True, "population": "adult subjects",
                "study": "In a pharmacokinetic study of the anti-asthmatic drug theophylline",
                "dose_text": f"{lo:.1f}-{hi:.1f} mg per kg of body weight (each subject's dose is in the file)",
                "error_text": ("The assay's precision was not reported with these data; published analyses of them "
                               "model the measurement error as additive, with a residual standard deviation of about "
                               "0.68 mg/L.")}
        prov = {"benchmark": "R datasets::Theoph (Boeckmann, Sheiner & Beal 1994)", "source_url": THEOPH_URL,
                "published": "Pinheiro & Bates (2000) section 6.4, fm1Theo.nlme: lKe = -2.4327, lKa = 0.4509, "
                             "lCl = -3.2145 (reproduced with nlme 3.1: -2.4327, 0.4514, -3.2145)",
                "published_answer": THEOPH_PUBLISHED}
        return RealTarget("pk_real_theoph", {"df": df, "meta": meta}, dict(THEOPH_PUBLISHED), prov, {CONC: csv})


FAMILIES = [OralPK()]
REAL_SOURCES = {"theoph": TheophSource}
