"""Health care analytics: hospital quality profiling and diagnostic test accuracy (a SAGE plug-in).

Two standard analyses of routinely collected health care data, each with the irregularities such data carry.

`hc_quality` -- risk-adjusted hospital outcomes (health services research). One year of admissions for one
condition to a region's 20-60 hospitals (12,000-40,000 admissions), linked to the death registry, as a quality
board receives them. The DGP builds in:

  * case mix that differs systematically between hospitals (a shift of each hospital's age, comorbidity,
    severity and emergency mix, common to the four plus one per factor): the hospital under review takes
    markedly sicker or healthier patients than the region, so its crude death rate misleads;
  * hospital sizes spanning two orders of magnitude (15-3,000 admissions): a small hospital's death rate is
    mostly chance, so the spread of raw ratios overstates how much hospitals truly differ;
  * an outcome that must be built from dates: death within 30 days of admission, in hospital or after
    discharge (hospitals that discharge early have more of their deaths at home), with registry deaths
    after day 30 also on file;
  * a logistic risk model with a normally distributed hospital effect on the log-odds (the latent truth).

  Reference (the gold, on the sampled admissions):
    expected_deaths  indirect standardisation: a logistic regression of 30-day death on age, comorbidity,
                     severity and emergency admission fitted to every admission with no hospital term; the
                     named hospital's expected deaths are the sum of its admissions' fitted probabilities
    smr              observed 30-day deaths / expected_deaths
    mor              the median odds ratio (Merlo et al. 2006) exp(sqrt(2) z_0.75 s) from a random-intercept
                     logistic regression with the same risk factors, fitted by maximum likelihood with
                     adaptive Gauss-Hermite quadrature (7 nodes per hospital, centred on its posterior mode)
  Latent (for I1): expected deaths under the population-averaged risk model fitted to the TRUE risks of the
  sampled admissions (the estimand indirect standardisation targets), the named hospital's true expected deaths
  over that, and the MOR of the realised hospital effects.
  Tolerance: expected_deaths max(2 SE, 1%) with the delta-method SE of the risk model; smr 2 SE with
  SE = smr sqrt(1/O + (SE_E/E)^2); mor max(2 SE, 0.02), SE from the Hessian of the quadrature likelihood.

  Steps, skills and traps:
    1 inspect       in_hospital_deaths_only    deaths at discharge (discharge_status) as the outcome: misses
                                               deaths after discharge, counts late deaths of long stays
    1 inspect       any_recorded_death         every registry death, including those after day 30
    2 normalise     crude_no_casemix           expected = admissions x the regional crude rate; the MOR
                                               from a random-intercept model without the risk factors
    3 model-choice  hospital_in_casemix_model  the risk model includes hospital (fixed effects), so the
                                               expected deaths equal the observed and the SMR is 1
    4 uncertainty   raw_smr_spread             the MOR from the spread of the hospitals' raw log SMRs
                                               (log((O+0.5)/(E+0.5))): chance variation of small hospitals
                                               read as real differences
    -  variant      laplace_approximation      the random-intercept model by the Laplace approximation (one
                                               quadrature node; lme4's default)

`hc_dx` -- diagnostic accuracy under partial verification (clinical diagnostics). A cohort of 15,000-50,000
patients in which every patient had a continuous index test (PSA, D-dimer, faecal immunochemical test or
NT-proBNP) measured at one of two laboratories, and only some went on to the reference standard (biopsy, CT
pulmonary angiography, colonoscopy, echocardiography). The DGP builds in:

  * verification (referral) that depends on the index result AND on a clinical covariate that also predicts
    disease (abnormal rectal examination, a high Wells score, rectal bleeding, previous myocardial
    infarction): missing at random given both, so the referred patients are not representative; the
    referral mechanism leaves 25-70% of the diseased unverified, so the correction always matters;
  * on 60% of the D-dimer, FIT and NT-proBNP draws, one laboratory reports in another unit (D-dimer units
    rather than fibrinogen-equivalent units, ng Hb/mL buffer rather than ug Hb/g faeces, pmol/L rather than
    pg/mL), so the stated threshold applies only after conversion;
  * a binormal (log scale) index test, so the ROC area is defined over the continuous value.

  Reference: the probability of referral in each combination of test result and covariate (the referred
  fraction), each referred patient weighted by its inverse (Alonzo & Pepe 2005; with these saturated cells it
  equals Begg & Greenes' 1983 estimator with the covariate); sensitivity and specificity are the weighted
  proportions and the AUC the weighted Mann-Whitney statistic among the referred.
  Latent: the whole cohort's sensitivity, specificity and AUC with every patient's true disease status.
  Tolerance: sensitivity max(2 SE, 0.01) and specificity max(2 SE, 0.005), SE by the delta method over the
  cells; auc max(2 SE, 0.01), SE from 100 bootstrap resamples of the whole cohort.

  Steps, skills and traps:
    1 normalise     units_not_harmonised       the threshold applied to the raw values (cond: live on the
                                               draws where one laboratory reports in another unit)
    2 qc            unreferred_as_negative     patients never referred counted as disease-free
    3 model-choice  referred_only              complete-case accuracy among the referred
    4 model-choice  referral_by_test_only      referral probabilities by test result only (Begg and Greenes
                                               without the covariate)
    5 estimate      auc_of_binary_result       the "AUC" of the dichotomised test, (sensitivity + specificity)/2
    -  variant      logistic_referral_model    referral probabilities from a logistic regression on the test
                                               result and the covariate (main effects)

Real data: `SpectSource` (sage real --source spect --families hc_dx) asks the hc_dx question of the SPECT
thallium study (Cecil et al. 1996; 2,688 patients, 471 verified by coronary angiography), from CRAN's
PVBcorrect package, graded on sensitivity and specificity against the published estimate (see its docstring).
`hc_quality` has no real source: public hospital-outcome releases (CMS Hospital Compare, NHS SHMI, state
cardiac surgery reports) give hospital-level observed and expected counts, and the patient-level data a
risk model needs (Medicare claims, Hospital Episode Statistics) are licensed.

The analysis functions are self-contained (numpy/pandas/scipy/statsmodels/scikit-learn), so a demonstration
embeds their source verbatim: the trace and the ground truth are the same code.
"""
from __future__ import annotations

import bz2
import datetime as dt
import gzip
import lzma
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from sage.core import Cell, Family, RealSource, RealTarget, Sample, Trap

ADMISSIONS, SCREENING = "admissions.csv", "screening.csv"
RISK = ("age", "comorbidity", "severity", "emergency")
NAN = float("nan")


# ----------------------------------------------------------------------------- analysis: hc_quality
def thirty_day_deaths(df, days=30):
    """1 if the registry records a death within `days` days of admission, in hospital or after discharge."""
    gap = (pd.to_datetime(df["death_date"]) - pd.to_datetime(df["admission_date"])).dt.days
    return (gap <= days).astype(int).to_numpy()


def casemix_expected(df, died, risk=RISK):
    """Indirect standardisation: a logistic regression of death on the risk factors, fitted to every admission
    with no hospital term. Returns each admission's fitted (expected) risk and the fit."""
    import statsmodels.api as sm
    X = sm.add_constant(df[list(risk)].astype(float))
    fit = sm.GLM(np.asarray(died, float), X, family=sm.families.Binomial()).fit()
    return np.asarray(fit.fittedvalues), fit


def random_intercept_logit(y, X, hospital, nodes=7, se=True):
    """Maximum-likelihood random-intercept logistic regression, logit P(y = 1) = X b + u_h with u_h ~ N(0, s^2),
    by adaptive Gauss-Hermite quadrature: the nodes are centred on each hospital's posterior mode and scaled by
    its curvature (nodes=1 is the Laplace approximation). X includes a constant column. Returns (b, s, SE of log s)."""
    import statsmodels.api as sm
    from numpy.polynomial.hermite_e import hermegauss
    o = np.argsort(np.asarray(hospital), kind="stable")
    y, X, h = np.asarray(y, float)[o], np.asarray(X, float)[o], np.asarray(hospital)[o]
    mu, sc = X.mean(0), X.std(0)
    const = sc == 0
    mu[const], sc[const] = 0.0, 1.0
    X = (X - mu) / sc                                         # standardised columns, for the optimiser
    X[:, const] = 1.0
    start = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    g = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(h)]))
    per = lambda a: np.add.reduceat(a, start, axis=0)       # sum over each hospital's admissions
    z, w = hermegauss(nodes) if nodes > 1 else (np.zeros(1), np.full(1, np.sqrt(2 * np.pi)))
    lw = np.log(w) + z ** 2 / 2
    u = np.zeros(len(start))

    def nll(theta):                                           # -log likelihood and its gradient
        nonlocal u
        b, s = theta[:-1], np.exp(theta[-1])
        eta = X @ b
        for _ in range(100):                                  # posterior modes by Newton, warm-started
            p = expit(eta + u[g])
            info = per(p * (1 - p)) + 1 / s ** 2
            step = (per(y - p) - u / s ** 2) / info
            u = u + step
            if np.max(np.abs(step)) < 1e-9:
                break
        sd = 1 / np.sqrt(info)
        uk = u[:, None] + sd[:, None] * z                     # the nodes: hospitals x nodes
        e = eta[:, None] + uk[g]
        t = per(y[:, None] * e - np.logaddexp(0, e)) - uk ** 2 / (2 * s * s) - np.log(s) + np.log(sd)[:, None] + lw
        L = logsumexp(t, axis=1)
        wt = np.exp(t - L[:, None])                           # posterior weight of each node
        grad_b = X.T @ np.sum(wt[g] * (y[:, None] - expit(e)), axis=1)
        grad_s = np.sum(wt * (uk ** 2 / s ** 2 - 1))
        return len(start) * np.log(2 * np.pi) / 2 - np.sum(L), -np.r_[grad_b, grad_s]
    b0 = sm.GLM(y, X, family=sm.families.Binomial()).fit().params
    r = minimize(nll, np.r_[b0, np.log(0.3)], jac=True, method="BFGS", options={"gtol": 1e-6})
    se_log_s = NAN
    if se:                                                    # SE of log s from the numerical Hessian
        k = len(r.x)
        H = np.array([nll(r.x + 1e-4 * e)[1] - nll(r.x - 1e-4 * e)[1] for e in np.eye(k)]) / 2e-4
        se_log_s = float(np.sqrt(np.linalg.inv((H + H.T) / 2)[-1, -1]))
    b = r.x[:-1] / sc
    b[const] -= np.sum(b * mu) / const.sum()                  # back to the original columns
    return b, float(np.exp(r.x[-1])), se_log_s


def median_odds_ratio(s):
    """MOR = exp(sqrt(2) z_0.75 s) for a between-hospital SD s of the log-odds (Merlo et al. 2006)."""
    return float(np.exp(np.sqrt(2) * norm.ppf(0.75) * s))


# ----------------------------------------------------------------------------- analysis: hc_dx
def corrected_accuracy(T, X, D, referred, score):
    """Sensitivity, specificity and AUC corrected for partial verification: P(referred) is estimated as the
    referred fraction in each combination of test result T and covariate X, and each referred patient is
    weighted by its inverse (inverse-probability weighting; with these cells, Begg and Greenes' estimator)."""
    T, X, r = np.asarray(T, int), np.asarray(X, int), np.asarray(referred, bool)
    cell = T * (X.max() + 1) + X
    w = 1 / (np.bincount(cell, r) / np.bincount(cell))[cell[r]]
    d, t = np.asarray(D, float)[r], T[r]
    sens = np.sum(w * d * t) / np.sum(w * d)
    spec = np.sum(w * (1 - d) * (1 - t)) / np.sum(w * (1 - d))
    auc = roc_auc_score(d, np.asarray(score, float)[r], sample_weight=w)
    return float(sens), float(spec), float(auc)


def accuracy_se(T, X, D, referred):
    """Delta-method SEs of the cell-weighted sensitivity and specificity: binomial error in each cell's
    disease fraction among the referred, and multinomial error in the cells' shares of the cohort."""
    T, X, r = np.asarray(T, int), np.asarray(X, int), np.asarray(referred, bool)
    m = X.max() + 1
    cell = T * m + X
    K = 2 * m
    n = np.bincount(cell, minlength=K).astype(float)
    nr = np.bincount(cell[r], minlength=K).astype(float)
    q = np.bincount(cell[r], np.asarray(D, float)[r], minlength=K) / np.maximum(nr, 1)
    Tc, p = (np.arange(K) // m).astype(float), n / n.sum()
    out = []
    for qq, tt in ((q, Tc), (1 - q, 1 - Tc)):
        A = np.sum(p * qq)
        est = np.sum(p * qq * tt) / A
        var = (np.sum((p * (tt - est) / A) ** 2 * q * (1 - q) / np.maximum(nr, 1))
               + np.sum(p * (qq * (tt - est) / A) ** 2) / n.sum())
        out.append(float(np.sqrt(var)))
    return tuple(out)


# ----------------------------------------------------------------------------- shared
class HealthcareFamily(Family):
    domain = "healthcare"
    max_tries = 12
    rng_policy = "stream"
    demo_timeout = 300

    def _cached(self, data, key, fn):
        cache = data.setdefault("_cache", {})
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    def extra_provenance(self, inst) -> dict:
        m = inst.sample.data["meta"]
        return {"knobs": inst.knobs, "design": {k: v for k, v in m.items() if np.isscalar(v) or v is None},
                "reference_diagnostics": {k: v for k, v in inst.a.gold.items() if k.startswith("_") and np.isscalar(v)},
                "skills": {str(s): v for s, v in self.skills.items()}}

    def sanity(self, gold, tol, latent) -> list[str]:
        bad = [k for k in self.keys if not (np.isfinite(gold[k]) and np.isfinite(tol[k]) and tol[k] > 0)]
        return [f"{k}_not_estimable" for k in bad]


def _dates(start, days) -> np.ndarray:
    return (pd.Timestamp(start) + pd.to_timedelta(np.asarray(days), "D")).strftime("%Y-%m-%d").to_numpy()


# ----------------------------------------------------------------------------- hc_quality
CONDITIONS = (("acute myocardial infarction", 0.06, 0.12), ("heart failure", 0.07, 0.13), ("pneumonia", 0.08, 0.15),
              ("acute ischaemic stroke", 0.08, 0.14), ("hip fracture", 0.05, 0.09), ("sepsis", 0.10, 0.15),
              ("an exacerbation of chronic obstructive pulmonary disease", 0.05, 0.10))
MOR_TEXT = "MOR = exp(0.6745 x sqrt(2 v))"


class HospitalQuality(HealthcareFamily):
    name = "hc_quality"
    keys = ("expected_deaths", "smr", "mor")
    steps = {1: "build the outcome as defined: death within 30 days of admission, in hospital or after discharge",
             2: "adjust for case mix: expected deaths from a patient-level risk model (indirect standardisation)",
             3: "take the expected deaths from the regional risk model without the hospital's own term",
             4: "separate true between-hospital variation from chance (a random-intercept model), not raw SMRs"}
    skills = {1: "inspect", 2: "normalise", 3: "model-choice", 4: "uncertainty"}
    seed_blocks = {"train": 19_000_000, "test": 19_050_000}
    tol_rule = ("expected_deaths: max(1%, 2 SE), delta-method SE of the risk model; smr: 2 SE, SE = smr "
                "sqrt(1/O + (SE_E/E)^2); mor: max(0.02, 2 SE), SE from the Hessian of the quadrature likelihood")
    note = ("ground truth is the reference analysis (30-day deaths from the registry dates; a logistic risk model of "
            "age, comorbidity, severity and emergency admission fitted to all admissions without a hospital term, "
            "expected deaths summed over the named hospital; a random-intercept logistic model by adaptive "
            "Gauss-Hermite quadrature for the median odds ratio) on the sampled admissions, not the latent "
            "parameters; deviation is in units of tolerance, maxed over the answer keys")

    def knobs(self, rng):
        return {"casemix": 1.0, "effect": 1.0, "small": 1.0, "post": 1.0}

    def draw(self, rng, k) -> Sample:
        import statsmodels.api as sm
        cond, r_lo, r_hi = CONDITIONS[int(rng.integers(len(CONDITIONS)))]
        H = int(rng.integers(20, 61))
        rate, tau, s_cm = rng.uniform(r_lo, r_hi), rng.uniform(0.12, 0.32), rng.uniform(0.4, 0.8)
        small = rng.random(H) < rng.uniform(0.4, 0.7)                 # low-volume hospitals
        sizes = np.where(small, np.exp(rng.uniform(np.log(15), np.log(150), H)) / k["small"],
                         np.exp(rng.uniform(np.log(300), np.log(2500), H)))
        sizes = np.maximum(sizes.astype(int), 8)
        j = int(rng.integers(H))                                       # the hospital under review
        sizes[j] = int(rng.integers(1000, 3001))
        c = rng.normal(0, s_cm, H)                                     # case-mix shift of each hospital
        c[j] = rng.choice([-1.0, 1.0]) * min(rng.uniform(0.4, 2.5) * k["casemix"], 3.5) * s_cm
        shift = c[:, None] + rng.normal(0, 0.5 * s_cm, (H, 4))         # ... and of each risk factor
        u = tau * rng.permutation(norm.ppf((np.arange(H) + 0.5) / H))  # hospital effects on the log-odds
        u[j] = rng.choice([-1.0, 1.0]) * min(rng.uniform(0.2, 0.5) * k["effect"], 0.9)
        u -= u.mean()
        stay = np.exp(rng.normal(np.log(5), 0.35, H)) / k["post"]      # discharge practice: typical stay, days
        g = np.repeat(np.arange(H), sizes)
        n, sh = len(g), shift[g]
        age = np.clip(np.round(rng.normal(68 + 6 * sh[:, 0], 13)), 18, 100).astype(int)
        com = np.minimum(rng.poisson(np.exp(0.4 + 0.35 * sh[:, 1])), 12)
        sev = rng.binomial(10, expit(-1.3 + 0.5 * sh[:, 2]))
        em = (rng.random(n) < expit(0.8 + 0.9 * sh[:, 3])).astype(int)
        beta = np.array([rng.uniform(0.03, 0.06), rng.uniform(0.15, 0.3), rng.uniform(0.15, 0.3), rng.uniform(0.6, 1.2)])
        lp = np.column_stack([age - 70, com, sev, em]) @ beta + u[g]
        p = expit(brentq(lambda a: np.mean(expit(a + lp)) - rate, -15, 5) + lp)
        died = rng.random(n) < p                                       # death within 30 days of admission
        adm = rng.integers(0, 365, n)                                  # admission day within the year
        los = np.maximum(1, np.round(rng.gamma(2.0, stay[g] * (1 + 0.06 * sev) / 2))).astype(int)
        t30 = np.minimum(np.where(rng.random(n) < 0.55, np.floor(rng.exponential(4.0, n)), rng.integers(0, 31, n)), 30)
        late = ~died & (rng.random(n) < 0.7 * p)                       # registry deaths after day 30
        dday = np.where(died, t30, np.where(late, rng.integers(31, 366, n), -1)).astype(int)
        in_hosp = (dday >= 0) & (dday <= np.where(died, np.round(1.5 * los), los))
        start = dt.date(int(rng.integers(2014, 2024)), 1, 1) + dt.timedelta(days=int(rng.integers(0, 365)))
        codes = np.array([f"H{i + 1:02d}" for i in range(H)])
        o = np.argsort(adm, kind="stable")
        df = pd.DataFrame({"admission_id": [f"A{i + 1:06d}" for i in range(n)], "hospital": codes[g][o],
                           "admission_date": _dates(start, adm[o]),
                           "discharge_date": _dates(start, (adm + np.where(in_hosp, dday, los))[o]),
                           "discharge_status": np.where(in_hosp, "died", "alive")[o], "age": age[o],
                           "comorbidity": com[o], "severity": sev[o], "emergency": em[o],
                           "death_date": pd.Series(_dates(start, (adm + np.maximum(dday, 0))[o])).where(dday[o] >= 0)})
        cutoff = (start + dt.timedelta(days=365 + 90)).isoformat()
        named = str(codes[j])
        meta = {"condition": cond, "H": H, "n": n, "named": named, "cutoff": cutoff,
                "first": df["admission_date"].iloc[0], "last": df["admission_date"].iloc[-1]}
        # latent: the population-averaged risk model the pooled regression estimates, fitted to the TRUE risks
        pt = p[o]
        h = (df["hospital"] == named).to_numpy()
        X = sm.add_constant(df[list(RISK)].astype(float))
        e_lat = float(np.asarray(sm.GLM(pt, X, family=sm.families.Binomial()).fit().fittedvalues)[h].sum())
        latent = {"expected_deaths": e_lat, "smr": float(pt[h].sum() / e_lat), "mor": median_odds_ratio(np.std(u)),
                  "tau": float(tau), "hospital_log_odds_sd": float(np.std(u)), "named_log_odds_effect": float(u[j]),
                  "named_casemix_shift": [float(x) for x in shift[j]], "named_admissions": int(sizes[j]),
                  "true_deaths_expected_at_named": float(pt[h].sum()), "rate_30d": float(died.mean()),
                  "share_30d_deaths_after_discharge": float(np.mean(~in_hosp[died])), "n_small": int(small.sum()),
                  "coefficients": [float(x) for x in beta]}
        return Sample({"df": df, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / ADMISSIONS, index=False)
        return [ADMISSIONS]

    # ------------------------------------------------------------------ analysis
    def _outcome(self, data, kind):
        df = data["df"]
        if kind == "in_hospital":
            return (df["discharge_status"] == "died").astype(int).to_numpy()
        if kind == "any_death":
            return df["death_date"].notna().astype(int).to_numpy()
        return thirty_day_deaths(df)

    def _fit(self, data, outcome="died30", adjust=True, nodes=7, se=False):
        """The analysis with one stage switched off: another outcome, no case-mix adjustment, or another
        quadrature. Cached, because the random-intercept fits dominate the cost."""
        def run():
            df = data["df"]
            died = self._outcome(data, outcome)
            h = (df["hospital"] == data["meta"]["named"]).to_numpy()
            if adjust:
                p, fit = casemix_expected(df, died)
                X = np.asarray(fit.model.exog)
                grad = (p[h] * (1 - p[h])) @ X[h]
                se_E = float(np.sqrt(grad @ np.asarray(fit.cov_params()) @ grad))
            else:
                p, X, se_E = np.full(len(df), died.mean()), np.ones((len(df), 1)), NAN
            _, s, se_log_s = random_intercept_logit(died, X, df["hospital"], nodes=nodes, se=se)
            O, E = float(died[h].sum()), float(p[h].sum())
            return {"expected_deaths": E, "smr": O / E, "mor": median_odds_ratio(s), "_O": O, "_se_E": se_E,
                    "_s": s, "_se_log_s": se_log_s, "_p": p}
        return self._cached(data, ("_fit", outcome, adjust, nodes, se), run)

    def reference(self, data) -> dict:
        return {k: v for k, v in self._fit(data, se=True).items() if k != "_p"}

    def tolerance(self, gold) -> dict:
        E, smr, mor = gold["expected_deaths"], gold["smr"], gold["mor"]
        return {"expected_deaths": max(2 * gold["_se_E"], 0.01 * E),
                "smr": 2 * smr * float(np.sqrt(1 / max(gold["_O"], 1) + (gold["_se_E"] / E) ** 2)),
                "mor": max(2 * mor * np.sqrt(2) * norm.ppf(0.75) * gold["_s"] * gold["_se_log_s"], 0.02)}

    def sanity(self, gold, tol, latent) -> list[str]:
        out = super().sanity(gold, tol, latent)
        return out + ([] if gold["_s"] > 0.02 and gold["_se_log_s"] < 1.0 else ["between_hospital_sd_not_identified"])

    def traps(self, data) -> dict:
        ref = self._fit(data, se=True)
        pick = lambda r: {k: float(r[k]) for k in self.keys}
        by = pd.DataFrame({"h": data["df"]["hospital"], "O": thirty_day_deaths(data["df"]), "E": ref["_p"]}).groupby("h").sum()
        s_raw = float(np.std(np.log((by["O"] + 0.5) / (by["E"] + 0.5)), ddof=1))
        return {"in_hospital_deaths_only": Trap(pick(self._fit(data, "in_hospital")), 1),
                "any_recorded_death": Trap(pick(self._fit(data, "any_death")), 1),
                "crude_no_casemix": Trap(pick(self._fit(data, adjust=False)), 2),
                # a hospital term makes the fitted deaths of each hospital sum to its observed deaths (the score
                # equation of its indicator), so expected = observed and the SMR is 1 whatever the hospital did
                "hospital_in_casemix_model": Trap({"expected_deaths": ref["_O"], "smr": 1.0, "mor": ref["mor"]}, 3),
                "raw_smr_spread": Trap({**pick(ref), "mor": median_odds_ratio(s_raw)}, 4),
                "laplace_approximation": Trap({**pick(ref), "mor": self._fit(data, nodes=1)["mor"]}, None)}

    def strengthen(self, k, dead):
        k = dict(k)
        if "crude_no_casemix" in dead:
            k["casemix"] = min(k["casemix"] * 1.4, 3.0)
        if "hospital_in_casemix_model" in dead:
            k["effect"] = min(k["effect"] * 1.4, 3.0)
        if "raw_smr_spread" in dead:
            k["small"] = min(k["small"] * 1.4, 4.0)
        if {"in_hospital_deaths_only", "any_recorded_death"} & set(dead):
            k["post"] = min(k["post"] * 1.4, 3.0)
        return k

    def shortcuts(self, data) -> dict:
        """Data-light answers: expected deaths = the hospital's admissions x the regional crude rate (or its
        observed deaths), with the MOR from the spread of the hospitals' crude log death-rate ratios."""
        df, died = data["df"], thirty_day_deaths(data["df"])
        h = (df["hospital"] == data["meta"]["named"]).to_numpy()
        by = pd.DataFrame({"h": df["hospital"], "O": died, "N": 1}).groupby("h").sum()
        s = float(np.std(np.log((by["O"] + 0.5) / (by["N"] * died.mean() + 0.5)), ddof=1))
        E, O = float(h.sum() * died.mean()), float(died[h].sum())
        return {"crude_expected_crude_spread": {"expected_deaths": E, "smr": O / E, "mor": median_odds_ratio(s)},
                "observed_as_expected": {"expected_deaths": O, "smr": 1.0, "mor": median_odds_ratio(s)}}

    # ------------------------------------------------------------------ prompt
    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        head = (f"You are a health services researcher at a regional health-care quality board. The region's {m['H']} "
                f"hospitals admitted {m['n']} patients with {m['condition']} between {m['first']} and {m['last']}; every "
                f"admission is in the file below, linked to the national death registry (deaths registered up to "
                f"{m['cutoff']}). The board is reviewing hospital {m['named']} and how much the "
                "hospitals' 30-day mortality differs.\n\n"
                "Data files:\n"
                f"- data_files/{ADMISSIONS}: one row per admission. Columns: admission_id, hospital (code), "
                "admission_date, discharge_date, discharge_status (alive or died: the patient's status when leaving "
                "this hospital), age (years), comorbidity (Charlson comorbidity index), severity (admission severity "
                "score, 0-10; higher is sicker), emergency (1 = emergency admission, 0 = elective), death_date (date of "
                "death from any cause, in or out of hospital, from the registry linkage; empty if no death was "
                f"registered by {m['cutoff']}).\n\n"
                "The outcome is death from any cause within 30 days of admission (death_date at most 30 days after "
                "admission_date), in hospital or after discharge. Age, comorbidity, severity and emergency admission "
                "are the recognised risk factors for it.\n\n")
        q = (f"Question: (1) How does hospital {m['named']}'s 30-day mortality compare with what its case mix predicts? "
             "Report expected_deaths, the number of deaths expected among its admissions if each of its patients had "
             "the average risk of patients with the same age, comorbidity, severity and admission type across all the "
             "region's hospitals, and smr, its standardised mortality ratio (observed 30-day deaths / expected_deaths). "
             "(2) How much do the hospitals differ in the risk of death of the same patient, beyond chance? Report mor, "
             "the median odds ratio: for two patients with the same risk factors admitted to two randomly chosen "
             "hospitals, the median of the odds ratio of death between the higher-risk and the lower-risk hospital, "
             f"{MOR_TEXT}, where v is the between-hospital variance of the log-odds of death given the risk "
             "factors.\n\n")
        method = ("" if blind else
                  "Approach: build the outcome from death_date (at most 30 days after admission), not from "
                  "discharge_status, which misses deaths after discharge. Fit a logistic regression of 30-day death on "
                  "age, comorbidity, severity and emergency to every admission with no hospital term (a hospital term "
                  "would make each hospital's expected deaths equal its observed deaths); expected_deaths is the sum of "
                  f"the fitted probabilities over hospital {m['named']}'s admissions (indirect standardisation). For "
                  "mor, fit a random-intercept logistic regression (the same risk factors and a normally distributed "
                  "intercept per hospital) by maximum likelihood with adaptive Gauss-Hermite quadrature: v is the "
                  "intercept variance. The spread of the hospitals' raw SMRs overstates v, because small hospitals' "
                  "SMRs are mostly chance.\n\n")
        return head + q + method + ('Return your answer as JSON: {"answer": {"expected_deaths": <deaths>, "smr": <ratio>, '
                                    '"mor": <odds ratio>}}')

    def demo(self, cfg) -> list[Cell]:
        named = re.search(r"reviewing hospital (H\d+)", cfg["task"]).group(1)
        inspect = (f'import numpy as np, pandas as pd\ndf = pd.read_csv("data_files/{ADMISSIONS}", '
                   'parse_dates=["admission_date", "discharge_date", "death_date"])\n'
                   'print(df.head(3).to_string(index=False))\n'
                   'df["died30"] = ((df.death_date - df.admission_date).dt.days <= 30).astype(int)\n'
                   'in_hosp = df.discharge_status.eq("died")\n'
                   'print(len(df), "admissions,", df.hospital.nunique(), "hospitals |", df.died30.sum(), "deaths within 30 days,",\n'
                   '      "of them", (df.died30.eq(1) & ~in_hosp).sum(), "after discharge |", in_hosp.sum(), "died in hospital |",\n'
                   '      df.death_date.notna().sum(), "registry deaths in all")\n'
                   'size = df.groupby("hospital").size()\n'
                   'print("admissions per hospital: min", size.min(), "median", int(size.median()), "max", size.max(),\n'
                   '      "|", (size < 100).sum(), "hospitals with fewer than 100")\n'
                   f'risk = ["age", "comorbidity", "severity", "emergency"]\n'
                   f'mix = df.groupby(df.hospital.eq("{named}"))[risk + ["died30"]].mean().T\n'
                   f'mix.columns = ["other hospitals", "{named}"]\n'
                   'print(mix.round(3).to_string())')

        def prose2(outs):
            o = outs[0]
            num = lambda pat: [float(x) for x in re.search(pat, o, re.M).groups()] if re.search(pat, o, re.M) else None
            dd = num(r"\| (\d+) deaths within 30 days, of them (\d+) after discharge \| (\d+) died in hospital \| "
                     r"(\d+) registry deaths")
            sz = num(r"min (\d+) median (\d+) max (\d+)\s*\|\s*(\d+) hospitals with fewer than 100")
            mix = {r: num(rf"^{r}\s+([\d.]+)\s+([\d.]+)\s*$") for r in ("age", "emergency", "died30")}
            txt = (f"Of the {dd[0]:.0f} deaths within 30 days of admission, {dd[1]:.0f} happened after discharge, so "
                   f"discharge_status (which counts {dd[2]:.0f} deaths, some of them after day 30) is the wrong outcome, "
                   f"and so is any registry death ({dd[3]:.0f}, including later ones); I use death_date. " if dd else
                   "Some 30-day deaths happen after discharge, so I build the outcome from death_date. ")
            if all(mix.values()):
                a, e, d = mix["age"], mix["emergency"], mix["died30"]
                txt += (f"{named}'s patients differ from the other hospitals' (mean age {a[1]:.1f} against {a[0]:.1f}, "
                        f"{e[1]:.0%} emergency admissions against {e[0]:.0%}), so its crude 30-day death rate "
                        f"({d[1]:.1%} against {d[0]:.1%}) mixes case mix with performance. ")
            txt += ("For its expected deaths I fit a logistic regression of 30-day death on the four risk factors to "
                    "every admission, with no hospital term (a hospital term would make its expected deaths equal its "
                    f"observed deaths, an SMR of 1 by construction), and sum the fitted probabilities over {named}'s "
                    "admissions. ")
            if sz:
                txt += (f"Hospitals range from {sz[0]:.0f} to {sz[2]:.0f} admissions, {sz[3]:.0f} of them fewer than 100, "
                        "so the raw SMRs of small hospitals are mostly chance and their spread would overstate how much "
                        "hospitals differ. ")
            return txt + ("The between-hospital variance comes instead from a random-intercept logistic regression (the "
                          "same risk factors, a normal intercept per hospital) fitted by maximum likelihood with adaptive "
                          "Gauss-Hermite quadrature; the median odds ratio is exp(0.6745 sqrt(2 v)). The spread of the "
                          "raw log SMRs is printed for comparison.")
        fit_code = ('import statsmodels.api as sm\n'
                    'p, casemix = casemix_expected(df, df.died30)\n'
                    f'h = df.hospital.eq("{named}").to_numpy()\n'
                    'O, E = df.died30[h].sum(), p[h].sum()\n'
                    'print("risk model (log-odds):", casemix.params.round(4).to_dict())\n'
                    f'print(f"{named}: observed {{O}}, expected {{E:.2f}}, SMR {{O / E:.3f}} (crude ratio '
                    '{df.died30[h].mean() / df.died30.mean():.3f})")\n'
                    'X = sm.add_constant(df[risk].astype(float))\n'
                    'b, s, se_log_s = random_intercept_logit(df.died30, X, df.hospital)\n'
                    'by = pd.DataFrame({"hospital": df.hospital, "O": df.died30, "E": p}).groupby("hospital").sum()\n'
                    'raw_sd = np.std(np.log((by.O + 0.5) / (by.E + 0.5)), ddof=1)\n'
                    'print(f"between-hospital SD of the log-odds {s:.4f} (SE of log SD {se_log_s:.3f}); SD of the raw '
                    'log SMRs {raw_sd:.4f}")\n'
                    'print(f"MOR {median_odds_ratio(s):.4f} (the raw spread would give {median_odds_ratio(raw_sd):.4f})")\n'
                    'print("FINAL expected_deaths = %.6f smr = %.6f mor = %.6f" % (E, O / E, median_odds_ratio(s)))')
        return [Cell(f"I load the admissions, build the 30-day outcome from the registry dates and compare it with the "
                     f"discharge status, look at how many admissions each hospital has, and compare {named}'s case mix "
                     "with the other hospitals'.", inspect),
                Cell(prose2, fit_code, uses=(casemix_expected, random_intercept_logit, median_odds_ratio),
                     preamble="from scipy.optimize import minimize\nfrom scipy.special import expit, logsumexp\n"
                              "from scipy.stats import norm\nNAN = float(\"nan\")\nRISK = tuple(risk)")]


# ----------------------------------------------------------------------------- hc_dx
# index tests: unit, positivity threshold, reporting precision, and the other laboratory's unit as
# (name, multiplier, divisor, precision) with value_in_unit = value_in_alt x multiplier / divisor
TESTS = {
    "psa": dict(test="PSA", unit="ng/mL", thr=4.0, dec=2, alt=None, disease="prostate cancer",
                reference="prostate biopsy", who="men aged 50-74 taking part in a PSA screening study",
                cov="dre_abnormal", cov_text="digital rectal examination abnormal: 1 = abnormal, 0 = normal",
                cov_short="the digital rectal examination", prev=(0.06, 0.25)),
    "d_dimer": dict(test="D-dimer", unit="ng/mL FEU", thr=500.0, dec=-1, alt=("ng/mL DDU", 2.0, 1.0, 0),
                    conv="1 ng/mL DDU (D-dimer units) = 2 ng/mL FEU (fibrinogen-equivalent units)",
                    disease="pulmonary embolism", reference="CT pulmonary angiography",
                    who="outpatients with suspected pulmonary embolism", cov="wells_likely",
                    cov_text="two-level Wells score above 4, embolism likely: 1 = likely, 0 = unlikely",
                    cov_short="the two-level Wells score", prev=(0.06, 0.30)),
    "fit": dict(test="the faecal immunochemical test (FIT)", unit="ug Hb/g", thr=10.0, dec=1, alt=("ng Hb/mL", 1.0, 5.0, 0),
                conv="for this analyser, ug haemoglobin per g faeces = ng haemoglobin per mL buffer / 5",
                disease="advanced colorectal neoplasia (cancer or advanced adenoma)", reference="colonoscopy",
                who="patients referred from primary care with lower gastrointestinal symptoms", cov="rectal_bleeding",
                cov_text="rectal bleeding reported: 1 = yes, 0 = no", cov_short="whether they reported rectal bleeding",
                prev=(0.05, 0.25)),
    "nt_probnp": dict(test="NT-proBNP", unit="pg/mL", thr=400.0, dec=0, alt=("pmol/L", 8.457, 1.0, 1),
                      conv="1 pmol/L = 8.457 pg/mL", disease="heart failure", reference="echocardiography",
                      who="patients presenting to primary care with breathlessness or ankle swelling", cov="previous_mi",
                      cov_text="history of myocardial infarction: 1 = yes, 0 = no",
                      cov_short="a history of myocardial infarction", prev=(0.10, 0.35))}


def in_common_unit(values, units, alt_unit=None, mult=1.0, div=1.0):
    """Results reported in `alt_unit` converted to the threshold's unit (x mult / div); others unchanged."""
    values = np.asarray(values, float)
    return np.where(np.asarray(units) == alt_unit, values * mult / div, values) if alt_unit else values


class DiagnosticAccuracy(HealthcareFamily):
    name = "hc_dx"
    keys = ("sensitivity", "specificity", "auc")
    steps = {1: "express every result in one unit before applying the threshold",
             2: "treat patients never referred as missing a reference result, not as disease-free",
             3: "correct for verification that depends on the test result (weight by 1 / P(referred))",
             4: "model referral on everything it depended on: the test result and the clinical covariate",
             5: "take the ROC area over the continuous result, not at the single threshold"}
    skills = {1: "normalise", 2: "qc", 3: "model-choice", 4: "model-choice", 5: "estimate"}
    seed_blocks = {"train": 19_100_000, "test": 19_150_000}
    tol_rule = ("sensitivity: max(0.01, 2 SE), specificity: max(0.005, 2 SE), SE by the delta method over the (result, "
                "covariate) cells; auc: max(0.01, 2 SE), SE from 100 bootstrap resamples of the cohort")
    note = ("ground truth is the reference analysis (results in one unit; P(referred) as the referred fraction in each "
            "combination of test result and covariate; inverse-probability-weighted sensitivity, specificity and ROC "
            "area among the referred, i.e. Begg and Greenes' correction with the covariate) on the sampled cohort, not "
            "the latent parameters; deviation is in units of tolerance, maxed over the answer keys")

    def knobs(self, rng):
        return {"n": 1.0, "cov": 1.0, "sel": 1.0}

    def draw(self, rng, k) -> Sample:
        key = sorted(TESTS)[int(rng.integers(len(TESTS)))]
        t = TESTS[key]
        n = int(round(rng.integers(15000, 50001) * k["n"]))
        prev, px = rng.uniform(*t["prev"]), rng.uniform(0.2, 0.5)
        X = (rng.random(n) < px).astype(int)
        b = min(rng.uniform(1.2, 2.2) * k["cov"], 4.0)               # the covariate predicts disease
        a = brentq(lambda a: px * expit(a + b) + (1 - px) * expit(a) - prev, -15, 10)
        D = (rng.random(n) < expit(a + b * X)).astype(int)
        se_t, sp_t = rng.uniform(0.5, 0.95), rng.uniform(0.55, 0.97)
        s1, s0 = rng.uniform(0.5, 0.9), rng.uniform(0.4, 0.7)       # binormal on the log scale
        mu = np.where(D == 1, np.log(t["thr"]) + s1 * norm.ppf(se_t), np.log(t["thr"]) - s0 * norm.ppf(sp_t))
        v = np.exp(mu + np.where(D == 1, s1, s0) * (rng.standard_normal(n) + rng.uniform(0, 0.3) * X))
        site_b = rng.random(n) < rng.uniform(0.3, 0.6)
        mixed = bool(t["alt"] is not None and rng.random() < 0.6)   # site B reports in another unit
        alt_unit, mult, div, dec = t["alt"] or (None, 1.0, 1.0, 0)
        other = site_b & mixed
        reported = np.where(other, np.round(v * div / mult, dec), np.round(v, t["dec"]))
        units = np.where(other, alt_unit or "", t["unit"])
        value = in_common_unit(reported, units, alt_unit, mult, div)
        T = (value >= t["thr"]).astype(int)
        for _ in range(200):                     # referral that leaves 25-70% of the diseased unverified
            v0 = logit(rng.uniform(0.02, 0.4)) - np.log(k["sel"])    # more selective with k["sel"]
            vt, vx = rng.uniform(1.5, 5.0) + np.log(k["sel"]), min(rng.uniform(1.0, 3.5) * k["cov"], 5.0)
            # docs/DIRECTIONS.md A1 hazard toggle (sage/hazards.py): k["dep"] scales how strongly referral depends on the
            # result and the covariate; 0 = referral at random (the M5 selection hazard absent). NOT in knobs(), so every
            # existing draw is unchanged (1.0 * x == x exactly, and no extra random numbers are consumed).
            if "dep" in k:
                vt, vx = vt * k["dep"], vx * k["dep"]
            if 0.25 <= 1 - expit(v0 + vt * T[D == 1] + vx * X[D == 1]).mean() <= 0.7:
                break
        R = (rng.random(n) < expit(v0 + vt * T + vx * X)).astype(int)
        df = pd.DataFrame({"patient_id": [f"P{i + 1:05d}" for i in range(n)], "site": np.where(site_b, "B", "A"),
                           key: reported, "unit": units, t["cov"]: X, "referred": R,
                           "reference": pd.Series(np.where(D == 1, "positive", "negative")).where(R == 1)})
        meta = {"test": key, "col": key, "cov": t["cov"], "thr": t["thr"], "unit": t["unit"], "mixed": mixed,
                "alt_unit": alt_unit if mixed else None, "mult": mult, "div": div, "n": n, "binary_index": False}
        latent = {"sensitivity": float(T[D == 1].mean()), "specificity": float(1 - T[D == 0].mean()),
                  "auc": float(roc_auc_score(D, value)), "prevalence": float(D.mean()), "referred": float(R.mean()),
                  "diseased_unreferred": float(1 - R[D == 1].mean()),
                  "covariate_log_odds_disease": float(b), "referral_log_odds": [float(v0), float(vt), float(vx)],
                  "target_sensitivity": float(se_t), "target_specificity": float(sp_t)}
        return Sample({"df": df, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / SCREENING, index=False)
        return [SCREENING]

    # ------------------------------------------------------------------ analysis
    def _arrays(self, data, convert=True):
        df, m = data["df"], data["meta"]
        if m["binary_index"]:                                          # a real study reporting positive/negative
            T = df[m["col"]].eq("positive").astype(int).to_numpy()
            score = T.astype(float)
        else:
            score = (in_common_unit(df[m["col"]], df["unit"], m["alt_unit"], m["mult"], m["div"]) if convert
                     else df[m["col"]].to_numpy(float))
            T = (score >= m["thr"]).astype(int)
        return T, df[m["cov"]].to_numpy(int), df["referred"].to_numpy(int), df["reference"].eq("positive").to_numpy(float), score

    def _acc(self, data, convert=True, by_cov=True, logistic=False):
        def run():
            T, X, R, D, score = self._arrays(data, convert)
            if not logistic:
                return dict(zip(self.keys, corrected_accuracy(T, X if by_cov else 0 * X, D, R, score)))
            import statsmodels.api as sm                               # referral model: logit P = T + X
            pi = sm.GLM(R, sm.add_constant(np.column_stack([T, X]).astype(float)),
                        family=sm.families.Binomial()).fit().fittedvalues
            r = R == 1
            w, d, t = 1 / np.asarray(pi)[r], D[r], T[r]
            return {"sensitivity": float(np.sum(w * d * t) / np.sum(w * d)),
                    "specificity": float(np.sum(w * (1 - d) * (1 - t)) / np.sum(w * (1 - d))),
                    "auc": float(roc_auc_score(d, score[r], sample_weight=w))}
        return self._cached(data, ("_acc", convert, by_cov, logistic), run)

    def reference(self, data) -> dict:
        def run():
            T, X, R, D, score = self._arrays(data)
            se_sens, se_spec = accuracy_se(T, X, D, R)
            rng, boot = np.random.default_rng(20260915), []
            for _ in range(100):                                       # the whole estimator, resampled
                i = rng.integers(0, len(T), len(T))
                boot.append(corrected_accuracy(T[i], X[i], D[i], R[i], score[i])[2])
            neg = (R == 1) & (T == 0)
            return {**self._acc(data), "_se_sens": se_sens, "_se_spec": se_spec, "_se_auc": float(np.std(boot, ddof=1)),
                    "_min_cell_referred": int(np.bincount(T * (X.max() + 1) + X, R == 1).min()),
                    "_diseased_referred_negatives": int(D[neg].sum())}
        return dict(self._cached(data, ("_ref",), run))

    def tolerance(self, gold) -> dict:
        return {"sensitivity": max(2 * gold["_se_sens"], 0.01), "specificity": max(2 * gold["_se_spec"], 0.005),
                "auc": max(2 * gold["_se_auc"], 0.01)}

    def sanity(self, gold, tol, latent) -> list[str]:
        out = super().sanity(gold, tol, latent)
        if gold["_min_cell_referred"] < 30 or gold["_diseased_referred_negatives"] < 3:
            out.append("too_few_referred_to_correct")
        return out

    def traps(self, data) -> dict:
        ref = self._acc(data)
        T, X, R, D, score = self._arrays(data)
        r = R == 1
        D0 = np.where(r, D, 0.0)
        return {"units_not_harmonised": Trap(dict(self._acc(data, convert=False)), 1, bool(data["meta"]["mixed"])),
                "unreferred_as_negative": Trap({"sensitivity": float(T[D0 == 1].mean()),
                                                "specificity": float(1 - T[D0 == 0].mean()),
                                                "auc": float(roc_auc_score(D0, score))}, 2),
                "referred_only": Trap({"sensitivity": float(T[r & (D == 1)].mean()),
                                       "specificity": float(1 - T[r & (D == 0)].mean()),
                                       "auc": float(roc_auc_score(D[r], score[r]))}, 3),
                "referral_by_test_only": Trap(dict(self._acc(data, by_cov=False)), 4),
                "auc_of_binary_result": Trap({**ref, "auc": (ref["sensitivity"] + ref["specificity"]) / 2}, 5),
                "logistic_referral_model": Trap(dict(self._acc(data, logistic=True)), None)}

    def strengthen(self, k, dead):
        k = dict(k)
        if {"unreferred_as_negative", "referred_only"} & set(dead):
            k["sel"] = min(k["sel"] * 1.6, 6.0)                       # referral more selective: a larger bias
        if "auc_of_binary_result" in dead:
            k["n"] = min(k["n"] * 1.4, 2.0)                           # a larger cohort: smaller SEs
        if "referral_by_test_only" in dead:
            k["cov"] = min(k["cov"] * 1.3, 2.5)
        return k

    def shortcuts(self, data) -> dict:
        """Data-light answers: the complete-case sensitivity, specificity read as 1 - the cohort's positive
        rate (as if disease were rare), and their mean as the AUC."""
        T, X, R, D, score = self._arrays(data)
        se = float(T[(R == 1) & (D == 1)].mean())
        return {"complete_case_sens_positivity_spec": {"sensitivity": se, "specificity": float(1 - T.mean()),
                                                       "auc": (se + float(1 - T.mean())) / 2}}

    # ------------------------------------------------------------------ prompt
    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        t = TESTS[m["test"]]
        thr = f"{m['thr']:g} {m['unit']}"
        units = (f"unit (the unit of the result: site A reports in {m['unit']}, site B in {m['alt_unit']}; {t['conv']})"
                 if m["mixed"] else f"unit (the unit of the result: {m['unit']} at both sites)")
        head = (f"You are a clinical epidemiologist assessing {t['test']} as a test for {t['disease']} in {t['who']}. "
                f"Each of the {m['n']} patients in the cohort had {t['test']} measured at one of two laboratories (sites A "
                f"and B); a result of {thr} or more is positive. Only some patients went on to {t['reference']}, the "
                f"reference standard: the clinician decided, based on the {t['test']} result and on {t['cov_short']}, "
                "and nothing else about the patient influenced the decision.\n\n"
                "Data files:\n"
                f"- data_files/{SCREENING}: one row per patient. Columns: patient_id, site, {m['col']} (the {t['test']} "
                f"result), {units}, {m['cov']} ({t['cov_text']}), referred (1 = underwent {t['reference']}), reference "
                f"(the {t['reference']} result, positive or negative for {t['disease']}; empty if not referred).\n\n")
        q = (f"Question: how accurate is {t['test']} for {t['disease']} in this population? Report the sensitivity and "
             f"specificity of a positive result ({t['test']} at least {thr}) and auc, the area under the ROC curve of "
             f"the {t['test']} result, for the whole cohort, as if every patient had undergone {t['reference']}.\n\n")
        method = ("" if blind else
                  f"Approach: express every result in {m['unit']} before applying the threshold. Patients who were not "
                  "referred have no reference result: they are missing, not negative, and the referred patients are not "
                  f"representative, because referral depended on the result and on {m['cov']}. Correct for this partial "
                  f"verification by inverse-probability weighting (Begg and Greenes' method with {m['cov']}): estimate "
                  f"the probability of referral in each of the four groups of test result (positive or negative) and "
                  f"{m['cov']} as the referred fraction of the group, weight each referred patient by one over it, and "
                  "compute the weighted sensitivity, specificity and ROC area (a weighted Mann-Whitney statistic) among "
                  "the referred.\n\n")
        return head + q + method + ('Return your answer as JSON: {"answer": {"sensitivity": <proportion>, '
                                    '"specificity": <proportion>, "auc": <area>}}')

    def demo(self, cfg) -> list[Cell]:
        task = cfg["task"]
        key = re.search(r"Columns: patient_id, site, (\w+) \(the", task).group(1)
        t = TESTS[key]
        cov, thr = t["cov"], t["thr"]
        alt = re.search(r"site B in ([^;]+);", task)
        convert = (f'df["value"] = np.where(df.unit == "{alt.group(1)}", df.{key}'
                   + (f" * {t['alt'][1]:g}" if t["alt"][1] != 1 else "") + (f" / {t['alt'][2]:g}" if t["alt"][2] != 1 else "")
                   + f', df.{key})   # all in {t["unit"]}\n' if alt else f'df["value"] = df.{key}   # one unit throughout\n')
        inspect = (f'import numpy as np, pandas as pd\ndf = pd.read_csv("data_files/{SCREENING}")\n'
                   'print(df.head().to_string(index=False))\n'
                   'print(len(df), "patients | patients by site and unit:", df.groupby(["site", "unit"]).size().to_dict())\n'
                   f'print("median result by unit:", df.groupby("unit").{key}.median().to_dict())\n'
                   + convert +
                   f'df["positive"] = (df.value >= {thr:g}).astype(int)\n'
                   f'tab = df.groupby(["positive", "{cov}"]).agg(patients=("referred", "size"), referred=("referred", "mean"),\n'
                   '    diseased_among_referred=("reference", lambda s: s.eq("positive").sum() / max(s.notna().sum(), 1)))\n'
                   'print(tab.round(3).reset_index().to_string(index=False))')

        def prose2(outs):
            o = outs[0]
            rows = {(int(a), int(b)): (float(c), float(d)) for a, b, c, d in
                    re.findall(r"^\s*(\d)\s+(\d)\s+\d+\s+([\d.]+)\s+([\d.]+)\s*$", o, re.M)}
            meds = re.search(r"median result by unit: \{(.*)\}", o)
            txt = ""
            if alt and meds:
                txt += (f"Site B reports in {alt.group(1)} (medians by unit: {meds.group(1)}), so I converted every "
                        f"result to {t['unit']} ({t['conv']}) before applying the {thr:g} {t['unit']} threshold. ")
            if len(rows) == 4:
                txt += (f"Referral depended strongly on the result and on {cov}: {rows[(1, 1)][0]:.0%} of positives with "
                        f"{cov} = 1 and {rows[(1, 0)][0]:.0%} with {cov} = 0 were referred, against {rows[(0, 1)][0]:.0%} "
                        f"and only {rows[(0, 0)][0]:.0%} of negatives. Among referred negatives, disease was found in "
                        f"{rows[(0, 1)][1]:.1%} of those with {cov} = 1 and {rows[(0, 0)][1]:.1%} of those without, so "
                        f"{cov} predicts disease too. ")
            else:
                txt += f"Referral depended on the result and on {cov}. "
            return txt + ("The referred patients are therefore not representative: accuracy among them alone overstates "
                          "sensitivity and understates specificity, and counting the unreferred as disease-free is wrong "
                          f"too. Because referral depended only on the result and {cov}, each referred patient stands for "
                          "the unreferred patients of its group: I weight it by 1 / (referred fraction of its result-by-"
                          f"{cov} group), which is Begg and Greenes' correction, and compute the weighted sensitivity, "
                          "specificity and ROC area of the continuous result. Weighting by the result alone would ignore "
                          f"that {cov} also drove referral; both shortcuts are printed for comparison.")
        fit_code = (f'T, X = df.positive.to_numpy(), df.{cov}.to_numpy()\n'
                    'R, D = df.referred.to_numpy(), df.reference.eq("positive").to_numpy(float)\n'
                    'r = R == 1\n'
                    'print("referred only: sensitivity %.3f, specificity %.3f" % (T[r & (D == 1)].mean(), 1 - T[r & (D == 0)].mean()))\n'
                    'print("weights by test result only: sensitivity %.3f, specificity %.3f" % corrected_accuracy(T, 0 * X, D, R, df.value)[:2])\n'
                    'sens, spec, auc = corrected_accuracy(T, X, D, R, df.value)\n'
                    f'print("weights by result and {cov}: sensitivity %.4f, specificity %.4f, AUC %.4f (at the threshold '
                    'alone: %.4f)" % (sens, spec, auc, (sens + spec) / 2))\n'
                    'print("FINAL sensitivity = %.6f specificity = %.6f auc = %.6f" % (sens, spec, auc))')
        return [Cell("I load the cohort, check the units each laboratory reports in, put every result on the threshold's "
                     f"unit, and tabulate referral and the disease found by test result and {cov}.", inspect),
                Cell(prose2, fit_code, uses=(corrected_accuracy,), preamble="from sklearn.metrics import roc_auc_score")]


# ----------------------------------------------------------------------------- real data: SPECT thallium
PVB_PAGE = "https://cran.r-project.org/package=PVBcorrect"
PVB_URLS = ("https://cran.r-project.org/src/contrib/PVBcorrect_0.3.1.tar.gz",
            "https://cran.r-project.org/src/contrib/Archive/PVBcorrect/PVBcorrect_0.3.1.tar.gz")
# (age 60 or older, SPECT positive, CAD at angiography: 1 / 0 / None = not verified) -> patients, as in cad_pvb
SPECT_CELLS = {(0, 0, 1): 1, (0, 0, 0): 21, (0, 0, None): 783, (0, 1, 1): 67, (0, 1, 0): 133, (0, 1, None): 547,
               (1, 0, 1): 4, (1, 0, 0): 18, (1, 0, None): 438, (1, 1, 1): 128, (1, 1, 0): 99, (1, 1, None): 449}
SPECT_PUBLISHED = {"sensitivity": 0.8400495, "specificity": 0.5912022}


def read_rda(raw: bytes) -> dict:
    """The objects of an R .rda file (XDR serialisation, format 2 or 3) holding plain vectors, lists and data
    frames: {name: value}, a data frame as {column: list}; NA integers become None. Enough for CRAN data sets."""
    for unpack in (bz2.decompress, gzip.decompress, lzma.decompress):
        try:
            raw = unpack(raw)
            break
        except Exception:                                              # noqa: BLE001 -- try the next codec
            continue
    if raw[:5] not in (b"RDX2\n", b"RDX3\n") or raw[5:7] != b"X\n":
        raise ValueError("not an XDR RData file")
    pos, refs = 7, []

    def i32():
        nonlocal pos
        pos += 4
        return struct.unpack(">i", raw[pos - 4:pos])[0]

    def length():
        n = i32()
        return (i32() << 32) + i32() if n == -1 else n

    def item():
        nonlocal pos
        flags = i32()
        t, has_attr, has_tag = flags & 0xFF, flags & (1 << 9), flags & (1 << 10)
        if t in (254, 253, 252, 251, 247, 242):                       # NULL, environments, markers
            return None
        if t == 255:                                                   # a reference to an earlier symbol
            return refs[(flags >> 8 or i32()) - 1]
        if t == 1:                                                     # a symbol
            refs.append(item())
            return refs[-1]
        if t in (2, 6, 17):                                            # a pairlist -> {tag: value}
            out = {}
            while t in (2, 6, 17):
                if has_attr:
                    item()
                tag = item() if has_tag else None
                out[tag] = item()
                flags = i32()
                t, has_attr, has_tag = flags & 0xFF, flags & (1 << 9), flags & (1 << 10)
            if t != 254:
                raise ValueError(f"unsupported pairlist tail (type {t})")
            return out
        if t == 9:                                                     # a string
            n = i32()
            pos += max(n, 0)
            return None if n == -1 else raw[pos - n:pos].decode("utf-8", "replace")
        if t in (10, 13):                                              # logical / integer
            n = length()
            val = [None if x == -2 ** 31 else x for x in struct.unpack(f">{n}i", raw[pos:pos + 4 * n])]
            pos += 4 * n
        elif t == 14:                                                  # double
            n = length()
            val = list(struct.unpack(f">{n}d", raw[pos:pos + 8 * n]))
            pos += 8 * n
        elif t in (16, 19, 20):                                        # character vector / list
            val = [item() for _ in range(length())]
        else:
            raise ValueError(f"unsupported R object type {t}")
        attrs = item() if has_attr else {}
        return dict(zip(attrs["names"], val)) if t == 19 and attrs and attrs.get("names") else val

    version = i32()
    i32(), i32()
    if version == 3:
        pos += i32()                                                   # native encoding
    return item()


class SpectSource(RealSource):
    """SPECT thallium-201 imaging for coronary artery disease (CAD: at least 50% stenosis at coronary angiography):
    2,688 patients tested, 471 verified by angiography (Cecil et al. 1996, J Clin Epidemiol 49:735; the data of
    Kosinski & Barnhart 2003, Biometrics 59:163), as CRAN's PVBcorrect 0.3.1 ships it (`cad_pvb`: SPECT result,
    angiography result or NA, sex, stress mode, age 60 or older). Published answer: Arifin & Yusof (2022),
    "Correcting for partial verification bias in diagnostic accuracy studies: a tutorial using R", Statistics in
    Medicine 41(9):1709-1727, Table 2, the extended Begg and Greenes estimate with age group as the covariate
    (saturated logistic model, `acc_ebg(cad_pvb, covariate = "X3", saturated_model = TRUE)`): sensitivity
    0.8400495, specificity 0.5912022. The same table's complete-case estimate (0.975, 0.144) is this family's
    `referred_only` trap, and its estimate without the covariate (0.819, 0.592) is `referral_by_test_only`.

    The question is the family's, on a binary index test: the file carries the SPECT result, the age group
    (the covariate the published analysis conditions on), referral and the angiography result; sex and stress
    mode are left out, because conditioning on them is a different (unpublished) analysis. With these cells the
    family's inverse-probability weights reproduce the published values to the printed digits. Graded on
    sensitivity and specificity only (partial_keys: an ROC area of a binary test is not asked).

      venv/bin/python -m sage real --source spect --families hc_dx --out ROOT [--raw DIR]
    """
    label = "hc_dx_real_spect"
    keep_controls = False
    partial_keys = True

    def __init__(self, fam: str = "hc_dx", raw: Path | None = None, limit: int = 0):
        from sage.registry import get
        self.family = get(fam)
        if raw is None:
            from paths import data_root
            raw = data_root() / "raw" / "spect_pvb"
        self.raw = Path(raw)
        self.note = ("ground truth is the published estimate (Arifin & Yusof 2022, extended Begg and Greenes with the age "
                     "group), not an analysis output; the family's reference analysis reproduces it. Ablations are "
                     "scored against the published value: a trap inside tolerance on these data has fails_at_step None")

    def keys(self) -> list:
        return ["spect"]

    def fetch(self) -> Path:
        """cad_pvb from the PVBcorrect source package on CRAN (plain HTTPS, once), checked cell by cell against
        SPECT_CELLS and converted to the family's file format. Without network access the file is rebuilt from
        SPECT_CELLS, which is the whole data set (five binary variables; rows are exchangeable)."""
        import collections
        import io
        import tarfile
        import urllib.request
        out = self.raw / SCREENING
        if out.exists():
            return out
        self.raw.mkdir(parents=True, exist_ok=True)
        cells, src = None, "SPECT_CELLS (offline)"
        for url in PVB_URLS:
            try:
                blob = urllib.request.urlopen(url, timeout=60).read()
                with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
                    frame = read_rda(tar.extractfile("PVBcorrect/data/cad_pvb.rda").read())["cad_pvb"]
                rows = list(zip(frame["X3"], frame["T"], frame["D"]))
                cells, src = collections.Counter(rows), url
                break
            except Exception:                                          # noqa: BLE001 -- next mirror, then offline
                continue
        if cells is not None and dict(cells) != SPECT_CELLS:
            raise ValueError(f"cad_pvb from {src} does not match the published cell counts")
        rows = rows if cells is not None else [c for c, k in SPECT_CELLS.items() for _ in range(k)]
        age, test, cad = (np.array(x, dtype=object) for x in zip(*rows))
        df = pd.DataFrame({"patient_id": [f"S{i + 1:04d}" for i in range(len(rows))],
                           "spect": np.where(test == 1, "positive", "negative"), "age_60_plus": age.astype(int),
                           "referred": (cad != None).astype(int),                                 # noqa: E711
                           "reference": pd.Series(np.where(cad == 1, "positive", "negative")).where(cad != None)})  # noqa: E711
        df.to_csv(out, index=False)
        (self.raw / "SOURCE.txt").write_text(f"cad_pvb (PVBcorrect 0.3.1) from {src}\n")
        return out

    def load(self, key) -> RealTarget:
        csv = self.fetch()
        df = pd.read_csv(csv)
        meta = {"test": None, "col": "spect", "cov": "age_60_plus", "thr": NAN, "unit": None, "mixed": False,
                "alt_unit": None, "mult": 1.0, "div": 1.0, "n": len(df), "binary_index": True}
        prov = {"benchmark": "SPECT thallium work-up bias data (Cecil et al. 1996; Kosinski & Barnhart 2003)",
                "source_url": PVB_PAGE, "data_set": "PVBcorrect::cad_pvb (CRAN, version 0.3.1)",
                "fetched_from": (self.raw / "SOURCE.txt").read_text().strip() if (self.raw / "SOURCE.txt").exists() else None,
                "published": "Arifin & Yusof (2022) Stat Med 41(9):1709-1727, Table 2, EBGX (extended Begg and Greenes, "
                             "covariate X3 = age 60 or older, saturated model): Sn 0.8400495, Sp 0.5912022",
                "published_answer": SPECT_PUBLISHED}
        return RealTarget("hc_dx_real_spect", {"df": df, "meta": meta}, dict(SPECT_PUBLISHED), prov, {SCREENING: csv})

    def task_text(self, target: RealTarget) -> str:
        n, nr = len(target.data["df"]), int(target.data["df"]["referred"].sum())
        return ("You are a clinical epidemiologist assessing single-photon-emission computed tomography (SPECT) thallium-201 "
                "imaging as a test for coronary artery disease (CAD: at least 50% stenosis of a coronary artery at "
                f"coronary angiography). All {n} patients in the study had a SPECT thallium test, read as positive or "
                f"negative; only {nr} went on to coronary angiography, the reference standard. Assume that, among "
                "patients with the same SPECT result and the same age group, whether a patient went on to angiography "
                "did not depend on whether they had CAD.\n\n"
                "Data files:\n"
                f"- data_files/{SCREENING}: one row per patient. Columns: patient_id, spect (the SPECT thallium result, "
                "positive or negative), age_60_plus (1 = aged 60 or older, 0 = younger), referred (1 = underwent "
                "coronary angiography), reference (the angiography result, positive or negative for CAD; empty if not "
                "referred).\n\n"
                "Question: how accurate is SPECT thallium for CAD in this population? Report its sensitivity and "
                "specificity for the whole study population, as if every patient had undergone angiography.\n\n"
                "Approach: patients who did not undergo angiography are missing a reference result, not negative, and "
                "those who did are not representative. Correct for this partial verification by inverse-probability "
                "weighting (Begg and Greenes' method with age_60_plus): estimate the probability of angiography in each "
                "of the four groups of SPECT result and age group as the verified fraction of the group, weight each "
                "verified patient by one over it, and compute the weighted sensitivity and specificity among the "
                "verified.\n\n"
                'Return your answer as JSON: {"answer": {"sensitivity": <proportion>, "specificity": <proportion>}}')


FAMILIES = [HospitalQuality(), DiagnosticAccuracy()]
REAL_SOURCES = {"spect": SpectSource}
