"""Survival analysis: a treatment's effect on time to death in an observational cohort (a SAGE plug-in).

The data are what a biostatistician receives from a registry or an electronic-health-record
extract: one row per patient, with baseline covariates, the treatment, follow-up time and vital
status at the data cut-off. The DGP builds in the irregularities such data carry:

  * right censoring: administrative (staggered enrolment, a fixed cut-off) and loss to follow-up;
  * confounding: older and frailer patients are less likely to get the new treatment and die
    sooner, so the crude comparison flatters the treatment;
  * `surv_cox`: hospitals differ in baseline risk and in how readily they prescribe, and the new
    therapy's uptake grows over calendar time, so treated patients could be followed for less time
    before the cut-off (the reason an analysis of WHETHER death occurred is biased); higher-risk
    hospitals adopt it less, so every confounder flatters the new treatment in a crude comparison;
    follow-up in whole months, so many death times are tied;
  * `surv_tvc`: the drug is started some time after diagnosis, so a patient must survive long
    enough to be treated (immortal time); dates, not durations, are given.

Reference analyses (gold on the sampled data): Cox proportional hazards (statsmodels PHReg,
Efron's method for ties) adjusted for the confounders -- stratified by hospital in `surv_cox`,
with treatment as a time-varying covariate (counting-process rows, `entry`) in `surv_tvc` -- and
the Kaplan-Meier median (statsmodels SurvfuncRight). Tolerances: 2 SE of the log hazard ratio on
the ratio scale (floor 3%); for the median, the half-width of its 95% confidence interval
(Brookmeyer-Crowley), at least one month (`surv_cox`) or 15 days (`surv_tvc`).

Families, answer keys and traps (the step each skips; skills from the shared taxonomy):

  surv_cox  multi-hospital cohort, a new therapy versus standard; keys hr (new vs standard, same
            age, performance status and hospital) and median_control (KM median survival, months,
            of the standard-therapy patients)
    1 qc            censored_as_deaths        every follow-up time taken as a death
    1 qc            deaths_only               censored patients dropped
    2 model-choice  logistic_death_status     logistic regression of died on treatment + covariates
                                              (+ hospital): an odds ratio that ignores follow-up time
    3 model-choice  no_confounder_adjustment  Cox with treatment only (stratified by hospital)
    4 normalise     ignore_hospital           Cox adjusted for age and ECOG, one baseline hazard
    -  variants     hospital_indicators       hospital as indicator covariates instead of strata
                    breslow_ties              Breslow's approximation for tied death times
  surv_tvc  chronic-disease cohort from diagnosis, a drug started later by some; keys hr (on the drug
            vs not yet or never on it, same age and comorbidity) and median_survival (KM median,
            days from diagnosis, whole cohort)
    1 qc            censored_as_deaths
    2 model-choice  ever_treated_from_diagnosis  "ever treated" as a baseline covariate: the
                                              time before the drug is credited to it (immortal time)
    3 model-choice  clock_from_treatment_start  treated patients followed from their start date,
                                              untreated from diagnosis (misaligned time zero; cond:
                                              its bias depends on the waiting times, live on most draws)
    4 model-choice  no_confounder_adjustment  time-varying Cox with treatment only
    -  variants     breslow_ties
                    risk_set_includes_entry_day  statsmodels' own convention, which counts a row at
                                              risk at its entry time (the reference uses R's (start, stop]:
                                              otherwise a patient who starts the drug on a day someone dies is
                                              counted twice that day; the two differ by 0.1-1.2% of the HR)

The analysis functions are self-contained (numpy/pandas/statsmodels), so a demonstration embeds
their source verbatim: the trace and the ground truth are the same code.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight

from sage.core import Cell, Family, Sample, Trap

COHORT, PATIENTS = "cohort.csv", "patients.csv"
NAN = float("nan")
MONTH_DAYS = 30.4375


# ----------------------------------------------------------------------------- analysis
def cox_hr(df, time="months", covariates=("age", "ecog"), strata="hospital", ties="efron", status="died",
           entry=None, exposure="treated", open_entry=True):
    """Cox proportional-hazards fit (statsmodels PHReg): hazard ratio of `exposure` adjusted for
    `covariates`, a baseline hazard per `strata` level (None: one), `entry` for counting-process
    (start, stop] rows. Returns (hazard ratio, SE of its log)."""
    start = None if entry is None else df[entry].to_numpy(float)
    if start is not None and open_entry:
        start = start + 1e-6   # statsmodels counts a row at risk AT its entry time; (start, stop] as in R's survival
    fit = PHReg(df[time].to_numpy(float), df[[exposure, *covariates]].to_numpy(float),
                status=df[status].to_numpy(float), entry=start,
                strata=None if strata is None else df[strata].to_numpy(), ties=ties).fit()
    return float(np.exp(fit.params[0])), float(fit.bse[0])


def km_median(time, died):
    """Kaplan-Meier median (the first time the curve drops below 0.5) and its 95% CI."""
    sf = SurvfuncRight(np.asarray(time, float), np.asarray(died, float))
    lo, hi = sf.quantile_ci(0.5)
    return float(sf.quantile(0.5)), float(lo), float(hi)


def split_at_treatment(df):
    """Counting-process rows for a drug started after diagnosis: each patient is untreated from
    diagnosis (day 0) to the start date, and treated from then to the end of follow-up."""
    d0 = pd.to_datetime(df["diagnosis_date"])
    stop = (pd.to_datetime(df["end_date"]) - d0).dt.days.to_numpy(float)
    start = (pd.to_datetime(df["treatment_start_date"]) - d0).dt.days.to_numpy(float)   # NaN: never started
    on = np.isfinite(start)
    cols = {c: df[c].to_numpy() for c in ("patient", "age", "comorbidity")}
    pre = pd.DataFrame({**cols, "start": 0.0, "stop": np.where(on, start, stop), "treated": 0,
                        "died": np.where(on, 0, df["died"])})
    post = pd.DataFrame({**{c: v[on] for c, v in cols.items()}, "start": start[on], "stop": stop[on], "treated": 1,
                         "died": df["died"].to_numpy()[on]})
    return pd.concat([pre, post], ignore_index=True)


def _tol_median(med, lo, hi, floor):
    half = (hi - lo) / 2 if np.isfinite(hi) else med - lo
    return max(floor, float(half)) if np.isfinite(half) else NAN


# ----------------------------------------------------------------------------- shared
class SurvivalFamily(Family):
    domain = "survival"
    max_tries = 12
    rng_policy = "stream"
    demo_timeout = 240

    def _cached(self, data, key, fn):
        cache = data.setdefault("_cache", {})
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    def extra_provenance(self, inst) -> dict:
        return {"knobs": inst.knobs, "design": dict(inst.sample.data["meta"]),
                "reference_diagnostics": {k: v for k, v in inst.a.gold.items() if k.startswith("_")},
                "skills": {str(s): v for s, v in self.skills.items()}}

    def sanity(self, gold, tol, latent) -> list[str]:
        bad = [k for k in self.keys if not np.isfinite(gold[k]) or not np.isfinite(tol[k])]
        return [f"{k}_not_estimable" for k in bad]


def _dates(start: dt.date, days) -> list[str]:
    return [(start + dt.timedelta(days=int(d))).isoformat() for d in days]


# ----------------------------------------------------------------------------- surv_cox
DISEASES = (("advanced non-small-cell lung cancer", "an immunotherapy combination", "platinum doublet chemotherapy"),
            ("metastatic colorectal cancer", "a targeted-therapy combination", "FOLFOX chemotherapy"),
            ("multiple myeloma", "a new triplet regimen", "the standard doublet regimen"),
            ("advanced hepatocellular carcinoma", "a new first-line combination", "sorafenib"),
            ("glioblastoma", "a new adjuvant regimen", "standard temozolomide"))


class CohortCox(SurvivalFamily):
    name = "surv_cox"
    keys = ("hr", "median_control")
    steps = {1: "handle censoring: a patient alive at last contact is censored, neither a death nor dropped",
             2: "model time to death (a survival model), not whether death occurred",
             3: "adjust for the confounders (age, performance status)",
             4: "allow each hospital its own baseline hazard (stratify), or hospitals confound the comparison"}
    skills = {1: "qc", 2: "model-choice", 3: "model-choice", 4: "normalise"}
    seed_blocks = {"train": 16_000_000, "test": 16_050_000}
    tol_rule = ("hr: max(3% of HR, 2 HR SE(log HR)); median_control: max(1 month, half-width of the KM median's 95% "
                "CI (Brookmeyer-Crowley, cloglog))")
    note = ("ground truth is the reference analysis (Cox proportional hazards of follow-up on treatment, age and ECOG, "
            "stratified by hospital, Efron ties; Kaplan-Meier median of the standard-therapy arm) on the sampled "
            "cohort, not the latent parameters; deviation is in units of tolerance, maxed over the answer keys")

    def knobs(self, rng):
        return {"n": 1.0, "conf": 1.0, "site": 1.0, "uptake": 1.0, "cens": 1.0}

    def draw(self, rng, k) -> Sample:
        dis = DISEASES[int(rng.integers(len(DISEASES)))]
        G = int(rng.integers(3, 7))
        n = int(round(rng.integers(900, 3001) * k["n"]))
        shape, med0 = rng.uniform(0.8, 1.5), float(np.exp(rng.uniform(np.log(6), np.log(48))))
        accrual = min(med0 * rng.uniform(1.5, 3.5), 120.0)                 # months; follow-up scaled to the disease
        extra = med0 * rng.uniform(0.4, 1.2) / k["cens"]
        entry = rng.uniform(0, accrual, n)
        age = np.clip(np.round(rng.normal(rng.uniform(58, 70), rng.uniform(8, 11), n)), 25, 92)
        p0, p2 = rng.uniform(0.3, 0.5), rng.uniform(0.1, 0.25)
        ecog = rng.choice([0, 1, 2], n, p=[p0, 1 - p0 - p2, p2])
        z, e = (age - 64) / 10, ecog - ecog.mean()
        sd_site = rng.uniform(0.35, 0.65) * k["site"]
        site_eff = sd_site * rng.permutation(norm.ppf((np.arange(G) + 0.5) / G)) + rng.normal(0, 0.1, G)
        site_eff -= site_eff.mean()
        site = rng.integers(0, G, n)
        kappa = -rng.uniform(0.8, 1.5) * k["site"]                        # higher-risk hospitals adopt it less
        prop_site = kappa * site_eff / sd_site + rng.normal(0, 0.3, G)
        g_age, g_ecog = rng.uniform(0.8, 1.5) * k["conf"], rng.uniform(0.8, 1.5) * k["conf"]
        uptake = rng.uniform(0.8, 2.0) * k["uptake"]
        logit = rng.uniform(-0.4, 0.4) + uptake * (2 * entry / accrual - 1) - g_age * z - g_ecog * e + prop_site[site]
        A = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        hr = float(np.exp(rng.uniform(np.log(0.4), np.log(1.8))))
        d_age, d_ecog = rng.uniform(0.4, 0.8), rng.uniform(0.4, 0.8)
        lam = np.log(2) / med0 ** shape
        lp = np.log(hr) * A + d_age * z + d_ecog * e + site_eff[site]
        T = (rng.exponential(1, n) / (lam * np.exp(lp))) ** (1 / shape)
        C = np.minimum(accrual + extra - entry, rng.exponential(med0 * rng.uniform(4, 15), n))
        months = np.maximum(np.ceil(np.minimum(T, C)), 1).astype(int)
        died = (T <= C).astype(int)
        ctl = A == 0
        t_med = brentq(lambda s: np.mean(np.exp(-lam * s ** shape * np.exp(lp[ctl]))) - 0.5, 1e-6, 1e5)
        start = dt.date(int(rng.integers(2006, 2016)), 1, 1) + dt.timedelta(days=int(rng.integers(0, 365)))
        hosp = [f"H{j + 1}" for j in range(G)]
        order = np.argsort(entry, kind="stable")
        df = pd.DataFrame({"patient": [f"P{i + 1:05d}" for i in range(n)],
                           "hospital": np.array(hosp)[site[order]],
                           "enrolled": _dates(start, entry[order] * MONTH_DAYS),
                           "age": age[order].astype(int), "ecog": ecog[order], "treated": A[order],
                           "months": months[order], "died": died[order]})
        cutoff = (start + dt.timedelta(days=int((accrual + extra) * MONTH_DAYS))).isoformat()
        meta = {"disease": dis[0], "new": dis[1], "standard": dis[2], "hospitals": G, "n": n, "cutoff": cutoff,
                "first_enrolled": df["enrolled"].iloc[0], "last_enrolled": df["enrolled"].iloc[-1]}
        latent = {"hr": hr, "median_control": float(np.ceil(t_med)), "median_control_continuous": float(t_med),
                  "weibull_shape": float(shape), "log_hr_age_per_10y": float(d_age), "log_hr_ecog": float(d_ecog),
                  "hospital_log_hr": [float(x) for x in site_eff], "uptake_trend": float(uptake),
                  "treated_fraction": float(A.mean()), "death_fraction": float(died.mean())}
        return Sample({"df": df, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / COHORT, index=False)
        return [COHORT]

    def _ref(self, data):
        def run():
            df = data["df"]
            hr, se = cox_hr(df)
            ctl = df[df["treated"] == 0]
            med, lo, hi = km_median(ctl["months"], ctl["died"])
            return {"hr": hr, "median_control": med, "_se_log_hr": se, "_median_ci": [lo, hi]}
        return self._cached(data, ("_ref",), run)

    def reference(self, data) -> dict:
        return dict(self._ref(data))

    def tolerance(self, gold) -> dict:
        lo, hi = gold["_median_ci"]
        return {"hr": max(0.03 * gold["hr"], 2 * gold["hr"] * gold["_se_log_hr"]),
                "median_control": _tol_median(gold["median_control"], lo, hi, 1.0)}

    def traps(self, data) -> dict:
        import statsmodels.api as sm
        df, ref = data["df"], self._ref(data)
        med = ref["median_control"]
        allde = df.assign(died=1)
        dead = df[df["died"] == 1]
        ctl = lambda d: d[d["treated"] == 0]
        X = pd.concat([df[["treated", "age", "ecog"]], pd.get_dummies(df["hospital"], prefix="h", drop_first=True)], axis=1)
        odds_ratio = float(np.exp(sm.Logit(df["died"], sm.add_constant(X.astype(float))).fit(disp=0).params["treated"]))
        dummies = pd.concat([df, pd.get_dummies(df["hospital"], prefix="h", drop_first=True).astype(float)], axis=1)
        hcols = [c for c in dummies.columns if c.startswith("h_")]
        return {"censored_as_deaths": Trap({"hr": cox_hr(allde)[0], "median_control": km_median(ctl(allde)["months"], ctl(allde)["died"])[0]}, 1),
                "deaths_only": Trap({"hr": cox_hr(dead)[0], "median_control": km_median(ctl(dead)["months"], ctl(dead)["died"])[0]}, 1),
                "logistic_death_status": Trap({"hr": odds_ratio, "median_control": med}, 2),
                "no_confounder_adjustment": Trap({"hr": cox_hr(df, covariates=())[0], "median_control": med}, 3),
                "ignore_hospital": Trap({"hr": cox_hr(df, strata=None)[0], "median_control": med}, 4),
                "hospital_indicators": Trap({"hr": cox_hr(dummies, covariates=("age", "ecog", *hcols), strata=None)[0],
                                             "median_control": med}, None),
                "breslow_ties": Trap({"hr": cox_hr(df, ties="breslow")[0], "median_control": med}, None)}

    def strengthen(self, k, dead):
        k = dict(k)
        if {"censored_as_deaths", "deaths_only"} & set(dead):
            k["cens"] = min(k["cens"] * 1.4, 4.0)
        if "logistic_death_status" in dead:
            k["uptake"] = min(k["uptake"] * 1.4, 3.0)
        if "no_confounder_adjustment" in dead:
            k["conf"] = min(k["conf"] * 1.4, 3.0)
        if "ignore_hospital" in dead:
            k["site"] = min(k["site"] * 1.4, 3.0)
        return k

    def shortcuts(self, data) -> dict:
        """Data-light answers: the crude death-rate ratio (deaths per person-month, no adjustment) with
        the naive median of the standard arm's death times, and with the reference median."""
        df = data["df"]
        rate = lambda d: d["died"].sum() / d["months"].sum()
        crude = float(rate(df[df.treated == 1]) / rate(df[df.treated == 0]))
        naive_med = float(np.median(df[(df.treated == 0) & (df.died == 1)]["months"]))
        return {"crude_rate_ratio_naive_median": {"hr": crude, "median_control": naive_med},
                "crude_rate_ratio_true_median": {"hr": crude, "median_control": self._ref(data)["median_control"]}}

    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        head = (f"You are a biostatistician. A registry followed {m['n']} patients with {m['disease']} treated at "
                f"{m['hospitals']} hospitals, enrolled at diagnosis between {m['first_enrolled']} and {m['last_enrolled']}. "
                f"First-line treatment was either {m['new']} (new) or {m['standard']} (standard), chosen by the treating "
                f"physicians. The data were cut off on {m['cutoff']}; a patient alive at the cut-off or lost to "
                "follow-up is censored at the last contact.\n\n"
                "Data files:\n"
                f"- data_files/{COHORT}: one row per patient. Columns: patient, hospital, enrolled (date of enrolment), "
                "age (years at diagnosis), ecog (ECOG performance status 0-2; higher is frailer), treated (1 = new, "
                "0 = standard), months (follow-up from enrolment to death or last contact, in whole months, rounded up), "
                "died (1 = died, 0 = censored).\n\n")
        q = ("Question: what is the effect of the new treatment on survival? Report hr, the hazard ratio of death for "
             "the new versus the standard treatment among patients of the same age, performance status and hospital, "
             "and median_control, the median survival (months) of the patients given the standard treatment.\n\n")
        method = ("" if blind else
                  "Approach: follow-up differs between patients and many are censored, so use survival methods on the "
                  "time to death, not whether death occurred. Fit a Cox proportional-hazards model of months (event: "
                  "died) on treated, age and ecog, stratified by hospital (a baseline hazard per hospital), with Efron's "
                  "method for the tied death times; hr = exp(coefficient of treated). median_control is the "
                  "Kaplan-Meier median survival of the standard-treatment patients.\n\n")
        return head + q + method + 'Return your answer as JSON: {"answer": {"hr": <ratio>, "median_control": <months>}}'

    def demo(self, cfg) -> list[Cell]:
        cutoff = re.search(r"cut off on (\d{4}-\d\d-\d\d)", cfg["task"]).group(1)
        inspect = (f'import numpy as np, pandas as pd\ndf = pd.read_csv("data_files/{COHORT}")\n'
                   'print(df.head().to_string(index=False))\n'
                   'print(len(df), "patients,", df.died.sum(), "deaths,", (df.died == 0).sum(), "censored;",\n'
                   '      df[df.died == 1].months.nunique(), "distinct death months (ties)")\n'
                   f'df["to_cutoff"] = (pd.Timestamp("{cutoff}") - pd.to_datetime(df.enrolled)).dt.days / 30.44   '
                   '# follow-up possible\n'
                   'g = df.groupby("treated")\n'
                   'for col in ("age", "ecog", "to_cutoff", "died"):\n'
                   '    print(f"mean {col}: standard {g[col].mean()[0]:.2f}, new {g[col].mean()[1]:.2f}")\n'
                   'print("share treated by enrolment year:", df.groupby(df.enrolled.str[:4]).treated.mean().round(2).to_dict())\n'
                   'h = df.groupby("hospital").agg(n=("died", "size"), treated=("treated", "mean"), died=("died", "mean"))\n'
                   'print(h.round(2).to_string())')

        def prose2(outs):
            o = outs[0]
            num = lambda pat: [float(x) for x in re.search(pat, o).groups()] if re.search(pat, o) else None
            dm = num(r"(\d+) deaths, (\d+) censored; (\d+) distinct death months")
            age, ecog, pot = (num(rf"mean {c}: standard ([\d.]+), new ([\d.]+)") for c in ("age", "ecog", "to_cutoff"))
            yr = re.search(r"share treated by enrolment year: \{(.*)\}", o)
            yrs = re.findall(r"'(\d{4})': ([\d.]+)", yr.group(1)) if yr else []
            hosp = [(float(a), float(b)) for a, b in re.findall(r"^H\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s*$", o, re.M)]
            txt = (f"{dm[1]:.0f} patients are censored and the {dm[0]:.0f} deaths fall on only {dm[2]:.0f} distinct "
                   "months, so there are many tied death times. " if dm else "Many patients are censored and death "
                   "times are tied. ")
            if len(yrs) >= 2 and pot:
                txt += (f"The new treatment's share rose from {float(yrs[0][1]):.0%} of patients enrolled in {yrs[0][0]} "
                        f"to {float(yrs[-1][1]):.0%} in {yrs[-1][0]}, so its patients could be followed for less time "
                        f"({pot[1]:.1f} months from enrolment to the cut-off on average, against {pot[0]:.1f}). ")
            txt += ("Whether a patient is seen to die therefore depends on when they enrolled: a logistic regression of "
                    "death status ignores that, and treating censored times as deaths or dropping censored patients "
                    "distorts it further. ")
            if age and ecog:
                txt += (f"The treated are also younger ({age[1]:.1f} against {age[0]:.1f} years) and fitter (mean ECOG "
                        f"{ecog[1]:.2f} against {ecog[0]:.2f}), which by itself makes the new treatment look better. ")
            if hosp:
                txt += (f"Hospitals differ in how often they use it ({min(h[0] for h in hosp):.0%} to "
                        f"{max(h[0] for h in hosp):.0%}) and in how many of their patients died "
                        f"({min(h[1] for h in hosp):.0%} to {max(h[1] for h in hosp):.0%}). ")
            return txt + ("So I fit a Cox model of time to death on treatment, age and ECOG, stratified by hospital (each "
                          "hospital its own baseline hazard), with Efron's method for the tied months; the unadjusted "
                          "model is shown for comparison. The standard arm's median survival is read off its "
                          "Kaplan-Meier curve.")
        fit_code = ('hr, se = cox_hr(df)\n'
                    'print("adjusted, stratified by hospital: HR %.3f (95%% CI %.3f-%.3f)" % (hr, hr * np.exp(-1.96 * se), '
                    'hr * np.exp(1.96 * se)))\n'
                    'print("unadjusted (treatment only): HR %.3f" % cox_hr(df, covariates=())[0])\n'
                    'std = df[df.treated == 0]\n'
                    'med, lo, hi = km_median(std.months, std.died)\n'
                    'print(f"standard arm: Kaplan-Meier median {med:.0f} months (95% CI {lo:.0f}-{hi:.0f})")\n'
                    'print("FINAL hr = %.6f median_control = %.4f" % (hr, med))')
        return [Cell("I load the cohort and check censoring, tied death times, how the two treatment groups differ "
                     "(age, ECOG, the follow-up each could have had before the cut-off), and how treatment use varies "
                     "over time and between hospitals.", inspect),
                Cell(prose2, fit_code, uses=(cox_hr, km_median),
                     preamble="from statsmodels.duration.hazard_regression import PHReg\n"
                              "from statsmodels.duration.survfunc import SurvfuncRight")]


# ----------------------------------------------------------------------------- surv_tvc
DRUGS = (("heart failure with reduced ejection fraction", "an SGLT2 inhibitor"),
         ("chronic kidney disease stage 4", "a new potassium binder"),
         ("idiopathic pulmonary fibrosis", "an antifibrotic drug"),
         ("atrial fibrillation with heart failure", "a new anticoagulant"))


def _weibull_tv(rng, lam, shape, lp0, beta, w):
    """Death times with hazard lam*shape*t^(shape-1)*exp(lp0 + beta*[t >= w]) (w = inf: never treated)."""
    E = rng.exponential(1, len(lp0))
    H0w = lam * np.where(np.isfinite(w), w, 0) ** shape
    pre = np.isinf(w) | (E < H0w * np.exp(lp0))
    T = np.empty(len(lp0))
    T[pre] = (E[pre] / (lam * np.exp(lp0[pre]))) ** (1 / shape)
    post = ~pre
    T[post] = ((H0w[post] + (E[post] - H0w[post] * np.exp(lp0[post])) * np.exp(-lp0[post] - beta)) / lam) ** (1 / shape)
    return T


class ImmortalTime(SurvivalFamily):
    name = "surv_tvc"
    keys = ("hr", "median_survival")
    steps = {1: "handle censoring: a patient alive at last contact is censored, not a death",
             2: "treat the drug as time-varying: the time before a patient starts it is untreated time",
             3: "follow every patient from the same time zero (diagnosis)",
             4: "adjust for the confounders (age, comorbidity)"}
    skills = {1: "qc", 2: "model-choice", 3: "model-choice", 4: "model-choice"}
    seed_blocks = {"train": 16_100_000, "test": 16_150_000}
    tol_rule = ("hr: max(3% of HR, 2 HR SE(log HR)); median_survival: max(15 days, half-width of the KM median's 95% "
                "CI (Brookmeyer-Crowley, cloglog))")
    note = ("ground truth is the reference analysis (Cox proportional hazards with the drug as a time-varying covariate "
            "-- counting-process rows split at the start date -- adjusted for age and comorbidity, Efron ties; "
            "Kaplan-Meier median survival from diagnosis of the whole cohort) on the sampled cohort, not the latent "
            "parameters; deviation is in units of tolerance, maxed over the answer keys")

    def knobs(self, rng):
        return {"n": 1.0, "conf": 1.0, "wait": 1.0, "cens": 1.0}

    def draw(self, rng, k) -> Sample:
        dis = DRUGS[int(rng.integers(len(DRUGS)))]
        n = int(round(rng.integers(800, 2501) * k["n"]))
        shape, med0 = rng.uniform(0.7, 1.3), float(np.exp(rng.uniform(np.log(9), np.log(48))))
        accrual = min(med0 * rng.uniform(1.2, 3.0), 120.0)                 # months; follow-up scaled to the disease
        extra = med0 * rng.uniform(0.4, 1.2) / k["cens"]
        entry = rng.uniform(0, accrual, n)
        age = np.clip(np.round(rng.normal(rng.uniform(66, 76), rng.uniform(8, 11), n)), 35, 98)
        comorb = np.minimum(rng.poisson(rng.uniform(1.0, 2.0), n), 6)
        z, c = (age - 70) / 10, comorb - comorb.mean()
        g_age, g_com = rng.uniform(0.5, 1.0) * k["conf"], rng.uniform(0.3, 0.6) * k["conf"]
        eligible = rng.random(n) < rng.uniform(0.5, 0.85)
        wait_mean = rng.uniform(2, 8) * k["wait"]                                          # months
        W = rng.exponential(wait_mean * np.exp(g_age * z + g_com * c))
        W = np.where(eligible, np.maximum(W, 0.1), np.inf)
        hr = float(np.exp(rng.uniform(np.log(0.45), np.log(1.6))))
        d_age, d_com = rng.uniform(0.4, 0.8), rng.uniform(0.2, 0.4)
        lam = np.log(2) / med0 ** shape
        lp0 = d_age * z + d_com * c
        T = _weibull_tv(rng, lam, shape, lp0, np.log(hr), W)
        C = np.minimum(accrual + extra - entry, rng.exponential(med0 * rng.uniform(4, 15), n))
        stop_d = np.maximum(np.round(np.minimum(T, C) * MONTH_DAYS), 1)
        died = (T <= C).astype(int)
        w_d = np.round(W * MONTH_DAYS)
        started = np.isfinite(W) & (w_d >= 1) & (w_d < stop_d)
        start = dt.date(int(rng.integers(2010, 2019)), 1, 1) + dt.timedelta(days=int(rng.integers(0, 365)))
        order = np.argsort(entry, kind="stable")
        diag = np.round(entry * MONTH_DAYS)
        df = pd.DataFrame({"patient": [f"C{i + 1:05d}" for i in range(n)],
                           "diagnosis_date": _dates(start, diag[order]),
                           "age": age[order].astype(int), "comorbidity": comorb[order],
                           "treatment_start_date": [(start + dt.timedelta(days=int(diag[i] + w_d[i]))).isoformat()
                                                    if started[i] else "" for i in order],
                           "end_date": _dates(start, (diag + stop_d)[order]), "died": died[order]})
        df["treatment_start_date"] = df["treatment_start_date"].replace("", np.nan)
        cutoff = (start + dt.timedelta(days=int((accrual + extra) * MONTH_DAYS))).isoformat()
        meta = {"disease": dis[0], "drug": dis[1], "n": n, "cutoff": cutoff,
                "first_diagnosed": df["diagnosis_date"].iloc[0], "last_diagnosed": df["diagnosis_date"].iloc[-1]}
        latent = {"hr": hr, "median_survival": float(np.median(T) * MONTH_DAYS), "weibull_shape": float(shape),
                  "log_hr_age_per_10y": float(d_age), "log_hr_comorbidity": float(d_com),
                  "started_fraction": float(started.mean()), "death_fraction": float(died.mean()),
                  "mean_wait_months": float(wait_mean)}
        return Sample({"df": df, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / PATIENTS, index=False)
        return [PATIENTS]

    def _long(self, data):
        return self._cached(data, ("_long",), lambda: split_at_treatment(data["df"]))

    def _days(self, data):
        df = data["df"]
        d0 = pd.to_datetime(df["diagnosis_date"])
        return ((pd.to_datetime(df["end_date"]) - d0).dt.days.to_numpy(float),
                (pd.to_datetime(df["treatment_start_date"]) - d0).dt.days.to_numpy(float))

    def _ref(self, data):
        def run():
            hr, se = cox_hr(self._long(data), time="stop", entry="start", covariates=("age", "comorbidity"), strata=None)
            t, _ = self._days(data)
            med, lo, hi = km_median(t, data["df"]["died"])
            return {"hr": hr, "median_survival": med, "_se_log_hr": se, "_median_ci": [lo, hi]}
        return self._cached(data, ("_ref",), run)

    def reference(self, data) -> dict:
        return dict(self._ref(data))

    def tolerance(self, gold) -> dict:
        lo, hi = gold["_median_ci"]
        return {"hr": max(0.03 * gold["hr"], 2 * gold["hr"] * gold["_se_log_hr"]),
                "median_survival": _tol_median(gold["median_survival"], lo, hi, 15.0)}

    def traps(self, data) -> dict:
        df, ref = data["df"], self._ref(data)
        med = ref["median_survival"]
        long = self._long(data)
        t, w = self._days(data)
        cov = ("age", "comorbidity")
        tv = lambda d, **kw: cox_hr(d, time="stop", entry="start", strata=None, **{"covariates": cov, **kw})[0]
        base = df.assign(t=t, ever=np.isfinite(w).astype(int))
        base["t_from_start"] = np.where(np.isfinite(w), t - np.nan_to_num(w), t)
        last = long.groupby("patient")["stop"].transform("max") == long["stop"]
        return {"censored_as_deaths": Trap({"hr": tv(long.assign(died=last.astype(int))),
                                            "median_survival": km_median(t, np.ones_like(t))[0]}, 1),
                "ever_treated_from_diagnosis": Trap({"hr": cox_hr(base, time="t", covariates=cov, strata=None,
                                                                  exposure="ever")[0], "median_survival": med}, 2),
                "clock_from_treatment_start": Trap({"hr": cox_hr(base, time="t_from_start", covariates=cov, strata=None,
                                                                 exposure="ever")[0], "median_survival": med}, 3, False),
                "no_confounder_adjustment": Trap({"hr": tv(long, covariates=()), "median_survival": med}, 4),
                "breslow_ties": Trap({"hr": tv(long, ties="breslow"), "median_survival": med}, None),
                "risk_set_includes_entry_day": Trap({"hr": tv(long, open_entry=False), "median_survival": med}, None)}

    def strengthen(self, k, dead):
        k = dict(k)
        if "censored_as_deaths" in dead:
            k["cens"] = min(k["cens"] * 1.4, 4.0)
        if {"ever_treated_from_diagnosis", "clock_from_treatment_start"} & set(dead):
            k["wait"] = min(k["wait"] * 1.4, 4.0)
        if "no_confounder_adjustment" in dead:
            k["conf"] = min(k["conf"] * 1.4, 3.0)
        return k

    def shortcuts(self, data) -> dict:
        """Data-light answers: the crude death-rate ratio of ever- vs never-treated patients (person-days
        from diagnosis), with the naive median of the death times, and with the reference median."""
        df = data["df"]
        t, w = self._days(data)
        ever = np.isfinite(w)
        d = df["died"].to_numpy()
        crude = float((d[ever].sum() / t[ever].sum()) / (d[~ever].sum() / t[~ever].sum()))
        return {"crude_rate_ratio_naive_median": {"hr": crude, "median_survival": float(np.median(t[d == 1]))},
                "crude_rate_ratio_true_median": {"hr": crude, "median_survival": self._ref(data)["median_survival"]}}

    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        head = (f"You are a pharmacoepidemiologist. An electronic-health-record cohort follows {m['n']} patients from "
                f"their diagnosis of {m['disease']} (diagnosed between {m['first_diagnosed']} and {m['last_diagnosed']}). "
                f"Some patients were started on {m['drug']} by their physicians at some point after diagnosis; others "
                f"never were. Follow-up ends at death, loss to follow-up or the data cut-off on {m['cutoff']}.\n\n"
                "Data files:\n"
                f"- data_files/{PATIENTS}: one row per patient. Columns: patient, diagnosis_date, age (years at "
                "diagnosis), comorbidity (number of major comorbidities at diagnosis), treatment_start_date (date the "
                "drug was started; empty if never), end_date (date of death or of last contact), died (1 = died on "
                "end_date, 0 = alive at last contact).\n\n")
        q = (f"Question: does {m['drug']} affect survival? Report hr, the hazard ratio of death for a patient on the "
             "drug versus a patient not (or not yet) on it, of the same age and comorbidity, and median_survival, the "
             "median survival from diagnosis (days) of the whole cohort.\n\n")
        method = ("" if blind else
                  "Approach: patients must survive until they start the drug, so treatment status changes during "
                  "follow-up. Measure time in days from diagnosis for everyone, split each treated patient's follow-up "
                  "at the start date (untreated before, treated after: counting-process rows with an entry time), and "
                  "fit a Cox proportional-hazards model with the time-varying treatment, age and comorbidity (Efron "
                  "ties); hr = exp(coefficient of treatment). median_survival is the Kaplan-Meier median of the time "
                  "from diagnosis to death, censoring patients alive at last contact.\n\n")
        return head + q + method + 'Return your answer as JSON: {"answer": {"hr": <ratio>, "median_survival": <days>}}'

    def demo(self, cfg) -> list[Cell]:
        inspect = (f'import numpy as np, pandas as pd\ndf = pd.read_csv("data_files/{PATIENTS}", parse_dates=["diagnosis_date", '
                   '"treatment_start_date", "end_date"])\n'
                   'print(df.head().to_string(index=False))\n'
                   'df["t"] = (df.end_date - df.diagnosis_date).dt.days\n'
                   'df["wait"] = (df.treatment_start_date - df.diagnosis_date).dt.days\n'
                   'ever = df.wait.notna()\n'
                   'print(len(df), "patients,", df.died.sum(), "deaths;", ever.sum(), "started the drug, after a median of",\n'
                   '      df.wait.median(), "days")\n'
                   'g = df.groupby(ever)\n'
                   'for col in ("age", "comorbidity", "t", "died"):\n'
                   '    print(f"mean {col}: never started {g[col].mean()[False]:.2f}, started {g[col].mean()[True]:.2f}")\n'
                   'early = (df.t <= 90) & (df.died == 1)\n'
                   'print("deaths within 90 days of diagnosis:", early.sum(), "- of these, had started the drug:", (early & ever).sum())')

        def prose2(outs):
            o = outs[0]
            num = lambda pat: [float(x) for x in re.search(pat, o).groups()] if re.search(pat, o) else None
            wait = num(r"started the drug, after a median of ([\d.]+) days")
            early = num(r"deaths within 90 days of diagnosis: (\d+) - of these, had started the drug: (\d+)")
            age, com = (num(rf"mean {c}: never started ([\d.]+), started ([\d.]+)") for c in ("age", "comorbidity"))
            txt = (f"Patients started the drug a median of {wait[0]:.0f} days after diagnosis, so a patient had to "
                   "survive that long to count as treated" if wait else "Patients started the drug some time after "
                   "diagnosis, so a patient had to survive until then to count as treated")
            txt += (f": of the {early[0]:.0f} deaths in the first 90 days, only {early[1]:.0f} were in patients who had "
                    "started it. " if early else ". ")
            txt += ("Comparing ever- with never-treated patients from diagnosis credits that immortal waiting time to "
                    "the drug, and starting the treated patients' clock at their start date misaligns time zero. "
                    "Instead I follow everyone from diagnosis and split each treated patient's follow-up at the start "
                    "date (untreated before, treated after), and fit a Cox model with the time-varying treatment, ")
            txt += (f"adjusted for age and comorbidity, since the patients who started were younger ({age[1]:.1f} "
                    f"against {age[0]:.1f} years) and had fewer comorbidities ({com[1]:.2f} against {com[0]:.2f}). "
                    if age and com else "adjusted for age and comorbidity, which differ between the groups. ")
            return txt + ("The naive ever-treated comparison is shown for contrast. The median survival from diagnosis "
                          "is the Kaplan-Meier median, with patients alive at last contact censored.")
        fit_code = ('long = split_at_treatment(df)\n'
                    'long["days"] = long.stop - long.start\n'
                    'print(long.groupby("treated").agg(rows=("died", "size"), deaths=("died", "sum"), '
                    'person_days=("days", "sum")))\n'
                    'hr, se = cox_hr(long, time="stop", entry="start", covariates=("age", "comorbidity"), strata=None)\n'
                    'naive = cox_hr(df.assign(ever=ever.astype(int)), time="t", covariates=("age", "comorbidity"), '
                    'strata=None, exposure="ever")[0]\n'
                    'print("time-varying treatment: HR %.3f (95%% CI %.3f-%.3f); naive ever-treated HR %.3f"\n'
                    '      % (hr, hr * np.exp(-1.96 * se), hr * np.exp(1.96 * se), naive))\n'
                    'med, lo, hi = km_median(df.t, df.died)\n'
                    'print(f"median survival from diagnosis {med:.0f} days (95% CI {lo:.0f}-{hi:.0f})")\n'
                    'print("FINAL hr = %.6f median_survival = %.4f" % (hr, med))')
        return [Cell("I load the cohort, compute follow-up and waiting times from the dates, and compare the patients "
                     "who started the drug with those who never did.", inspect),
                Cell(prose2, fit_code, uses=(cox_hr, km_median, split_at_treatment),
                     preamble="from statsmodels.duration.hazard_regression import PHReg\n"
                              "from statsmodels.duration.survfunc import SurvfuncRight")]


FAMILIES = [CohortCox(), ImmortalTime()]
