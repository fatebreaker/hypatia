"""Epidemiology: outbreak transmissibility from surveillance counts (a SAGE plug-in).

The data are what an outbreak analyst gets from a surveillance system: confirmed cases per day
by specimen-collection date, extracted from the case database on the last day of the series,
plus the distributions a public-health team has on file -- the reporting delay (specimen to
database, estimated from an earlier, completed period), the generation interval and the
incubation period. The DGP builds in the irregularities real counts carry:

  * right truncation: a case collected on day t is in the extract only if its reporting delay
    was at most (extraction day - t), so the last days are progressively incomplete;
  * day-of-week effects: fewer specimens are collected at weekends;
  * under-ascertainment: only a fraction of infections is ever confirmed (a level, which
    leaves growth rates untouched -- the task states it anyway, as real briefings do);
  * overdispersion (negative-binomial counts);
  * an optional control-measure change-point (the growth rate drops, with a short, smooth
    transition), and, in `epi_r0`, a widening of testing on a stated date (a step up in
    ascertainment that is not epidemic growth).

Reference analysis (the gold, computed on the sampled counts): a log-linear negative-binomial
(NB2) regression of daily counts on time with day-of-week effects and log(fraction reported by the extraction
day) as an offset -- the delay distribution nowcasts the incomplete days instead of discarding
them; the NB2 dispersion is a moment estimate from the fit. A continuous change-point (segmented regression; the break located by profiling the
deviance and accepted when the likelihood-ratio statistic exceeds 20) separates phases, and its
covariance carries the break-location uncertainty (Muggeo 2003). The growth rate r becomes a
reproduction number through the generation interval g: R = 1 / sum_s g(s) exp(-r s)
(Wallinga & Lipsitch 2007).

Families, their answer keys and their traps (the step each skips; "cond" = conditional):

  epi_rt   the CURRENT growth rate and reproduction number (the latest phase)
    1 qc            no_truncation_correction   raw counts, incomplete days included
    2 normalise     no_weekday_effects (cond)  harmless over long windows (edge weeks only)
    3 model-choice  single_exponential (cond)  one growth rate across the change-point
                                               (core on draws with a change-point)
    4 estimate      incubation_for_gi          r -> R with the incubation period
  epi_r0   the INITIAL growth rate and R0 (before control measures), testing widened mid-growth
    1 qc            no_truncation_correction (cond)
    2 normalise     ignore_testing_change      the ascertainment step read as growth
    2 normalise     no_weekday_effects (cond)
    3 model-choice  single_exponential         one growth rate across the change-point
    4 estimate      incubation_for_gi

A conditional trap is recorded with step None on draws where it coincides with the
reference (the engine's rule), so "always add weekday dummies" is neither rewarded nor
penalised where it changes nothing. Day-of-week effects barely bias a long log-linear fit --
only the partial weeks at the window's ends carry any imbalance -- so that trap is conditional
by design: it is live only where the window is short enough for the imbalance to matter.

The analysis functions are self-contained (numpy/pandas/statsmodels) so a demonstration embeds
their source verbatim: the trace and the ground truth are the same code.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sage.core import Cell, Family, Sample, Trap

CASES, DELAY, NATHIST = "cases.csv", "reporting_delay.csv", "natural_history.csv"
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
NAN = float("nan")


# ----------------------------------------------------------------------------- analysis
def reported_fraction(delay, n_days):
    """Fraction of each day's cases already in the extract taken at the end of the last day:
    F(n_days - 1 - t), F the cumulative reporting-delay distribution."""
    cdf = np.clip(np.cumsum(delay["probability"].to_numpy(float)), 0, 1)
    lag = n_days - 1 - np.arange(n_days)
    return np.where(lag < len(cdf), cdf[np.minimum(lag, len(cdf) - 1)], 1.0)


def growth_fit(cases, frac, step_day=None, dow=True, correct=True, change=True, min_frac=0.1, min_lr=20.0):
    """Log-linear negative-binomial regression of daily counts on time (days since the first
    date), with day-of-week effects, log(frac) as offset for the incomplete recent days, an
    optional level shift from `step_day` on, and a continuous change-point profiled over the
    days, accepted if the likelihood-ratio statistic exceeds `min_lr`. Returns the growth rate
    before and after the break (equal if none is accepted), their standard errors (including
    the break-location uncertainty) and the break day."""
    import statsmodels.api as sm
    from scipy.optimize import minimize_scalar
    keep = frac >= min_frac if correct else np.ones(len(cases), bool)
    d = cases[keep]
    y, t = d["cases"].to_numpy(float), np.arange(len(cases), dtype=float)[keep]
    off = np.log(frac[keep]) if correct else None
    cols = [np.ones_like(t), t]
    if dow:
        cols += [(d["weekday"] == w).to_numpy(float) for w in ("Tue", "Wed", "Thu", "Fri", "Sat", "Sun")]
    if step_day is not None:
        cols.append((t >= step_day).astype(float))

    def fit(X, alpha):                       # NB2 GLM, variance mu + alpha mu^2
        return sm.GLM(y, np.column_stack(X), family=sm.families.NegativeBinomial(alpha=alpha), offset=off).fit()

    def dispersion(m, p):                    # moment estimator of alpha from a fit's means
        mu = m.fittedvalues
        return max(np.sum(((y - mu) ** 2 - mu) / mu ** 2) / (len(y) - p), 1e-4)
    alpha = dispersion(sm.GLM(y, np.column_stack(cols), family=sm.families.Poisson(), offset=off).fit(), len(cols))
    hinge = lambda tau: np.maximum(t - tau, 0)
    tau = None
    if change:                               # profile the break over the days, then refine it
        dev = lambda tau: fit(cols + [hinge(tau)], alpha).deviance
        grid = np.arange(t[0] + 10, t[-1] - 6)
        g0 = grid[int(np.argmin([dev(g) for g in grid]))]
        tau = minimize_scalar(dev, bounds=(g0 - 1, g0 + 1), method="bounded", options={"xatol": 0.01}).x
        alpha = dispersion(fit(cols + [hinge(tau)], alpha), len(cols) + 2)
    m0 = fit(cols, alpha)
    out = {"r_before": m0.params[1], "r_after": m0.params[1], "se_before": m0.bse[1], "se_after": m0.bse[1],
           "tau": None, "lr": 0.0, "alpha": alpha}
    if not change:
        return out
    m1 = fit(cols + [hinge(tau)], alpha)
    lr = m0.deviance - m1.deviance
    if lr < min_lr:
        return {**out, "lr": lr}
    # Muggeo's step column -I(t > tau): its covariance carries the uncertainty in the break
    c = fit(cols + [hinge(tau), -(t > tau).astype(float)], alpha).cov_params()
    j = len(cols)
    b1, b2 = m1.params[1], m1.params[j]
    return {"r_before": b1, "r_after": b1 + b2, "se_before": np.sqrt(c[1, 1]),
            "se_after": np.sqrt(c[1, 1] + c[j, j] + 2 * c[1, j]), "tau": tau, "lr": lr, "alpha": alpha}


def r_to_R(r, interval):
    """Reproduction number from growth rate r via a discretised interval distribution:
    R = 1 / sum_s g(s) exp(-r s) (Wallinga & Lipsitch 2007)."""
    s = interval["day"].to_numpy(float)
    g = interval["probability"].to_numpy(float)
    return float(1.0 / np.sum(g / g.sum() * np.exp(-r * s)))


# ----------------------------------------------------------------------------- simulator helpers
def _disc_gamma(mean, sd, first=0):
    """Gamma(mean, sd) discretised to whole days by P(s - 1/2 < X <= s + 1/2), from day `first`."""
    k, th = (mean / sd) ** 2, sd ** 2 / mean
    smax = int(np.ceil(stats.gamma.ppf(0.999, k, scale=th))) + 1
    s = np.arange(first, smax + 1)
    p = stats.gamma.cdf(s + 0.5, k, scale=th) - stats.gamma.cdf(np.maximum(s - 0.5, 0), k, scale=th)
    return s, np.round(p / p.sum(), 5)


def _renorm(p):
    p = np.asarray(p, float)
    return p / p.sum()


def _gi(data, which="generation_interval"):
    nh = data["nathist"]
    return pd.DataFrame({"day": nh["day"], "probability": nh[which]})


# ----------------------------------------------------------------------------- the family
class Outbreak(Family):
    """Shared DGP, writer, reference machinery and tolerance of both epidemiology families."""
    domain = "epidemiology"
    keys: tuple = ()
    max_tries = 12
    rng_policy = "stream"
    demo_timeout = 240
    tol_rule = ("growth_rate: max(3 SE, 0.004) with SE from the reference's negative-binomial covariance (incl. the "
                "change-point location); R: that band propagated through R(r)")
    note = ("ground truth is the reference analysis (nowcast-corrected, day-of-week-adjusted segmented negative-binomial "
            "regression; r -> R via the generation interval) on the sampled counts, not the latent parameter; "
            "deviation is in units of tolerance, maxed over the answer keys")
    change_prob = 1.0
    testing = False                     # a widening of testing (ascertainment step) during growth

    # ------------------------------------------------------------------ draw
    def knobs(self, rng):
        return {"delay": 1.0, "contrast": 1.0, "ratio": 1.0, "testing": 1.0}

    def _growth(self, rng, n, k):
        """Log expected-count trajectory (before weekday effects), and its latent truth."""
        t = np.arange(n, dtype=float)
        change = bool(rng.random() < self.change_prob)
        if not change:
            r = rng.uniform(2.5, 7.0) / n
            return r * t, {"change": False, "r_initial": r, "r_current": r, "tau": None}
        tau = self._tau(rng, n)
        r1 = rng.uniform(2.5, 7.5) / tau
        r2 = -rng.uniform(0.015, 0.10) if rng.random() < 0.7 else rng.uniform(0.015, 0.05)
        r2 = min(r2, r1 - 0.05 * k["contrast"])
        if abs(r2) < 0.012:
            r2 = -0.012 if r2 <= 0 else 0.012
        s = 1.0                                                  # transition width, days
        lg = r1 * t + (r2 - r1) * s * np.logaddexp(0, (t - tau) / s)
        return lg, {"change": True, "r_initial": r1, "r_current": r2, "tau": float(tau)}

    def _tau(self, rng, n):
        raise NotImplementedError

    def draw(self, rng, k) -> Sample:
        n = int(rng.integers(60, 121))
        start = dt.date(2014, 1, 1) + dt.timedelta(days=int(rng.integers(0, 4000)))
        dates = [start + dt.timedelta(days=i) for i in range(n)]
        wd = np.array([d.weekday() for d in dates])
        eff = rng.normal(0, 0.06, 7)
        eff[5] -= rng.uniform(0.2, 0.5)
        eff[6] -= rng.uniform(0.4, 0.9)
        eff -= eff.mean()
        lg, lat = self._growth(rng, n, k)
        ascert = rng.uniform(0.1, 0.6)
        level = np.log(rng.uniform(2, 15))
        step_day, mult = None, 1.0
        if self.testing:
            hi = 0.8 * (lat["tau"] if lat["tau"] else n)
            step_day = int(rng.uniform(0.3, 1.0) * (hi - 14)) + 12
            mult = min(1.0 + rng.uniform(0.8, 3.0) * k["testing"], 8.0)
            lg = lg + np.log(mult) * (np.arange(n) >= step_day)
        mu = np.exp(level + lg + eff[wd])
        size = rng.uniform(30, 200)
        complete = rng.negative_binomial(size, size / (size + mu))
        dmean = min(rng.uniform(2.0, 6.0) * k["delay"], 12.0)
        ds, dp = _disc_gamma(dmean, dmean * rng.uniform(0.5, 0.8))
        delay = pd.DataFrame({"delay_days": ds, "probability": dp})
        frac = reported_fraction(delay, n)
        observed = rng.binomial(complete, np.clip(frac, 0, 1))
        gm = rng.uniform(3.0, 8.0)
        gsd = gm * rng.uniform(0.3, 0.6)
        ratio = rng.uniform(0.45, 0.7) ** k["ratio"] if rng.random() < 0.4 else rng.uniform(1.4, 2.2) ** k["ratio"]
        im = gm * ratio
        isd = im * rng.uniform(0.3, 0.5)
        gs, gp = _disc_gamma(gm, gsd, first=1)
        is_, ip = _disc_gamma(im, isd, first=1)
        days = np.arange(1, max(gs.max(), is_.max()) + 1)
        nh = pd.DataFrame({"day": days,
                           "generation_interval": np.round(_renorm(np.interp(days, gs, gp, left=0, right=0)), 5),
                           "incubation_period": np.round(_renorm(np.interp(days, is_, ip, left=0, right=0)), 5)})
        cases = pd.DataFrame({"date": [d.isoformat() for d in dates], "weekday": [WEEKDAYS[w] for w in wd],
                              "cases": observed.astype(int)})
        gi = pd.DataFrame({"day": nh["day"], "probability": nh["generation_interval"]})
        meta = {"gi_mean": round(gm, 1), "gi_sd": round(gsd, 1), "inc_mean": round(im, 1), "inc_sd": round(isd, 1),
                "delay_mean": round(dmean, 1), "ascertainment": round(ascert, 2), "extraction": dates[-1].isoformat(),
                "first": dates[0].isoformat(), "n": n,
                "testing_date": dates[step_day].isoformat() if step_day is not None else None,
                "step_day": step_day,
                "pathogen": ("respiratory virus", "enteric virus", "influenza A subtype")[int(rng.integers(3))]}
        latent = {"change": lat["change"], "tau": lat["tau"], "r_initial": float(lat["r_initial"]),
                  "r_current": float(lat["r_current"]), "testing_multiplier": float(mult),
                  "weekday_log_effects": [float(x) for x in eff], "nb_size": float(size), "delay_mean": float(dmean),
                  "gi_mean": float(gm), "incubation_mean": float(im)}
        latent.update(self._latent_keys(lat, gi))
        return Sample({"cases": cases, "delay": delay, "nathist": nh, "meta": meta}, latent)

    # ------------------------------------------------------------------ files
    def write(self, data, d: Path) -> list[str]:
        data["cases"].to_csv(d / CASES, index=False)
        data["delay"].to_csv(d / DELAY, index=False)
        data["nathist"].to_csv(d / NATHIST, index=False)
        return [CASES, DELAY, NATHIST]

    # ------------------------------------------------------------------ analysis
    def _fit(self, data, **kw):
        """growth_fit on the draw with one stage switched off by `kw` (none: the reference), cached."""
        cache = data.setdefault("_cache", {})
        key = tuple(sorted(kw.items()))
        if key not in cache:
            cases, kw = data["cases"], dict(kw)
            step = data["meta"]["step_day"] if kw.pop("testing", True) else None
            cache[key] = growth_fit(cases, reported_fraction(data["delay"], len(cases)), step_day=step, **kw)
        return cache[key]

    def _answer(self, fit, interval) -> dict:
        r = float(fit[self.phase])
        return {"growth_rate": r, self.keys[1]: r_to_R(r, interval)}

    def reference(self, data) -> dict:
        f, gi = self._fit(data), _gi(data)
        se = f["se_after"] if self.phase == "r_after" else f["se_before"]
        return {**self._answer(f, gi), "_se": float(se), "_tau": f["tau"], "_changed": f["tau"] is not None,
                "_lr": float(f["lr"]), "_gi": (gi["day"].to_numpy(float), _renorm(gi["probability"]))}

    def tolerance(self, gold) -> dict:
        tr = max(3.0 * gold["_se"], 0.004)
        R, r = gold[self.keys[1]], gold["growth_rate"]
        s, g = gold["_gi"]
        rr = lambda x: 1.0 / np.sum(g * np.exp(-x * s))
        return {"growth_rate": tr, self.keys[1]: float(max(abs(rr(r + tr) - R), abs(rr(r - tr) - R)))}

    def sanity(self, gold, tol, latent) -> list[str]:
        out = []
        if gold["_changed"] != latent["change"]:
            out.append("change_point_misclassified")
        elif latent["change"] and abs(gold["_tau"] - latent["tau"]) > 4:
            out.append("change_point_misplaced")
        return out

    def traps(self, data) -> dict:
        gi, inc = _gi(data), _gi(data, "incubation_period")
        out = {}
        for name, kw in self.arms().items():
            try:
                out[name] = self._answer(self._fit(data, **kw), gi)
            except Exception:                                            # noqa: BLE001
                out[name] = {k: NAN for k in self.keys}
        ref = self._fit(data)
        out["incubation_for_gi"] = {"growth_rate": float(ref[self.phase]),
                                    self.keys[1]: r_to_R(float(ref[self.phase]), inc)}
        return {n: Trap(a, self.trap_step[n], self.trap_core(n, data, ref)) for n, a in out.items()}

    def strengthen(self, k, dead):
        k = dict(k)
        if "no_truncation_correction" in dead:
            k["delay"] *= 1.4
        if "single_exponential" in dead:
            k["contrast"] *= 1.5
        if "incubation_for_gi" in dead:
            k["ratio"] *= 1.3
        if "ignore_testing_change" in dead:
            k["testing"] *= 1.5
        return k

    def shortcuts(self, data) -> dict:
        """Week-on-week growth from the raw extract: log(last 7 days / previous 7) / 7,
        converted with the generation interval -- no truncation or phase handling."""
        y = data["cases"]["cases"].to_numpy(float)
        r = float(np.log(max(y[-7:].sum(), 1) / max(y[-14:-7].sum(), 1)) / 7)
        return {"week_on_week_raw": {"growth_rate": r, self.keys[1]: r_to_R(r, _gi(data))}}

    def extra_provenance(self, inst) -> dict:
        g = inst.a.gold
        return {"knobs": inst.knobs, "reference_diagnostics": {"se": g["_se"], "tau": g["_tau"], "lr": g["_lr"]},
                "skills": {str(s): v for s, v in self.skills.items()}}

    # ------------------------------------------------------------------ prompt
    def _files_text(self, data) -> str:
        m = data["meta"]
        return ("Data files:\n"
                f"- data_files/{CASES}: confirmed cases per day by specimen-collection date, {m['first']} to "
                f"{m['extraction']}. Columns: date, weekday, cases. The extract was taken from the case database at "
                f"the end of {m['extraction']}.\n"
                f"- data_files/{DELAY}: the distribution of the delay (whole days) from specimen collection to the case "
                "appearing in the database, estimated from an earlier period whose reporting is complete. Columns: "
                "delay_days, probability.\n"
                f"- data_files/{NATHIST}: the natural history of the infection, discretised to whole days. Columns: day, "
                f"generation_interval (probability that infector and infectee are infected that many days apart; mean "
                f"{m['gi_mean']} d, SD {m['gi_sd']} d), incubation_period (infection to symptom onset; mean "
                f"{m['inc_mean']} d, SD {m['inc_sd']} d).\n")

    def demo(self, cfg) -> list[Cell]:
        return _demo(self, cfg)


class CurrentGrowth(Outbreak):
    name = "epi_rt"
    keys = ("growth_rate", "R")
    phase = "r_after"
    change_prob = 0.6
    seed_blocks = {"train": 13_000_000, "test": 13_050_000}
    steps = {1: "correct the right-truncated recent days with the reporting-delay distribution",
             2: "adjust for the day-of-week pattern in specimen collection",
             3: "locate the change-point and estimate the current phase only",
             4: "convert the growth rate to R with the generation interval (not the incubation period)"}
    skills = {1: "qc", 2: "normalise", 3: "model-choice", 4: "estimate"}
    trap_step = {"no_truncation_correction": 1, "no_weekday_effects": 2, "single_exponential": 3,
                 "incubation_for_gi": 4}

    def _tau(self, rng, n):
        return n - rng.uniform(22, 45)

    def _latent_keys(self, lat, gi):
        return {"growth_rate": float(lat["r_current"]), "R": r_to_R(lat["r_current"], gi)}

    def arms(self) -> dict:
        return {"no_truncation_correction": {"correct": False}, "no_weekday_effects": {"dow": False},
                "single_exponential": {"change": False}}

    def trap_core(self, name, data, ref) -> bool:
        if name == "no_weekday_effects":
            return False
        if name == "single_exponential":
            return ref["tau"] is not None
        return True

    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        head = (f"You are an infectious-disease epidemiologist. An outbreak of a novel {m['pathogen']} is under way in a "
                f"city; public-health officials estimate that about {round(100 * m['ascertainment'])}% of "
                "infections are confirmed by a test. Control measures may or may not have been introduced during the "
                "period covered.\n\n")
        q = ("Question: at the time of the extract, how fast is the epidemic growing or declining? Report the current "
             "exponential growth rate of daily infections r (per day; negative if declining) and the corresponding "
             "current reproduction number R.\n\n")
        method = ("" if blind else
                  "Approach: the counts for the most recent specimen dates are incomplete, because cases keep arriving "
                  "in the database for days after collection; weekend collection is lower. Fit a log-linear "
                  "negative-binomial regression of daily counts on time with day-of-week effects and the log of the "
                  "fraction of each day's cases already reported (from the delay distribution) as an offset. If the "
                  "growth rate changed, fit a continuous change-point (segmented regression) and report the growth "
                  "rate of the latest phase. Convert r to R with the generation interval g: R = 1 / sum_s g(s) "
                  "exp(-r s).\n\n")
        return head + self._files_text(data) + "\n" + q + method + \
            'Return your answer as JSON: {"answer": {"growth_rate": <per day>, "R": <number>}}'


class InitialGrowth(Outbreak):
    name = "epi_r0"
    keys = ("growth_rate", "R0")
    phase = "r_before"
    change_prob = 1.0
    testing = True
    seed_blocks = {"train": 13_100_000, "test": 13_150_000}
    steps = {1: "correct the right-truncated recent days with the reporting-delay distribution",
             2: "separate reporting artefacts from growth: the testing change (a level shift) and weekday effects",
             3: "locate the change-point and estimate the initial phase only",
             4: "convert the growth rate to R0 with the generation interval (not the incubation period)"}
    skills = {1: "qc", 2: "normalise", 3: "model-choice", 4: "estimate"}
    trap_step = {"no_truncation_correction": 1, "ignore_testing_change": 2, "no_weekday_effects": 2,
                 "single_exponential": 3, "incubation_for_gi": 4}

    def _tau(self, rng, n):
        return n * rng.uniform(0.55, 0.75)

    def _latent_keys(self, lat, gi):
        return {"growth_rate": float(lat["r_initial"]), "R0": r_to_R(lat["r_initial"], gi)}

    def arms(self) -> dict:
        return {"no_truncation_correction": {"correct": False}, "ignore_testing_change": {"testing": False},
                "no_weekday_effects": {"dow": False}, "single_exponential": {"change": False}}

    def trap_core(self, name, data, ref) -> bool:
        return name not in ("no_weekday_effects", "no_truncation_correction")

    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        head = (f"You are an infectious-disease epidemiologist. An outbreak of a novel {m['pathogen']} spread in a "
                f"city; public-health officials estimate that about {round(100 * m['ascertainment'])}% of "
                f"infections are confirmed by a test. On {m['testing_date']} testing was widened from hospital patients "
                "to anyone with symptoms. Control measures were introduced at some point during the period covered.\n\n")
        q = ("Question: how transmissible was the pathogen before control measures? Report the exponential growth "
             "rate of daily infections r (per day) during the initial, uncontrolled phase and the basic reproduction "
             "number R0.\n\n")
        method = ("" if blind else
                  "Approach: the counts for the most recent specimen dates are incomplete (cases keep arriving in the "
                  "database after collection) and weekend collection is lower. Fit a log-linear "
                  "negative-binomial regression of daily counts on time with day-of-week effects, a level shift "
                  "from the date testing was widened, and the log of the fraction of each day's cases already reported "
                  "as an offset; locate the change in growth with a continuous change-point (segmented regression) "
                  "and report the growth rate before it. Convert r to R0 with the generation interval g: "
                  "R0 = 1 / sum_s g(s) exp(-r s).\n\n")
        return head + self._files_text(data) + "\n" + q + method + \
            'Return your answer as JSON: {"answer": {"growth_rate": <per day>, "R0": <number>}}'


# ----------------------------------------------------------------------------- demonstrations
INSPECT = f'''import numpy as np, pandas as pd
cases = pd.read_csv("data_files/{CASES}")
delay = pd.read_csv("data_files/{DELAY}")
nh = pd.read_csv("data_files/{NATHIST}")
print(cases.head(3).to_string(index=False), "...", cases.tail(3).to_string(index=False, header=False), sep="\\n")
print("days:", len(cases), "| total cases:", cases.cases.sum())
ratio = cases.cases / cases.cases.rolling(7, center=True).mean()      # each day against its own week
wk = ratio.groupby(cases.weekday).median().reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
print("cases relative to the centred 7-day mean, median by weekday:", wk.round(2).to_dict())
cdf = delay.probability.cumsum().to_numpy()
print("reporting delay: mean %.1f d; fraction reported within 0,1,2,3,5,7 days:" % (delay.delay_days * delay.probability).sum(),
      [round(float(cdf[min(d, len(cdf) - 1)]), 2) for d in (0, 1, 2, 3, 5, 7)])
for col in ("generation_interval", "incubation_period"):
    print(col, "mean %.2f d" % ((nh.day * nh[col]).sum() / nh[col].sum()))
print("last 10 days:", cases.cases.tail(10).tolist())'''


def _demo(fam, cfg) -> list[Cell]:
    r0 = fam.name == "epi_r0"
    key = fam.keys[1]
    step = ""
    if r0:
        m = re.search(r"On (\d{4}-\d\d-\d\d) testing was widened", cfg["task"])
        step = (f'step_day = int(np.flatnonzero(cases.date == "{m.group(1)}")[0])   # testing widened: a level shift\n'
                if m else "step_day = None\n")
    phase = "r_before" if r0 else "r_after"
    fit_code = (('' if r0 else 'step_day = None\n') + step +
                'frac = reported_fraction(delay, len(cases))\n'
                'print("fraction already reported, last 7 days:", frac[-7:].round(2).tolist())\n'
                'fit = growth_fit(cases, frac, step_day=step_day)\n'
                'print("change-point: likelihood ratio %.1f ->" % fit["lr"], "none accepted" if fit["tau"] is None else\n'
                '      "break at day %.1f (%s)" % (fit["tau"], cases.date[int(round(fit["tau"]))]))\n'
                'print("growth rate before %.4f (SE %.4f), after %.4f (SE %.4f) per day" % (fit["r_before"], '
                'fit["se_before"], fit["r_after"], fit["se_after"]))\n'
                f'r = fit["{phase}"]\n'
                'gi = pd.DataFrame({"day": nh.day, "probability": nh.generation_interval})\n'
                'inc = pd.DataFrame({"day": nh.day, "probability": nh.incubation_period})\n'
                'print("R with the generation interval %.3f (with the incubation period it would be %.3f, the wrong '
                'interval)" % (r_to_R(r, gi), r_to_R(r, inc)))\n'
                f'print("FINAL growth_rate = %.6f {key} = %.6f" % (r, r_to_R(r, gi)))')

    def prose2(outs):
        o = outs[0]
        wk = re.search(r"\{(.*)\}", o)
        wkend = ""
        if wk:
            try:
                d = {k.strip(" '"): float(v) for k, v in (x.split(":") for x in wk.group(1).split(","))}
                wkend = f" (Saturday {d.get('Sat', float('nan')):.2f} and Sunday {d.get('Sun', float('nan')):.2f} of an average day)"
            except ValueError:
                pass
        txt = ("Several features of these data would bias a naive exponential fit. First, the extract was taken on the last "
               "date, and cases keep arriving for days after collection, so the most recent days are incomplete: a "
               "fit to the raw tail would read the reporting lag as a decline. Rather than discard those days, I use "
               "the delay distribution to compute the fraction of each day's cases already reported and put its log "
               "in the regression as an offset (a nowcast). Second, fewer specimens are collected at weekends"
               f"{wkend}, so the model gets day-of-week effects. ")
        if r0:
            txt += ("Third, testing was widened during the growth phase; that raises the level of confirmed cases "
                    "without any change in transmission, so the model gets a level shift from that date, or the jump "
                    "would be read as faster growth. ")
        txt += ("The growth rate may also have changed when control measures began, so I profile a continuous "
                "change-point over the days, accept it only if the likelihood-ratio statistic is large (above 20), and "
                f"report the growth rate of the {'initial' if r0 else 'latest'} phase; its standard error includes "
                "the uncertainty in where the break is. Finally, a growth rate becomes a reproduction number through "
                "the generation interval (time between successive infections), not the incubation period: "
                "R = 1 / sum_s g(s) exp(-r s).")
        return txt

    return [Cell("I start by loading the three files and checking the shape of the series: the weekday pattern, how "
                 "complete the last days are likely to be given the reporting delay, and the two natural-history "
                 "distributions.", INSPECT),
            Cell(prose2, fit_code, uses=(reported_fraction, growth_fit, r_to_R))]


FAMILIES = [CurrentGrowth(), InitialGrowth()]
