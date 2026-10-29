"""Enzyme kinetics, a fourth science in one file: DEMONSTRATION ONLY, not in the paper's experiments.

Written from scratch against the contract to show what a new domain costs: a draw, a
writer, a reference analysis, two documented wrong analyses and a tolerance. Everything
else (validity loop, seeding, layout, manifests, audit, demonstration traces) is the
engine's.

The exercise. Continuous-assay progress curves -- product against time -- at seven initial
substrate concentrations; report Vmax and Km. The pipeline has two textbook pitfalls:

  step 1  initial rates must account for substrate depletion. A straight line through a
          whole progress curve underestimates the initial rate, most at low substrate,
          where the curve bends first, so Km and Vmax come out inflated.
          -> trap `linear_whole_curve`
  step 2  fit the rate law as a hyperbola. The Lineweaver-Burk double reciprocal gives
          the smallest, noisiest rates the most leverage and biases both constants
          (Dowd & Riggs 1965). -> trap `lineweaver_burk`, on correctly estimated rates

Reference: a global least-squares fit of the integrated Michaelis-Menten equation,
P(t) = S0 - Km W((S0/Km) exp((S0 - Vmax t)/Km)), W the Lambert function (Goudar, Sonnad &
Duggleby 1999), to all curves at once: no initial-rate approximation at all. Accepted
variant: initial rates from each curve's early, curved part (a quadratic through the
origin over points below 25% conversion) and a nonlinear hyperbola fit -- the careful
textbook route, which must agree with the reference (I3).

The analysis functions are self-contained numpy/scipy so the demonstration trace can embed
their source verbatim: the trace and the ground truth are the same code.

Upgraded to a full family (2026-09-15) without changing a byte of its default output: a `skills`
tag per step from the shared taxonomy (step 1 model-choice: an integrated rate law, not straight
lines through curving progress curves; step 2 estimate: a hyperbola fit, not a double-reciprocal
line), and `task_text(data, blind=True)`. The prompt has never named its method -- it states the
assay, the files and the question -- so the blind prompt is the default one. Skills are not copied
into instance provenance (the newer families do) because that would change the emitted files.
Checked with `python -m sage check --family enzyme_mm --n 60` and the held-out constant audit;
produced by experiments/sage_domains/gen_osc_arr_enz.sh.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.special import wrightomega

from sage.core import Cell, Family, Sample, Trap

CSV = "progress_curves.csv"


# ----------------------------------------------------------------------------- analyses
def mm_product(t, s0, vmax, km):
    """Integrated Michaelis-Menten: product formed by time t from substrate s0 (none at t=0).
    wrightomega(z) = W(exp(z)) avoids overflowing exp for large S0/Km."""
    return s0 - km * np.real(wrightomega(np.log(s0 / km) + (s0 - vmax * t) / km))


def fit_progress_curves(curve, s0, t, p):
    """Global least squares of the integrated equation over every curve -> (vmax, km)."""
    ids = np.unique(curve)

    def resid(th):
        v, k = np.exp(th)
        return np.concatenate([p[curve == c] - mm_product(t[curve == c], s0[curve == c][0], v, k) for c in ids])
    r = least_squares(resid, np.log([2 * p.max() / t.max(), np.median(s0)]))
    return tuple(float(x) for x in np.exp(r.x))


def initial_rates(curve, s0, t, p, max_conversion=0.25):
    """Each curve's rate at t=0 from a quadratic through the origin over the readings below
    `max_conversion` of its substrate: the t^2 term absorbs the slowing from depletion."""
    S, v = [], []
    for c in np.unique(curve):
        m = curve == c
        tt, pp, ss = t[m], p[m], s0[m][0]
        k = pp <= max_conversion * ss
        if k.sum() < 4:
            k = np.argsort(np.argsort(tt)) < 4
        v.append(np.linalg.lstsq(np.column_stack([tt[k], tt[k] ** 2]), pp[k], rcond=None)[0][0])
        S.append(ss)
    return np.array(S), np.array(v)


def hyperbola_fit(S, v):
    """Nonlinear least squares of v = Vmax S / (Km + S) -> (vmax, km)."""
    (vm, km), _ = curve_fit(lambda s, a, b: a * s / (b + s), S, v, p0=[1.2 * v.max(), np.median(S)], maxfev=20000)
    return float(vm), float(km)


def whole_curve_rates(curve, s0, t, p):
    """TRAP (step 1): the slope of a straight line through each whole progress curve."""
    S, v = [], []
    for c in np.unique(curve):
        m = curve == c
        v.append(np.polyfit(t[m], p[m], 1)[0])
        S.append(s0[m][0])
    return np.array(S), np.array(v)


def lineweaver_burk(S, v):
    """TRAP (step 2): 1/v = (Km/Vmax)(1/S) + 1/Vmax by ordinary least squares."""
    slope, icept = np.polyfit(1 / S, 1 / v, 1)
    return float(1 / icept), float(slope / icept)


def _arrays(df):
    return (df["curve"].to_numpy(), df["s0_uM"].to_numpy(float), df["time_min"].to_numpy(float),
            df["product_uM"].to_numpy(float))


# ----------------------------------------------------------------------------- the family
class MichaelisMenten(Family):
    name = "enzyme_mm"
    domain = "biochemistry"
    keys = ("vmax", "km")
    steps = {1: "initial rates account for substrate depletion", 2: "fit the hyperbola, not a linearisation"}
    skills = {1: "model-choice", 2: "estimate"}
    tol_rule = "vmax 5% of gold, km 15% of gold"
    note = ("DEMONSTRATION family (sage/domains/enzyme.py), not part of the paper's experiments. Ground truth is the "
            "global integrated Michaelis-Menten fit to the sampled progress curves, not the latent constants.")
    seed_blocks = {"train": 11_000_000, "test": 11_050_000}
    demonstration_only = True

    def task_text(self, data, blind: bool = False) -> str:
        """The question and the data; the method was never named, so `blind` changes nothing."""
        return ("You are an expert in enzymology. You will analyse a continuous enzyme assay.\n\nData files:\n"
                f"- data_files/{CSV}: one row per reading. Columns: curve (assay id), s0_uM (initial substrate "
                "concentration in uM; no product at time 0), time_min (minutes after mixing), product_uM (product "
                "concentration in uM, from absorbance, with measurement noise).\n\n"
                "Every assay uses the same enzyme concentration; the enzyme is stable over the assay and the "
                "product does not inhibit it.\n\nQuestion: what are the enzyme's maximal rate Vmax (uM/min) and "
                "Michaelis constant Km (uM)?\n\n"
                'Return your answer as JSON: {"answer": {"vmax": <uM/min>, "km": <uM>}}')

    def knobs(self, rng):
        return 1.0                                               # measurement-noise multiplier

    def draw(self, rng, noise) -> Sample:
        km = float(np.exp(rng.uniform(np.log(5), np.log(200))))
        vmax = float(np.exp(rng.uniform(np.log(0.5), np.log(20))))
        s0s = np.round(km * np.geomspace(0.2, 5.0, 7) * np.exp(rng.normal(0, 0.05, 7)), 3)
        t_end = rng.uniform(1.5, 3.0) * km / vmax                # lowest curve converts ~60-90%
        times = np.round(t_end * np.linspace(0, 1, 15)[1:] ** 1.6, 3)   # dense early readings
        sigma = rng.uniform(0.005, 0.02) * noise * s0s.min()
        rows = [(f"A{i + 1}", s, tt, round(float(pp), 4)) for i, s in enumerate(s0s)
                for tt, pp in zip(times, mm_product(times, s, vmax, km) + rng.normal(0, sigma, len(times)))]
        df = pd.DataFrame(rows, columns=["curve", "s0_uM", "time_min", "product_uM"])
        return Sample({"df": df}, latent={"vmax": vmax, "km": km, "noise_sd": float(sigma), "t_end": float(t_end)})

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / CSV, index=False)
        return [CSV]

    def reference(self, data) -> dict:
        v, k = fit_progress_curves(*_arrays(data["df"]))
        return {"vmax": v, "km": k}

    def traps(self, data) -> dict:
        a = _arrays(data["df"])
        ans = lambda vk: {"vmax": vk[0], "km": vk[1]}
        out = {}
        for name, step, fn in (("linear_whole_curve", 1, lambda: hyperbola_fit(*whole_curve_rates(*a))),
                               ("lineweaver_burk", 2, lambda: lineweaver_burk(*initial_rates(*a))),
                               ("initial_rates_hyperbola", None, lambda: hyperbola_fit(*initial_rates(*a)))):
            try:
                out[name] = Trap(ans(fn()), step)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                out[name] = Trap({"vmax": float("nan"), "km": float("nan")}, step)   # unpaid -> I4 redraws
        return out

    def tolerance(self, gold) -> dict:
        return {"vmax": 0.05 * abs(gold["vmax"]), "km": 0.15 * abs(gold["km"])}

    def strengthen(self, noise, dead):
        return noise * 1.5 if "lineweaver_burk" in dead else noise   # LB bias grows with the noise

    def shortcuts(self, data) -> dict:
        """Read it off the plot: Vmax = the steepest early slope, Km = the substrate whose
        early slope is closest to half of it."""
        curve, s0, t, p = _arrays(data["df"])
        S, v = [], []
        for c in np.unique(curve):
            m = curve == c
            v.append(p[m][0] / t[m][0])
            S.append(s0[m][0])
        S, v = np.array(S), np.array(v)
        return {"half_max_by_eye": {"vmax": float(v.max()), "km": float(S[np.argmin(abs(v - v.max() / 2))])}}

    def demo(self, cfg) -> list[Cell]:
        inspect_code = (f'import numpy as np, pandas as pd\ndf = pd.read_csv("data_files/{CSV}")\n'
                        'print(df.head())\n'
                        'g = df.groupby("curve").agg(s0=("s0_uM", "first"), n=("time_min", "size"), '
                        't_last=("time_min", "max"), p_last=("product_uM", "last"))\n'
                        'g["conversion_at_end"] = (g.p_last / g.s0).round(3)\nprint(g)\n'
                        'print("lowest-substrate curve converts", round(100 * g.sort_values("s0").conversion_at_end.iloc[0]), '
                        '"% by the last reading")')

        def p2(outs):
            m = re.search(r"converts (\d+) %", outs[0])
            conv = f"{m.group(1)}%" if m else "most"
            return (f"The lowest-substrate curve converts {conv} of its substrate by the last reading, so the rate "
                    "falls visibly during each assay: a straight line through a whole curve would underestimate its "
                    "initial rate, most at low substrate. Instead of approximating initial rates, I fit the integrated "
                    "Michaelis-Menten equation (its closed form uses the Lambert W function) to every curve at once by "
                    "least squares. As a cross-check I also take initial rates from a quadratic through the origin over "
                    "each curve's first quarter of conversion, and fit the hyperbola to them directly (never a "
                    "double-reciprocal plot, which gives the noisiest low-substrate rates the most weight).")
        fit_code = ('curve, s0, t, p = df["curve"].to_numpy(), df["s0_uM"].to_numpy(float), '
                    'df["time_min"].to_numpy(float), df["product_uM"].to_numpy(float)\n'
                    'S, v0 = initial_rates(curve, s0, t, p)\n'
                    'print("initial rates:", dict(zip(S.round(3).tolist(), v0.round(4).tolist())))\n'
                    'print("hyperbola fit to initial rates (vmax, km):", np.round(hyperbola_fit(S, v0), 4))\n'
                    'vmax, km = fit_progress_curves(curve, s0, t, p)\n'
                    'print("FINAL vmax = %.6f km = %.6f" % (vmax, km))')
        return [Cell("I load the progress curves and check, per assay, how far the reaction has gone by the last "
                     "reading.", inspect_code),
                Cell(p2, fit_code, uses=(mm_product, fit_progress_curves, initial_rates, hyperbola_fit),
                     preamble="from scipy.optimize import curve_fit, least_squares\n"
                              "from scipy.special import wrightomega")]

    def extra_provenance(self, inst) -> dict:
        return {"demonstration_only": True, "knobs": {"noise_multiplier": inst.knobs}}


FAMILIES = [MichaelisMenten()]
