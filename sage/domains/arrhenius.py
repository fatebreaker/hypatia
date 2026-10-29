"""Chemical kinetics: activation energy and pre-exponential factor from rate constants (a SAGE plug-in).

The data are a temperature series of first-order (or pseudo-first-order) rate constants as a
kinetics lab records them: 6-12 temperatures with two or three replicate runs each, every k with the
relative standard error of the fit of its kinetic trace (the error is multiplicative, and its size
differs from run to run), the temperature as each instrument reads it in degrees Celsius, two
instruments -- the faster runs at the top of the range were done on a second instrument whose
thermocouple reads off by a calibration offset recorded in a column -- and an occasional failed run
the operator flagged in the notes (an air bubble, an unstable temperature, lamp drift, incomplete
mixing: its k is off by a factor of two to five).

Reference analysis (the gold, on the sampled data): drop the runs the notes flag as failed; correct
each reading by its instrument's calibration offset (true = reading - offset) and convert to kelvin;
weighted least squares of ln k on 1/T with weights 1/SE(ln k)^2, SE(ln k) = the run's relative SE of k.
Ea = -slope R, log10 A = intercept / ln 10 (A in s^-1).

Traps (the step each skips; skills from the shared taxonomy) and accepted variants:

  1 normalise  celsius_temperatures        ln k fitted on 1/T with T in degrees Celsius
  2 qc         failed_runs_included        the runs flagged as failed kept in the fit
  3 normalise  calibration_offset_ignored  instrument B's temperatures taken as read; B ran the hottest
                                           runs, so the offset tilts the Arrhenius line
  4 estimate   nonlinear_unweighted        k = A exp(-Ea/RT) fitted to k itself by unweighted nonlinear
               (conditional)               least squares: with multiplicative error the few fastest runs
                                           dominate; erratic rather than biased, so live on some draws
                                           (step None where it coincides)
  -  variant   nonlinear_weighted          the same model on k, weighted by each run's SE (sigma =
                                           k_rel_se x k): agrees with the log-scale fit to first order

Tolerance: Ea max(2 SE, 1% of Ea), log10 A max(2 SE, 0.05); SE from the weighted fit's covariance
(the runs' standard errors taken as known).

Real data: none shipped (checked 2026-09-15). The classic series -- Daniels & Johnston's N2O5
decomposition (J. Am. Chem. Soc. 43, 53, 1921) -- is behind the publisher's paywall, and textbook
copies of its table are not a stable machine-readable source for its published fit. Rdatasets and the
NIST StRD nonlinear-regression sets hold no rate-constant-versus-temperature series; the one
ln-versus-1/T table there, R's datasets::pressure (mercury vapour pressure, deg C), is a
Clausius-Clapeyron problem with no published fit to reproduce, and its enthalpy drifts over 0-360 C.

The analysis function is self-contained (numpy/pandas) so a demonstration embeds its source verbatim:
the trace and the ground truth are the same code.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sage.core import Cell, Family, Sample, Trap

CSV = "rate_constants.csv"
R_KJ = 8.314462618e-3                          # gas constant, kJ/(mol K)
NAN = float("nan")


# ----------------------------------------------------------------------------- analysis (embedded in demos)
def arrhenius_fit(df, kelvin=True, offset=True, drop_failed=True):
    """Weighted least squares of ln k on 1/T, ln k = ln A - (Ea/R)(1/T). The error of k is multiplicative, so
    on the log scale each run's SE is its relative SE of k, and its weight 1/SE^2. T = the instrument's
    reading minus its calibration offset, in kelvin; runs the notes flag as failed are left out.
    Returns (Ea in kJ/mol, log10 A with A in s^-1, SE(Ea), SE(log10 A), the rows used)."""
    failed = df["note"].fillna("").str.contains("fail|abort", case=False)
    d = df[~failed] if drop_failed else df
    T = d["temp_C"] - (d["temp_offset_C"] if offset else 0.0) + (273.15 if kelvin else 0.0)
    (slope, intercept), cov = np.polyfit(1 / T, np.log(d["k_per_s"]), 1, w=1 / d["k_rel_se"], cov="unscaled")
    R = 8.314462618e-3                                                    # kJ/(mol K)
    return (float(-slope * R), float(intercept / np.log(10)), float(np.sqrt(cov[0, 0]) * R),
            float(np.sqrt(cov[1, 1]) / np.log(10)), d)


# ----------------------------------------------------------------------------- the wrong analyses
def nonlinear_k_fit(df, weighted=False):
    """TRAP (step 4) / variant: k = k_m exp(-(Ea/R)(1/T - 1/T_m)) fitted to k itself by nonlinear least
    squares, unweighted or with sigma = k_rel_se x k; clean, corrected rows; started from the log fit."""
    from scipy.optimize import curve_fit
    ea0, _, _, _, d = arrhenius_fit(df)
    T = (d["temp_C"] - d["temp_offset_C"] + 273.15).to_numpy(float)
    k = d["k_per_s"].to_numpy(float)
    tm = float(np.mean(T))
    f = lambda T, lkm, ea: np.exp(lkm - ea / R_KJ * (1 / T - 1 / tm))
    sigma = (d["k_rel_se"].to_numpy(float) * k) if weighted else None
    lkm0 = float(np.mean(np.log(k) + ea0 / R_KJ * (1 / T - 1 / tm)))
    (lkm, ea), _ = curve_fit(f, T, k, p0=[lkm0, ea0], sigma=sigma, absolute_sigma=weighted, maxfev=20000)
    return {"Ea": float(ea), "log10_A": float((lkm + ea / (R_KJ * tm)) / np.log(10))}


# ----------------------------------------------------------------------------- the family
FAILED = ("failed: air bubble in the cell", "failed: temperature not stable during the run",
          "failed: lamp output drifted during the run", "aborted: incomplete mixing, trace unusable")
BENIGN = ("fresh stock solution", "cell cleaned before this run", "repeat of the previous run", "new buffer batch",
          "stock solution one day old")
REACTIONS = (
    "the pseudo-first-order alkaline hydrolysis of a new ester prodrug (followed at 290 nm)",
    "the thermal cis-to-trans isomerisation of an azobenzene photoswitch in toluene (followed at 440 nm)",
    "the first-order decomposition of an organic peroxide initiator in chlorobenzene (followed by UV absorbance)",
    "the degradation of a drug candidate in pH 7.4 buffer, an accelerated stability study (followed at 320 nm)",
    "the acid-catalysed hydrolysis of an acetal in buffered aqueous dioxane (followed at 270 nm)",
)
INSTRUMENTS = ("a UV-vis spectrophotometer with a Peltier-thermostatted cell holder",
               "a stopped-flow spectrometer with a circulating-bath jacket")


class ArrheniusKinetics(Family):
    name = "arr_kinetics"
    domain = "chemistry"
    keys = ("Ea", "log10_A")
    steps = {1: "use absolute temperature (kelvin), not degrees Celsius",
             2: "exclude the runs the notes flag as failed",
             3: "correct instrument B's temperatures by its calibration offset",
             4: "fit on the log scale, weighted by each run's standard error (multiplicative error)"}
    skills = {1: "normalise", 2: "qc", 3: "normalise", 4: "estimate"}
    tol_rule = "Ea: max(2 SE, 1% of Ea); log10_A: max(2 SE, 0.05); SE from the weighted fit (run SEs taken as known)"
    note = ("ground truth is the reference analysis (failed runs excluded, calibration offsets applied, kelvin, weighted "
            "least squares of ln k on 1/T with the runs' relative SEs) on the sampled rate constants, not the latent "
            "parameters; deviation is in units of tolerance, maxed over the answer keys")
    seed_blocks = {"train": 18_000_000, "test": 18_050_000}
    max_tries = 12
    rng_policy = "stream"
    demo_timeout = 120

    def knobs(self, rng):
        return {"offset": 1.0, "fail": 1.0}

    def draw(self, rng, k) -> Sample:
        ea = float(rng.uniform(40, 140))                                  # kJ/mol
        n_t = int(rng.integers(6, 13))
        lo, span = rng.uniform(5, 40), rng.uniform(30, 55)
        set_c = np.round(lo + np.linspace(0, span, n_t) + rng.normal(0, 0.3, n_t), 1)
        t_mid = float(np.mean(set_c)) + 273.15
        log10_a = (rng.uniform(-4.5, -1.5) * np.log(10) + ea / (R_KJ * t_mid)) / np.log(10)
        n_b = int(np.ceil(rng.uniform(0.3, 0.5) * n_t))                   # the hottest temperatures run on B
        off = float(np.round(rng.uniform(1.5, 4.0) * k["offset"] * rng.choice([-1, 1]), 1))
        rows = []
        for i, tc in enumerate(set_c):
            inst = "B" if i >= n_t - n_b else "A"
            o = off if inst == "B" else 0.0
            for _ in range(int(rng.integers(2, 4))):
                rel = float(np.round(np.exp(rng.uniform(np.log(0.03), np.log(0.20))), 3))
                t_true = tc + 273.15 + rng.normal(0, 0.05)
                kk = 10 ** log10_a * np.exp(-ea / (R_KJ * t_true)) * np.exp(rel * rng.normal())
                note = BENIGN[int(rng.integers(len(BENIGN)))] if rng.random() < 0.2 else ""
                rows.append([inst, round(float(tc + o), 1), o, kk, rel, note])
        df = pd.DataFrame(rows, columns=["instrument", "temp_C", "temp_offset_C", "k_per_s", "k_rel_se", "note"])
        n_f = int(rng.integers(1, 3))
        ext = np.where(df["temp_C"].rank(method="dense").isin([1, 2, df["temp_C"].nunique() - 1,
                                                                df["temp_C"].nunique()]))[0]
        pool = ext if rng.random() < 0.7 else np.arange(len(df))          # failures cluster at the extremes
        failed = [int(j) for j in rng.choice(pool, n_f, replace=False)]
        factors = []
        for j in failed:
            fac = float(rng.uniform(2.0, 5.0) ** (rng.choice([-1, 1]) * k["fail"]))
            df.loc[j, "k_per_s"] *= fac
            df.loc[j, "note"] = FAILED[int(rng.integers(len(FAILED)))]
            factors.append(fac)
        df["k_per_s"] = [float(f"{v:.4g}") for v in df["k_per_s"]]
        df.insert(0, "run", np.arange(1, len(df) + 1))
        meta = {"reaction": int(rng.integers(len(REACTIONS))), "n_temps": n_t, "cut_C": float(set_c[n_t - n_b])}
        latent = {"Ea": ea, "log10_A": float(log10_a), "offset_B_C": off, "failed_runs": [j + 1 for j in failed],
                  "failed_factors": factors, "n_temperatures": n_t}
        return Sample({"df": df, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / CSV, index=False)
        return [CSV]

    # ------------------------------------------------------------------ analysis
    def reference(self, data) -> dict:
        ea, la, se_ea, se_la, used = arrhenius_fit(data["df"])
        return {"Ea": ea, "log10_A": la, "_se_Ea": se_ea, "_se_log10_A": se_la, "_n_used": int(len(used))}

    def tolerance(self, gold) -> dict:
        return {"Ea": max(2 * gold["_se_Ea"], 0.01 * abs(gold["Ea"])), "log10_A": max(2 * gold["_se_log10_A"], 0.05)}

    def sanity(self, gold, tol, latent) -> list[str]:
        ok = all(np.isfinite(gold[k]) for k in ("Ea", "log10_A", "_se_Ea", "_se_log10_A")) and gold["Ea"] > 0
        return [] if ok else ["fit_not_identified"]

    def traps(self, data) -> dict:
        df = data["df"]

        def safe(fn):
            try:
                a = fn()
                return a if all(np.isfinite(a[k]) for k in self.keys) else {k: NAN for k in self.keys}
            except Exception:                                            # noqa: BLE001
                return {k: NAN for k in self.keys}
        pick = lambda r: {"Ea": r[0], "log10_A": r[1]}
        return {"celsius_temperatures": Trap(safe(lambda: pick(arrhenius_fit(df, kelvin=False))), 1),
                "failed_runs_included": Trap(safe(lambda: pick(arrhenius_fit(df, drop_failed=False))), 2),
                "calibration_offset_ignored": Trap(safe(lambda: pick(arrhenius_fit(df, offset=False))), 3),
                "nonlinear_unweighted": Trap(safe(lambda: nonlinear_k_fit(df)), 4, False),
                "nonlinear_weighted": Trap(safe(lambda: nonlinear_k_fit(df, weighted=True)), None)}

    def strengthen(self, k, dead):
        k = dict(k)
        if "calibration_offset_ignored" in dead:
            k["offset"] = round(min(k["offset"] * 1.5, 3.0), 4)
        if "failed_runs_included" in dead:
            k["fail"] = round(min(k["fail"] * 1.4, 3.0), 4)
        return k

    def shortcuts(self, data) -> dict:
        """Two-point Arrhenius from the mean ln k at the lowest and highest readings (kelvin, as read, all runs)."""
        df = data["df"]
        g = df.assign(lk=np.log(df["k_per_s"])).groupby("temp_C")["lk"].mean()
        x = 1 / (g.index.to_numpy(float) + 273.15)
        slope = (g.iloc[-1] - g.iloc[0]) / (x[-1] - x[0])
        return {"two_point_extremes": {"Ea": float(-slope * R_KJ),
                                       "log10_A": float((g.iloc[0] - slope * x[0]) / np.log(10))}}

    def extra_provenance(self, inst) -> dict:
        m = inst.sample.data["meta"]
        return {"knobs": inst.knobs, "design": {"reaction": REACTIONS[m["reaction"]], "n_temperatures": m["n_temps"],
                                                "n_runs": int(len(inst.sample.data["df"]))},
                "reference_diagnostics": {k[1:]: v for k, v in inst.a.gold.items() if k.startswith("_")},
                "skills": {str(s): v for s, v in self.skills.items()}}

    # ------------------------------------------------------------------ prompt
    def task_text(self, data, blind: bool = False) -> str:
        m, df = data["meta"], data["df"]
        lo, hi = df["temp_C"].min(), df["temp_C"].max()
        head = (f"You are a physical chemist. The rate of {REACTIONS[m['reaction']]} was measured at {m['n_temps']} "
                f"temperatures between about {lo:g} and {hi:g} deg C, with two or three replicate runs at each. Each "
                "run's first-order rate constant k was obtained by fitting that run's kinetic trace, which also gives "
                "the relative standard error of k. The runs were done on instrument A "
                f"({INSTRUMENTS[0]}), except the fastest ones, which needed instrument B ({INSTRUMENTS[1]}).\n\n"
                "Data files:\n"
                f"- data_files/{CSV}: one row per run. Columns: run, instrument (A or B), temp_C (the instrument's "
                "temperature reading, deg C), temp_offset_C (that instrument's offset at its latest calibration against "
                "a reference thermometer: reading minus true temperature, deg C), k_per_s (rate constant, s^-1), "
                "k_rel_se (relative standard error of k from the fit of the run's trace; the measurement error of k is "
                "multiplicative), note (the operator's remarks, e.g. whether a run failed).\n\n")
        q = ("Question: what are the activation energy Ea (kJ/mol) and the pre-exponential factor A of this reaction, "
             "k = A exp(-Ea / (R T))? Report log10 of A, with A in s^-1.\n\n")
        method = ("" if blind else
                  "Approach: leave out the runs the notes flag as failed; correct each temperature reading by its "
                  "instrument's calibration offset (true = reading - offset) and convert to kelvin; then fit ln k "
                  "against 1/T by weighted least squares with weights 1/k_rel_se^2 (the error is multiplicative, so on "
                  "the log scale each run's standard error is its relative standard error). Ea = -slope x R and "
                  "log10 A = intercept / ln 10.\n\n")
        return head + q + method + 'Return your answer as JSON: {"answer": {"Ea": <kJ/mol>, "log10_A": <log10 of A in s^-1>}}'

    def demo(self, cfg) -> list[Cell]:
        inspect = (f'import numpy as np, pandas as pd\ndf = pd.read_csv("data_files/{CSV}")\n'
                   'print(df.head(4).to_string(index=False))\n'
                   'print(df.groupby("instrument").agg(runs=("run", "size"), reading_min=("temp_C", "min"),\n'
                   '      reading_max=("temp_C", "max"), offset_C=("temp_offset_C", "first")).to_string())\n'
                   'print(f"{df.temp_C.nunique()} temperatures; k from {df.k_per_s.min():.3g} to {df.k_per_s.max():.3g} "\n'
                   '      f"1/s; relative SE of k from {df.k_rel_se.min():.3f} to {df.k_rel_se.max():.3f}")\n'
                   'notes = df[df.note.fillna("") != ""]\n'
                   'print(notes[["run", "instrument", "temp_C", "k_per_s", "note"]].to_string(index=False))\n'
                   'print("runs noted as failed or aborted:", notes.note.str.contains("fail|abort", case=False).sum())')

        def prose2(outs):
            o = outs[0]
            b = re.search(r"^B\s+\d+\s+\S+\s+\S+\s+(\S+)\s*$", o, re.M)
            off = f"instrument B reads {abs(float(b.group(1))):g} deg C {'high' if float(b.group(1)) > 0 else 'low'}" \
                if b else "instrument B carries a calibration offset"
            m = re.search(r"noted as failed or aborted: (\d+)", o)
            nf = int(m.group(1)) if m else 1
            se = re.search(r"relative SE of k from (\S+) to (\S+)", o)
            spread = f" (from {se.group(1)} to {se.group(2)})" if se else ""
            return (f"The notes flag {nf} run{'s' if nf != 1 else ''} as failed or aborted; I leave "
                    f"{'them' if nf != 1 else 'it'} out rather than let a broken trace pull the line. From its "
                    f"calibration, {off}, and it ran the hottest runs, so I correct every reading (true = reading - "
                    "offset) before converting to kelvin -- the Arrhenius equation needs absolute temperature. The error "
                    "of k is multiplicative, so I fit ln k against 1/T, where each run's standard error is its relative "
                    f"SE of k; those differ a lot between runs{spread}, so the fit is weighted by 1/SE^2 rather than "
                    "fitting k itself unweighted, which would let the few fastest runs dominate. Ea = -slope x R and "
                    "log10 A = intercept / ln 10.")
        fit = ('Ea, log10_A, se_Ea, se_log10_A, used = arrhenius_fit(df)\n'
               'print(f"{len(df) - len(used)} failed runs left out, {len(used)} runs used")\n'
               'print(f"Ea = {Ea:.2f} +/- {se_Ea:.2f} kJ/mol, log10 A = {log10_A:.3f} +/- {se_log10_A:.3f}")\n'
               'print("FINAL Ea = %.6f log10_A = %.6f" % (Ea, log10_A))')
        return [Cell("I load the rate constants and look at the design: which instrument ran which temperatures and "
                     "its calibration offset, the spread of k and of its relative error, and the operator's notes.",
                     inspect),
                Cell(prose2, fit, uses=(arrhenius_fit,))]


FAMILIES = [ArrheniusKinetics()]
