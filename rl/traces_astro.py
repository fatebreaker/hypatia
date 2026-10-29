#!/usr/bin/env python
"""Executed demonstration traces for the three Kepler light-curve families (third domain).

Same contract as rl/traces_causal.py: every cell is EXECUTED in the harness's Sandbox
against the instance's own data files, the tool response is the real output, the system
prompt and prompt builder are harness/agents' own, the answer is PARSED from the executed
output (never copied from ground truth), and a trace whose parsed answer fails the
instance's grader is discarded.

Each family's analysis is the reference analysis of generators/astro_kepler.py, and the
functions it calls are embedded in the cells VERBATIM from generators/astro_analysis.py
(inspect.getsource), so the demonstration and the ground truth are the same code:

  transit  inspect -> clip cosmic rays, 1-day running-median detrend, BLS search, odd/even
           check -> local out-of-transit baselines, limb-darkened (star.csv u1,u2) transit
           model averaged over the 29.4-min integration, least-squares fit, two passes
  eb       inspect -> eclipse-masked detrend, BLS -> odd/even test at twice the BLS period
           (half-period alias), primary = deeper eclipse, dip-fit timing, O-C period
  rot      inspect -> Lomb-Scargle for orientation -> McQuillan ACF (uniform grid, gaps = 0,
           Gaussian smoothing, first peak unless the second is higher), peak-series check

Three cells per family: the sandbox replays every earlier cell on each call, so the BLS and
the fits are kept to one cell each and the answer is printed from the last one.

  venv/bin/python rl/traces_astro.py --family transit --n 150 --out $DEEP_DATA_ROOT/sft/astro_transit
  venv/bin/python rl/traces_astro.py --family all --n 150 --out $DEEP_DATA_ROOT/sft/astro3 --workers 8
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402
from generators import astro_analysis as A      # noqa: E402
from generators.astro_kepler import ANSWER_KEYS, LC_FILE, STAR_FILE  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from agents import SYSTEM, build_prompt          # noqa: E402
from sandbox import Sandbox                      # noqa: E402

FAMILIES = ("transit", "eb", "rot")
NUM = r"(-?[\d.]+(?:[eE][-+]?\d+)?)"
FINAL_RE = {f: re.compile(r"FINAL\s+" + r"\s+".join(rf"{k}\s*=\s*{NUM}" for k in keys))
            for f, keys in ANSWER_KEYS.items()}

_MOD = Path(A.__file__).read_text()
CONST = "\n".join(ln for ln in _MOD.splitlines() if ln.startswith("CAD = "))
GAUSS = "\n".join(ln for ln in _MOD.splitlines() if ln.startswith(("_GX, _GW =", "_TH =", "_WT =")))
assert CONST and GAUSS.count("\n") == 2, "astro_analysis constants moved; update traces_astro"


def src(*fns) -> str:
    return "\n\n".join(inspect.getsource(fn).rstrip() for fn in fns)


def inspect_cell(fam: str) -> str:
    star = (f'star = pd.read_csv("data_files/{STAR_FILE}").iloc[0]\nprint(star.to_string())\n' if fam == "transit" else "")
    return (f'import numpy as np, pandas as pd\nlc = pd.read_csv("data_files/{LC_FILE}")\n'
            'print(lc.shape)\nprint(lc.head())\n' + star +
            'print(lc.groupby("quarter").agg(n=("time", "size"), start=("time", "min"), end=("time", "max"), '
            'median_flux=("flux", "median"), median_err=("flux_err", "median")).round(5))\n'
            't, f = lc["time"].to_numpy(), lc["flux"].to_numpy()\n'
            'gaps = np.diff(t)\n'
            'print("cadence (min):", round(np.median(gaps) * 1440, 2), "| gaps > 0.5 d:", np.round(gaps[gaps > 0.5], 2).tolist())\n'
            'p2p = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2)\n'
            'daily = pd.Series(f).groupby(np.floor(t)).median()\n'
            'print("point-to-point scatter (ppm):", round(p2p * 1e6), "| spread of daily medians, 5-95% (ppm):", '
            'round((daily.quantile(0.95) - daily.quantile(0.05)) * 1e6))')


def _obs(out: str) -> dict:
    """Numbers the analyst read off the inspection cell's output."""
    m = re.search(r"point-to-point scatter \(ppm\): (\d+) \| spread of daily medians, 5-95% \(ppm\): (\d+)", out)
    return {"p2p": int(m.group(1)), "range": int(m.group(2))} if m else {"p2p": 0, "range": 0}


def moves(fam: str) -> list:
    """(prose, code) per cell; prose may be a callable of the inspection output."""
    first = ("I start by loading the light curve" + (" and the star file" if fam == "transit" else "") +
             ", and checking the quarters, the cadence and gaps, the noise level, and how large the variability "
             "is compared with the noise.", inspect_cell(fam))
    if fam == "transit":
        def p2(o):
            return (f"The daily median flux wanders by about {o['range']} ppm (5-95%) against a point-to-point scatter "
                    f"of {o['p2p']} ppm: starspots and instrumental drifts on time scales of days that a transit search "
                    "must not see. I drop upward outliers (cosmic rays), then divide by a running median over a "
                    "1-day window, computed separately in each continuous stretch so the trend never bridges a gap. "
                    "A 1-day window follows the variability but not a transit lasting a few hours. On the flattened "
                    "light curve I run a box least-squares (BLS) search, and fold at twice the best period to check "
                    "that odd and even transits have the same depth, as a planet's must.")
        c2 = (CONST + "\n\n" + src(A.segments, A.running_median, A.clip_high, A.bls_power, A.bls, A.fold) + "\n\n\n"
              'keep = clip_high(t, f)\nt, f = t[keep], f[keep]\nprint("clipped", int((~keep).sum()), "upward outliers")\n'
              'flat = f / running_median(t, f, 1.0)\n'
              'periods, power, best = bls(t, flat)\n'
              'shown = []\nfor i in np.argsort(power)[::-1]:\n'
              '    if all(abs(periods[i] / periods[j] - 1) > 0.02 for j in shown):\n        shown.append(i)\n'
              '    if len(shown) == 5:\n        break\n'
              'print("strongest BLS peaks (period, power):", [(round(float(periods[i]), 4), round(float(power[i]))) for i in shown])\n'
              'print("best:", {k: round(v, 5) for k, v in best.items()})\n'
              'd2 = fold(t, 2 * best["period"], best["t0"])\n'
              'odd = 1 - np.mean(flat[np.abs(d2) < best["duration"] / 3])\n'
              'even = 1 - np.mean(flat[np.abs(np.abs(d2) - best["period"]) < best["duration"] / 3])\n'
              'print("odd / even transit depth (ppm):", round(odd * 1e6), round(even * 1e6))')
        p3 = ("The BLS box gives the period, epoch and duration, but a box is not the transit's shape. The star is "
              "limb-darkened: its disk is brighter at the centre than at the limb, so the share of light a planet hides "
              "depends on where it crosses the disk, the transit bottom is rounded, and the depth is not (Rp/R*)^2. "
              "rp_rs therefore comes from a physical model with the star's quadratic "
              "coefficients u1, u2 from star.csv, integrated over the stellar disk. Each 29.4-minute point is also the "
              "average over its exposure, which smears ingress and egress, so the model is averaged over 11 "
              "sub-exposures. To normalise without distorting the transit, I fit a quadratic to the out-of-transit "
              "flux around each transit (the transit masked) and divide it out, fit (t0, P, Rp/R*, a/R*, b) by least "
              "squares from several starting impact parameters, then redo the normalisation with the fitted duration "
              "and refit.")
        c3 = ("from scipy.optimize import least_squares\n\n" + GAUSS + "\n\n\n" +
              src(A.occult, A.transit_model, A.t14, A.transit_windows, A.fit_transit) + "\n\n\n"
              'u1, u2 = float(star["u1"]), float(star["u2"])\n'
              'tw, fw = transit_windows(t, f, best["period"], best["t0"], best["duration"])\n'
              'fit1 = fit_transit(tw, fw, best["period"], best["t0"], np.sqrt(max(best["depth"], 1e-6)), best["duration"], u1, u2,\n'
              '                   b_starts=(0.2, 0.7))\n'
              'T = max(fit1["t14"], 2 * CAD)\n'
              'tw, fw = transit_windows(t, f, fit1["period"], fit1["t0"], T)\n'
              'fit = fit_transit(tw, fw, fit1["period"], fit1["t0"], fit1["rp_rs"], T, u1, u2, b_starts=(fit1["b"], 0.1, 0.5, 0.8))\n'
              'print({k: round(v, 6) for k, v in fit.items()})\n'
              'core = np.abs(fold(tw, fit["period"], fit["t0"])) < fit["t14"] / 6\n'
              'depth = 1 - np.mean(fw[core])\n'
              'print("central depth (ppm):", round(depth * 1e6), "| sqrt(depth) =", round(np.sqrt(depth), 5), '
              '"vs limb-darkened Rp/R* =", round(fit["rp_rs"], 5))\n'
              'print("residual rms (ppm):", round(fit["rms"] * 1e6), "vs point-to-point", round(p2p * 1e6), '
              '"| sigma(Rp/R*):", round(fit["rp_rs_err"], 5), "| transits used:", len(np.unique(np.round((tw - fit["t0"]) / fit["period"]))))\n'
              'print("FINAL period = %.6f rp_rs = %.6f" % (fit["period"], fit["rp_rs"]))')
        return [first, (p2, c2), (p3, c3)]
    if fam == "eb":
        p2 = ("The eclipses are deep and last hours, so a plain running median would be dragged down into them. I drop "
              "upward outliers, then detrend iteratively: points far below a running median (the eclipses) are masked, "
              "the mask is widened by 0.1 d on each side, and the median is recomputed from the out-of-eclipse flux "
              "only. A BLS search on the flattened light curve then gives the spacing of the eclipses.")
        c2 = (CONST + "\n\n" + src(A.segments, A.running_median, A.clip_high, A.eb_detrend, A.bls_power, A.bls, A.fold) +
              "\n\n\n"
              'keep = clip_high(t, f)\nt, f = t[keep], f[keep]\n'
              'flat = eb_detrend(t, f)\n'
              'periods, power, best = bls(t, flat)\n'
              'shown = []\nfor i in np.argsort(power)[::-1]:\n'
              '    if all(abs(periods[i] / periods[j] - 1) > 0.02 for j in shown):\n        shown.append(i)\n'
              '    if len(shown) == 5:\n        break\n'
              'print("strongest BLS peaks (period, power):", [(round(float(periods[i]), 4), round(float(power[i]))) for i in shown])\n'
              'print("best:", {k: round(v, 5) for k, v in best.items()})')
        p3 = ("BLS reports the spacing of successive eclipses, which is the orbital period only if every eclipse is the "
              "same kind. When the secondary sits near phase 0.5 and is not much shallower than the primary, the two "
              "fold on top of each other and BLS returns half the period; in an eccentric orbit a secondary near phase "
              "1/3 or 2/3 can make it return a third. So I fit every event at the BLS spacing with one template and "
              "group the depths by epoch mod m (odd/even for m = 2): if the events split into groups that differ -- "
              "primary, secondary, no eclipse -- the true period is m times the BLS period, and if they are all alike "
              "there is one eclipse type per BLS period and the secondary is elsewhere in the fold. Depth differences "
              "are judged against the scatter between events, because spots and residual trends change eclipse depths "
              "from one event to the next; an odd-even timing offset also means a doubled period. At the adopted period I time "
              "the BLS eclipse with a symmetric dip fit, and look for the other eclipse only outside that eclipse's full "
              "extent: a secondary can be far shallower than the primary, and anything that grazes the primary's wings "
              "would look deeper than it. The search compares each candidate box with its own flanks, so leftover "
              "out-of-eclipse modulation does not pose as an eclipse. The deeper eclipse is the primary; I refine the "
              "period from a linear fit to the individual primary times and measure where the secondary falls, which for "
              "an eccentric orbit is not phase 0.5.")
        c3 = ("from scipy.optimize import least_squares\n\n" +
              src(A.dip_fit, A.event_depths, A.eclipse_times, A.secondary_search, A.fit_away, A.alias_multiple, A.eb_solve) + "\n\n\n"
              'sol = eb_solve(t, flat, best["period"], best["t0"], best["duration"])\n'
              'print(sol["n_events"], "eclipse events at the BLS spacing; grouped by epoch mod m:")\n'
              'for mm, (rms, means) in sol["groups"].items():\n'
              '    print("  m = %d: within-group depth rms %.5f, group mean depths %s" % (mm, rms, [round(x, 4) for x in means]))\n'
              'print("odd - even timing offset (min):", round(sol["odd_even_time_offset"] * 1440, 2), '
              '"| adopted period = %d x BLS period" % sol["multiple"])\n'
              'print("primary depth", round(sol["depth_primary"], 5), "at", round(sol["t_primary"], 5), '
              '"| secondary depth", round(sol["depth_secondary"], 5), "at", round(sol["t_secondary"], 5))\n'
              'print("e cos(omega) ~ pi/2 (phase - 0.5) =", round(np.pi / 2 * (sol["secondary_phase"] - 0.5), 4))\n'
              'print("FINAL period = %.6f secondary_phase = %.6f" % (sol["period"], sol["secondary_phase"]))')
        return [first, (p2, c2), (p3, c3)]
    if fam == "rot":
        p2 = ("The modulation comes from starspots, so it is quasi-periodic: spots grow and decay, and spots on opposite "
              "sides of the star put two dips into each rotation, where a periodogram peaks at half the rotation period. "
              "After dropping upward outliers I look at the Lomb-Scargle periodogram for orientation only.")
        c2 = (CONST + "\n\n" + src(A.segments, A.running_median, A.clip_high, A.lomb_scargle) + "\n\n\n"
              'keep = clip_high(t, f)\nt, f = t[keep], f[keep]\n'
              'df = 0.1 / (t.max() - t.min())\nn = int((1 / 0.5 - 1 / 70.0) / df)\n'
              'freq = 1 / 70.0 + np.arange(n) * df\n'
              'pw = lomb_scargle(t, f, 1 / 70.0, df, n)\n'
              'shown = []\nfor i in np.argsort(pw)[::-1]:\n'
              '    if all(abs(freq[i] / freq[j] - 1) > 0.05 for j in shown):\n        shown.append(i)\n'
              '    if len(shown) == 4:\n        break\n'
              'print("Lomb-Scargle peaks (period d, relative power):", [(round(float(1 / freq[i]), 3), round(float(pw[i] / pw.max()), 3)) for i in shown])')
        p3 = ("I measure the period with the autocorrelation function (McQuillan et al. 2013, 2014), which compares the "
              "light curve with itself shifted by whole rotations and does not care how many dips a rotation has. The "
              "flux goes onto a uniform grid at the Kepler cadence with the gaps set to zero, the ACF is smoothed with a "
              "Gaussian of 9 lags, and the period is the lag of the first ACF peak, unless the second peak is higher: "
              "with two active longitudes the peak at half the period is the lower one. A line through the lags of the "
              "successive peaks at multiples of that period is the cross-check.")
        c3 = ("from scipy.ndimage import gaussian_filter1d\nfrom scipy.signal import find_peaks\n\n" +
              src(A.uniform_grid, A.acf, A.acf_peaks, A.acf_period, A.acf_period_series) + "\n\n\n"
              'P, first, pk = acf_period(t, f)\n'
              'print("ACF peaks (lag d, height):", [(round(a, 3), round(b, 3)) for a, b in pk[:6]])\n'
              'print("first peak", round(first, 4), "| adopted", round(P, 4), "| peak-series slope", round(acf_period_series(pk, P), 4))\n'
              'print("strongest Lomb-Scargle period / ACF period =", round((1 / freq[np.argmax(pw)]) / P, 3))\n'
              'print("FINAL rotation_period = %.6f" % P)')
        return [first, (p2, c2), (p3, c3)]
    raise ValueError(fam)


# ============================================================================ library style
# `--style lib`: the same pipeline steps written as an astronomer would with astropy.timeseries
# (BoxLeastSquares, LombScargle), batman (limb-darkened, exposure-supersampled transit model)
# and scipy.optimize -- tens of lines instead of the reference's embedded source. The answer
# must still reproduce the instance's gold (the reference analysis) within tolerance, or the
# demonstration is discarded like any other.
LIB_TRANSIT_2 = """from astropy.timeseries import BoxLeastSquares
seg = np.cumsum(np.r_[0, np.diff(t) > 0.5])                 # continuous stretches between gaps
trend = pd.Series(f).groupby(seg).transform(lambda s: s.rolling(49, center=True, min_periods=10).median()).to_numpy()
keep = f / trend - 1 < 5 * p2p                               # drop cosmic rays; dips stay
t, f, flat = t[keep], f[keep], (f / trend)[keep]
durations = np.array([1, 1.5, 2, 3, 4, 5, 6.5, 8, 10]) / 24
bls = BoxLeastSquares(t, flat, dy=np.full(len(t), p2p))             # white noise from the point-to-point scatter
res = bls.power(np.exp(np.linspace(np.log(0.5), np.log(60), 30000)), durations)
i = np.argmax(res.power)
P, t0, dur, depth = res.period[i], res.transit_time[i], res.duration[i], res.depth[i]
print(f"BLS peak: P = {P:.5f} d, t0 = {t0:.4f}, duration = {24 * dur:.1f} h, depth = {1e6 * depth:.0f} ppm")
st = bls.compute_stats(P, dur, t0)
print("odd / even transit depth (ppm):", round(1e6 * st["depth_odd"][0]), round(1e6 * st["depth_even"][0]),
      "+/-", round(1e6 * st["depth_odd"][1]))"""

LIB_TRANSIT_3 = """import batman
from scipy.optimize import least_squares
u1, u2 = float(star["u1"]), float(star["u2"])

def local_flux(P, t0, dur):
    \"\"\"Each transit divided by a quadratic through its own out-of-transit flux; points within one duration.\"\"\"
    dt = (t - t0 + P / 2) % P - P / 2
    ep = np.round((t - t0) / P)
    tt, ff = [], []
    for e in np.unique(ep[np.abs(dt) < dur / 2]):
        w = (ep == e) & (np.abs(dt) < dur / 2 + max(2 * dur, 0.35))
        oot = w & (np.abs(dt) > 0.6 * dur + 0.02)
        if (oot & (dt < 0)).sum() >= 4 and (oot & (dt > 0)).sum() >= 4:
            near = w & (np.abs(dt) < dur)
            tt.append(t[near])
            ff.append(f[near] / np.polyval(np.polyfit(dt[oot], f[oot], 2), dt[near]))
    return np.concatenate(tt), np.concatenate(ff)

def fit(P, t0, dur, k0, b_starts):
    tt, ff = local_flux(P, t0, dur)
    pm = batman.TransitParams()
    pm.t0, pm.per, pm.rp, pm.a, pm.inc, pm.ecc, pm.w = t0, P, k0, 10.0, 89.0, 0.0, 90.0
    pm.u, pm.limb_dark = [u1, u2], "quadratic"
    # batman spaces its sub-exposures from -exp_time/2 to +exp_time/2 inclusive; exp_time * 10/11 puts the
    # 11 samples at the centres of 11 equal slices of the 29.4-min exposure (an unbiased average)
    model = batman.TransitModel(pm, tt, supersample_factor=11, exp_time=29.4244 / 1440 * 10 / 11)

    def resid(x):
        pm.t0, pm.per, pm.rp, pm.a = x[:4]
        pm.inc = np.degrees(np.arccos(x[4] / x[3]))          # impact parameter b = a cos(i)
        return model.light_curve(pm) - ff
    best = None
    for b0 in b_starts:
        a0 = np.clip(P / (np.pi * dur) * np.sqrt(max((1 + k0) ** 2 - b0 ** 2, 0.05)), 1.6, 250)
        r = least_squares(resid, [t0, P, k0, a0, b0], x_scale=[2e-3, 1e-5 * P, 0.1 * k0, 0.1 * a0, 0.1],
                          bounds=([t0 - 0.1, 0.99 * P, 0.2 * k0, 1.5, 0], [t0 + 0.1, 1.01 * P, min(0.6, 3 * k0), 300, 1.2]))
        best = r if best is None or r.cost < best.cost else best
    t0, P, k, a, b = best.x
    t14 = P / np.pi * np.arcsin(min(np.sqrt(max((1 + k) ** 2 - b ** 2, 0)) / (a * np.sqrt(1 - (b / a) ** 2)), 1))
    return t0, P, k, a, b, t14, np.std(best.fun)

t0, P, k, a, b, t14, rms = fit(P, t0, dur, np.sqrt(depth), (0.2, 0.7))
t0, P, k, a, b, t14, rms = fit(P, t0, max(t14, 0.041), k, (b, 0.1, 0.5, 0.8))
print(f"P = {P:.6f} d, t0 = {t0:.5f}, Rp/R* = {k:.5f}, a/R* = {a:.2f}, b = {b:.3f}, T14 = {24 * t14:.2f} h")
print(f"residual rms {1e6 * rms:.0f} ppm vs point-to-point scatter {1e6 * p2p:.0f} ppm")
print("FINAL period = %.6f rp_rs = %.6f" % (P, k))"""

LIB_EB_2 = """from astropy.timeseries import BoxLeastSquares
seg = np.cumsum(np.r_[0, np.diff(t) > 0.5])                 # continuous stretches between gaps

def trend(mask):
    \"\"\"1.5-day running median of the unmasked flux, per stretch, interpolated under the mask.\"\"\"
    s = pd.Series(np.where(mask, np.nan, f))
    s = s.groupby(seg).transform(lambda x: x.rolling(73, center=True, min_periods=5).median())
    return s.interpolate(limit_direction="both").to_numpy()

mask = np.zeros(len(t), bool)
r = f / trend(mask) - 1
thr = 5 * 1.4826 * np.median(np.abs(r - np.median(r)))      # threshold fixed once, so the mask cannot run away
for _ in range(3):                                           # mask the eclipses, widen ~0.1 d, recompute
    low = f / trend(mask) - 1 < -thr
    mask = pd.Series(low.astype(float)).rolling(11, center=True, min_periods=1).max().to_numpy() > 0
flat = f / trend(mask)
keep = flat - 1 < 5 * p2p                                    # drop cosmic rays
t, flat = t[keep], flat[keep]
durations = np.array([1, 1.5, 2, 3, 4, 5, 6.5, 8, 10]) / 24
res = BoxLeastSquares(t, flat).power(np.exp(np.linspace(np.log(0.5), np.log(60), 30000)), durations)
i = np.argmax(res.power)
Pb, t0, dur = res.period[i], res.transit_time[i], res.duration[i]
print(f"BLS peak: {Pb:.5f} d, t0 = {t0:.4f}, duration = {24 * dur:.1f} h, depth = {res.depth[i]:.4f}")"""

LIB_EB_3 = """from scipy.optimize import curve_fit
fold = lambda P, tc: (t - tc + P / 2) % P - P / 2

def dip(P, tc, use=True):
    \"\"\"Gaussian dip fitted to the fold at P near tc -> (centre, core depth, width).\"\"\"
    d = fold(P, tc)
    w = (np.abs(d) < 1.5 * dur) & use
    g = lambda x, c0, A, s, c: c - A * np.exp(-0.5 * ((x - c0) / s) ** 2)
    (c0, A, s, c), _ = curve_fit(g, d[w], flat[w], p0=[0, np.clip(1 - flat[w].min(), 1e-4, 0.99), dur / 4, 1],
                                 bounds=([-dur, 0, 0.005, 0.5], [dur, 1, 2 * dur, 1.5]))
    core = w & (np.abs(d - c0) < max(0.3 * s, 0.0205))
    return tc + c0, (1 - np.median(flat[core]) if core.sum() >= 3 else A), s

# 1) period alias: the depth of every eclipse at the BLS spacing, grouped by epoch mod m
tc0, _, s0 = dip(Pb, t0)
d = fold(Pb, tc0)
ev = pd.DataFrame({"ep": np.round((t - tc0) / Pb).astype(int), "g": np.exp(-0.5 * (d / s0) ** 2), "y": 1 - flat})
ev = ev[np.abs(d) < 3 * s0]
depth = ev.groupby("ep").apply(lambda e: (e.g * e.y).sum() / (e.g ** 2).sum() if (e.g > 0.5).sum() >= 2 else np.nan).dropna()
e, v, n = depth.index.to_numpy(), depth.to_numpy(), len(depth)
within = lambda m: sum(((v[e % m == j] - v[e % m == j].mean()) ** 2).sum() for j in range(m) if (e % m == j).any())
good = {}
for m in (2, 3, 4):
    if n < 2 * m:
        break
    means = [v[e % m == j].mean() for j in range(m) if (e % m == j).any()]
    F = (within(1) - within(m)) / (m - 1) / (within(m) / (n - m))
    print(f"m = {m}: group mean depths {np.round(means, 4).tolist()}, F = {F:.1f}")
    if F > 30 and max(means) - min(means) > 0.02 * max(means):
        good[m] = within(m) / (n - m)
m = min(good, key=good.get) if good else 1                   # best grouping, then its smallest adequate divisor
m = min([j for j in good if m % j == 0 and (within(j) - within(m)) / max(m - j, 1) / (within(m) / (n - m)) < 30] + [m])
t_odd, t_even = dip(2 * Pb, tc0)[0], dip(2 * Pb, tc0 + Pb)[0] - Pb
if m == 1 and abs(t_odd - t_even) > max(3 * 0.0204, 0.1 * dur):
    m = 2                                                    # odd and even eclipses offset in time
P = m * Pb

# 2) at P: the deepest group is one eclipse; the other is looked for only outside its full extent
means = {j: v[e % m == j].mean() for j in range(m) if (e % m == j).any()}
jp = max(means, key=means.get)
t1, d1, s1 = dip(P, tc0 + jp * Pb)
away = np.abs(fold(P, t1)) > max(4 * s1, 0.75 * dur)
others = {j: x for j, x in means.items() if j != jp}
if others and max(others.values()) > max(5 * np.std(v[e % m == jp]), 0.05 * means[jp]):
    t2 = tc0 + max(others, key=others.get) * Pb
else:
    t2 = BoxLeastSquares(t[away], flat[away]).power([P], durations[durations < P / 4]).transit_time[0]
t2, d2, s2 = dip(P, t2, use=away)
(t_pri, d_pri), (t_sec, d_sec) = sorted([(t1, d1), (t2, d2)], key=lambda x: -x[1])
phase = ((t_sec - t_pri) / P) % 1
print(f"period = {m} x BLS period | primary depth {d_pri:.4f} at {t_pri:.4f}, secondary depth {d_sec:.4f} at {t_sec:.4f}")
print("FINAL period = %.6f secondary_phase = %.6f" % (P, phase))"""

LIB_ROT_2 = """from astropy.timeseries import LombScargle
keep = f - pd.Series(f).rolling(5, center=True, min_periods=1).median().to_numpy() < 5 * p2p
t, f = t[keep], f[keep]                                      # drop cosmic rays
freq, power = LombScargle(t, f).autopower(minimum_frequency=1 / 70, maximum_frequency=2, samples_per_peak=10)
top = []
for j in np.argsort(power)[::-1]:
    if all(abs(freq[j] / freq[k] - 1) > 0.05 for k in top):
        top.append(j)
    if len(top) == 4:
        break
print("Lomb-Scargle peaks (period d, relative power):", [(round(1 / freq[j], 3), round(power[j] / power.max(), 3)) for j in top])"""

LIB_ROT_3 = """from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
cad = 29.4244 / 1440
idx = np.round((t - t[0]) / cad).astype(int)
x = np.zeros(idx[-1] + 1)
x[idx] = f - f.mean()                                        # uniform grid, gaps set to zero
spec = np.fft.rfft(x, 2 * len(x))
acf = np.fft.irfft(spec * np.conj(spec))[: len(x) // 2]
acf = gaussian_filter1d(acf / acf[0], 9, truncate=3.1)       # smoothed with a 9-lag Gaussian
lags = np.arange(len(acf)) * cad
pk, _ = find_peaks(acf, prominence=0.02)
print("ACF peaks (lag d, height):", [(round(lags[p], 3), round(acf[p], 3)) for p in pk[:6]])
P = lags[pk[1]] if len(pk) > 1 and acf[pk[1]] > acf[pk[0]] else lags[pk[0]]
print("first peak", round(lags[pk[0]], 3), "d | adopted", round(P, 3), "d | strongest Lomb-Scargle period / adopted =",
      round(1 / freq[np.argmax(power)] / P, 3))
print("FINAL rotation_period = %.6f" % P)"""


def moves_lib(fam: str) -> list:
    """Library-style cells: same inspection, same pipeline steps, astropy/batman/scipy instead of
    embedded source."""
    first = ("I start by loading the light curve" + (" and the star file" if fam == "transit" else "") +
             ", and checking the quarters, the cadence and gaps, the noise level, and how large the variability "
             "is compared with the noise.", inspect_cell(fam))
    if fam == "transit":
        def p2(o):
            return (f"The daily median flux wanders by about {o['range']} ppm (5-95%) against a point-to-point scatter "
                    f"of {o['p2p']} ppm: starspots and instrumental drifts that a transit search must not see. I divide "
                    "out a one-day running median (49 cadences), computed separately in each stretch between gaps; it "
                    "follows the variability but not a transit lasting hours. Then astropy's box least-squares searches "
                    "the flattened light curve, and its odd/even statistics check that alternate transits are equally "
                    "deep, as a planet's must be.")
        p3 = ("A box is not the transit's shape. The star is limb-darkened, so the share of light the planet hides "
              "depends on where it crosses the disk, the transit bottom is rounded, and the depth is not (Rp/R*)^2; each "
              "29.4-minute point also averages over its exposure, which smears ingress and egress. I model the transit "
              "with batman, using the star's quadratic coefficients from star.csv and supersampling each exposure. "
              "Around every transit I divide by a quadratic fitted to the out-of-transit flux only, so the "
              "normalisation cannot eat into the transit, then fit (t0, P, Rp/R*, a/R*, b) by least squares from "
              "several starting impact parameters; a second pass redoes the normalisation with the fitted duration.")
        return [first, (p2, LIB_TRANSIT_2), (p3, LIB_TRANSIT_3)]
    if fam == "eb":
        p2 = ("The eclipses are deep and last hours, so a plain running median would be dragged down into them. I "
              "detrend iteratively: points far below a 1.5-day running median (the eclipses) are masked, the mask is "
              "widened by about 0.1 d, and the median is recomputed from the out-of-eclipse flux only, separately in "
              "each stretch between gaps. astropy's box least-squares then gives the spacing of the eclipses.")
        p3 = ("That spacing is the orbital period only if every eclipse is the same kind. When the secondary sits near "
              "phase 0.5 and is not much shallower, primary and secondary fold on top of each other and BLS returns "
              "half the period (a third, for some eccentric orbits). So I measure the depth of every eclipse at the "
              "BLS spacing and group the depths by epoch mod m: if the groups differ significantly (an F test against "
              "the scatter between eclipses, since spots change depths from event to event), the period is m times "
              "longer; an odd/even timing offset also means a doubled period. At the adopted period I fit the deepest "
              "eclipse group and look for the other eclipse only outside that eclipse's full extent, because a "
              "secondary can be far shallower than the primary's wings. The deeper eclipse is the primary, and the "
              "secondary's phase need not be 0.5 if the orbit is eccentric.")
        return [first, (p2, LIB_EB_2), (p3, LIB_EB_3)]
    if fam == "rot":
        p2 = ("The modulation comes from starspots, so it is quasi-periodic: spots grow and decay, and spots on opposite "
              "sides of the star put two dips into each rotation, where a periodogram peaks at half the rotation "
              "period. After dropping upward outliers I look at astropy's Lomb-Scargle periodogram for orientation only.")
        p3 = ("I measure the period with the autocorrelation function (McQuillan et al. 2013, 2014), which compares the "
              "light curve with itself shifted by whole rotations and does not care how many dips a rotation has. The "
              "flux goes onto a uniform grid at the Kepler cadence with the gaps set to zero, the ACF is smoothed with a "
              "Gaussian of 9 lags, and the period is the lag of the first ACF peak, unless the second peak is higher: "
              "with two active longitudes the peak at half the period is the lower one.")
        return [first, (p2, LIB_ROT_2), (p3, LIB_ROT_3)]
    raise ValueError(fam)


def grade(cfg: dict, ans: dict) -> bool:
    keys = cfg["grader"]["config"]["keys"]
    return all(abs(ans[k] - cfg["ground_truth"][k]) <= spec["absolute_tolerance"] for k, spec in keys.items())


def make(cfg: dict, inst_dir: Path, fam: str, style: str = "verbatim") -> dict | None:
    sb = Sandbox(data_dir=inst_dir / "data_files", timeout=300)
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(cfg["task"], sb.listing())}]
        obs = {}
        for j, (prose, code) in enumerate(moves(fam) if style == "verbatim" else moves_lib(fam)):
            res = sb.run(code)
            if res.returncode != 0 or res.timed_out:
                return None
            text = prose(obs) if callable(prose) else prose
            msgs.append({"role": "assistant", "content": f"{text}\n\n```python\n{code}\n```"})
            msgs.append({"role": "user", "content": res.as_feedback()})
            if j == 0:
                obs = _obs(res.stdout)
        hit = FINAL_RE[fam].search(msgs[-1]["content"])
        if not hit:
            return None
        ans = {k: float(v) for k, v in zip(ANSWER_KEYS[fam], hit.groups())}
        if not grade(cfg, ans):
            return None                                  # never ship a wrong-answer trace
        msgs.append({"role": "assistant", "content": "```json\n" + json.dumps({"answer": ans}, indent=2) + "\n```"})
        return {"id": cfg["id"], "messages": msgs, "reward": 1.0, "n_cells": sb.attempts, "_family": f"astro_{fam}"}
    finally:
        sb.cleanup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=list(FAMILIES) + ["all"])
    ap.add_argument("--n", type=int, default=150, help="traces per family")
    ap.add_argument("--val", type=int, default=10, help="validation traces per family")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", default=None, help="generated-instance root (default $DEEP_DATA_ROOT/generated)")
    ap.add_argument("--workers", type=int, default=1, help="instances traced in parallel")
    ap.add_argument("--style", default="verbatim", choices=["verbatim", "lib"],
                    help="verbatim: reference source embedded (default); lib: astropy/batman/scipy demonstrations")
    ap.add_argument("--family-files", default=None, metavar="SUFFIX",
                    help="also write <dir>/<_family><SUFFIX>_{train,val}.jsonl per family (one-demonstration arms)")
    args = ap.parse_args()
    gen = Path(args.gen) if args.gen else data_root() / "generated"
    fams = FAMILIES if args.family == "all" else (args.family,)
    tr_all, va_all = [], []
    for fam in fams:
        paths = sorted((gen / f"astro_{fam}" / "train").glob("*/eval_config.json"))
        traces, bad = [], 0
        with ThreadPoolExecutor(max(1, args.workers)) as ex:
            for i0 in range(0, len(paths), max(1, args.workers)):
                if len(traces) >= args.n + args.val:
                    break
                batch = paths[i0:i0 + max(1, args.workers)]
                for p, t in zip(batch, ex.map(lambda q: make(json.loads(q.read_text()), q.parent, fam, args.style), batch)):
                    if t is None:
                        bad += 1
                        if bad <= 3:
                            print(f"  discarded {p.parent.name}", flush=True)
                    elif len(traces) < args.n + args.val:
                        traces.append(t)
        print(f"{fam}: {len(traces)} traces ({bad} discarded)", flush=True)
        nv = min(args.val, max(len(traces) - 1, 0))
        tr_all += traces[:len(traces) - nv]
        va_all += traces[len(traces) - nv:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}_train.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tr_all))
    Path(f"{out}_val.jsonl").write_text("".join(json.dumps(t) + "\n" for t in va_all))
    print(f"wrote {len(tr_all)} train / {len(va_all)} val to {out}_*.jsonl")
    if args.family_files is not None:                  # per-family splits, as experiments/third_domain/gen.sh
        for split, rows in (("train", tr_all), ("val", va_all)):
            for fam in sorted({r["_family"] for r in rows}):
                Path(out.parent / f"{fam}{args.family_files}_{split}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows if r["_family"] == fam))
        print(f"wrote per-family files {out.parent}/<family>{args.family_files}_{{train,val}}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
