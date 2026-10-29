#!/usr/bin/env python
"""Third scientific domain: time-domain astronomy on Kepler light curves, three families.

Simulated half of the domain; the real half (published catalogues, real PDCSAP photometry)
imports this module's interface -- FAMILIES, ANSWER_KEYS, LC_FILE, STAR_FILE, task_text,
reference, trap_answers, tolerance -- so both halves ask in the same words and are graded
by the same analysis. Conventions follow generators/causal_effects.py exactly:

  * the ground truth is the REFERENCE ANALYSIS APPLIED TO THE SAMPLED LIGHT CURVE, not the
    latent parameter (recorded in _provenance.latent); the reference must also land within
    tolerance of the latent value or the draw is redrawn;
  * every documented wrong analysis (a trap) must land outside tolerance -- a dead trap
    strengthens its knob and re-draws; every accepted variant must land inside it;
  * the ablation table (answer keys, the pipeline step the wrong analysis skips, deviation)
    is stored in `_provenance.ablations`, so rl/reward.py's step-attributed credit works
    unchanged. With two answer keys, `deviation` is in UNITS OF TOLERANCE, maxed over keys
    (gwas_loci's convention).

Data: two consecutive Kepler quarters (~180 d, 29.4-min long cadence, ~8,300 points after
downlink gaps, random gaps and flagged cadences), flux normalised to median 1 per quarter,
white noise 50-600 ppm plus granulation red noise, thermal-settling ramps after every gap and
slow PDCSAP residual drifts. Forward models in generators/astro_lcmodel.py (quadratic
limb-darkened occultations by numerical disk integration, supersampled over the 29.4-min
integration); reference analyses in generators/astro_analysis.py (numpy/scipy only, embedded
verbatim in the demonstration traces).

  family   answer                     reference analysis                    traps (fails_at_step)
  transit  period, rp_rs              running-median detrend -> BLS ->       no detrending (1),
                                      local out-of-transit polynomial ->     no limb darkening,
                                      LSQ fit of (t0,P,k,a/R*,b), given      rp_rs = sqrt(depth) (3),
                                      u1,u2, 11x supersampled                no supersampling (4)*
  eb       period, secondary_phase    iterated eclipse-masked detrend ->     half-period alias (2),
                                      BLS -> event depths grouped by epoch   BLS P/3 alias (2)*, phase
                                      mod m (odd/even; m<=4) ->              measured from the
                                      primary = deeper eclipse -> dip-fit    secondary (3)*, secondary
                                      timing, O-C period refinement          assumed at phase 0.5 (4)*
  rot      rotation_period            McQuillan ACF: uniform grid, gaps = 0, highest Lomb-Scargle
                                      Gaussian smoothing (sigma 9 lags),     peak (2)*, first ACF peak
                                      first peak unless the second is higher when the second is higher (3)*

  * conditional traps: live only on some draws (a short transit, an eccentric orbit, a star
    with two active longitudes). Where a conditional trap coincides with the reference it is
    recorded with fails_at_step None (an accepted variant on that instance). Each draw still
    carries its required traps -- transit: no_detrending and no_limb_darkening always;
    eb: half_period always, plus swapped/phase-0.5 on eccentric draws; rot: the Lomb-Scargle
    trap on two-longitude draws (70%; the first-ACF-peak trap is live where the half-period ACF
    peak survives the smoothing); single-longitude rot draws are deliberate controls, so that
    "double the naive period" is not a shortcut the family rewards.

Tolerances (fixed): period 0.5% (transit, eb); rp_rs 5%; secondary_phase 0.01; rotation 10%.

  venv/bin/python generators/astro_kepler.py --family transit --split train --n 400
  venv/bin/python generators/astro_kepler.py --family all --split train --mix   # RL pool astro_mix
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generators import astro_analysis as A          # noqa: E402
from generators import astro_lcmodel as M           # noqa: E402

ANSWER_KEYS = {"transit": ["period", "rp_rs"], "eb": ["period", "secondary_phase"], "rot": ["rotation_period"]}
LC_FILE = "lightcurve.csv"
STAR_FILE = "star.csv"
STEPS = {"transit": {1: "detrending", 2: "transit search", 3: "limb-darkened model", 4: "integration time"},
         "eb": {1: "detrending", 2: "period alias (odd/even, P/n)", 3: "eclipse identification", 4: "secondary timing"},
         "rot": {1: "uniform grid", 2: "period method", 3: "ACF peak selection"}}
# traps every draw must carry live; conditional ones are added per draw by the generator
CORE_TRAPS = {"transit": ("no_detrending", "no_limb_darkening"), "eb": ("half_period",), "rot": ()}

# ----------------------------------------------------------------------------- task text
_LC_DESC = (f"- data_files/{LC_FILE}: Kepler long-cadence photometry from the PDCSAP pipeline. Columns: time "
            "(BKJD = BJD - 2454833, days), flux (normalised to a median of 1 within each quarter), flux_err, and "
            "quarter. Each point is the average flux over a 29.4-minute integration. The light curve has gaps "
            "between quarters and at data downlinks, and besides the astrophysical signal it contains stellar "
            "variability, correlated noise and instrumental residuals.")


def task_text(family: str, star: dict | None = None) -> str:
    """The prompt, identical for generated and real instances (`star` is accepted so a caller can
    pass the star record; the text points to the file rather than quoting it)."""
    if family == "transit":
        return ("You are an expert in exoplanet photometry. You will analyse the Kepler light curve of a star that "
                "hosts a transiting planet.\n\nData files:\n" + _LC_DESC + "\n"
                f"- data_files/{STAR_FILE}: one row describing the host star: effective temperature teff (K), surface "
                "gravity logg (cgs), metallicity feh (dex), and the star's quadratic limb-darkening coefficients in "
                "the Kepler band, u1 and u2, defined by I(mu)/I(1) = 1 - u1 (1 - mu) - u2 (1 - mu)^2.\n\n"
                "Question: what are the planet's orbital period and its radius ratio Rp/R* (planet radius divided "
                "by stellar radius)? Assume a circular orbit.\n\n"
                'Return your answer as JSON: {"answer": {"period": <days>, "rp_rs": <ratio>}}')
    if family == "eb":
        return ("You are an expert in stellar photometry. You will analyse the Kepler light curve of a detached "
                "eclipsing binary.\n\nData files:\n" + _LC_DESC + "\n\n"
                "The binary shows two eclipses per orbit: the primary eclipse (the deeper one) and the secondary "
                "eclipse. The orbit may be eccentric, in which case the secondary eclipse does not fall half an "
                "orbit after the primary.\n\n"
                "Question: what are the orbital period, and the phase of the secondary eclipse measured from the "
                "primary eclipse, secondary_phase = ((t_secondary - t_primary) / period) mod 1, a number in [0, 1)?\n\n"
                'Return your answer as JSON: {"answer": {"period": <days>, "secondary_phase": <phase>}}')
    if family == "rot":
        return ("You are an expert in stellar photometry. You will analyse the Kepler light curve of a spotted, "
                "magnetically active star whose brightness is modulated by starspots as it rotates.\n\nData files:\n"
                + _LC_DESC + "\n\n"
                "Starspots grow and decay over a few rotations, and spots can be present at more than one longitude "
                "at the same time.\n\n"
                "Question: what is the star's rotation period?\n\n"
                'Return your answer as JSON: {"answer": {"rotation_period": <days>}}')
    raise ValueError(family)


def tolerance(family: str, gold: dict) -> dict:
    if family == "transit":
        return {"period": 0.005 * abs(gold["period"]), "rp_rs": 0.05 * abs(gold["rp_rs"])}
    if family == "eb":
        return {"period": 0.005 * abs(gold["period"]), "secondary_phase": 0.01}
    if family == "rot":
        return {"rotation_period": 0.10 * abs(gold["rotation_period"])}
    raise ValueError(family)


TOL_RULE = {"transit": "period 0.5% of gold, rp_rs 5% of gold",
            "eb": "period 0.5% of gold, secondary_phase 0.01 absolute",
            "rot": "rotation_period 10% of gold"}

# ----------------------------------------------------------------------------- light-curve skeleton
# approximate quarter start times (BKJD) of the Kepler prime mission
QSTART = {1: 131.51, 2: 169.76, 3: 260.23, 4: 352.39, 5: 443.51, 6: 539.46, 7: 630.20, 8: 735.37, 9: 808.54,
          10: 906.84, 11: 1001.23, 12: 1099.43, 13: 1182.76, 14: 1274.16, 15: 1373.51, 16: 1472.14, 17: 1559.25}


def time_grid(rng):
    """Two consecutive quarters at long cadence with downlink gaps, random gaps and flagged
    cadences removed. Returns (t, quarter, segment_starts)."""
    q = int(rng.integers(2, 16))
    ts, qs, starts = [], [], []
    for qq in (q, q + 1):
        t_a = QSTART[qq] + rng.uniform(0.0, 0.3)
        t_b = QSTART[qq + 1] - rng.uniform(1.0, 3.0)
        t = np.arange(t_a, t_b, M.CADENCE)
        keep = np.ones(len(t), bool)
        L = t_b - t_a
        gaps = [(t_a + L * fr + rng.uniform(-3, 3), rng.uniform(0.5, 1.0)) for fr in (1 / 3, 2 / 3)]   # downlinks
        gaps += [(rng.uniform(t_a, t_b), rng.uniform(0.1, 0.5)) for _ in range(rng.poisson(3))]
        for g0, gl in gaps:
            keep &= ~((t >= g0) & (t < g0 + gl))
        keep &= rng.random(len(t)) > rng.uniform(0.01, 0.04)                                # flagged cadences
        t = t[keep]
        ts.append(t)
        qs.append(np.full(len(t), qq))
        starts.append(t[0])
        starts += list(t[1:][np.diff(t) > 0.1])
    return np.concatenate(ts), np.concatenate(qs), np.array(starts)


def ou_noise(rng, t, sigma, tau):
    """Ornstein-Uhlenbeck (granulation-like) red noise sampled on the full cadence grid."""
    n = int(np.ceil((t.max() - t.min()) / M.CADENCE)) + 2
    a = np.exp(-M.CADENCE / tau)
    e = rng.normal(0, sigma * np.sqrt(1 - a * a), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x[np.round((t - t.min()) / M.CADENCE).astype(int)]


def smooth_process(rng, t, tau, n_pad=5):
    """Zero-mean, unit-variance smooth random process with correlation time ~tau (days)."""
    step = max(tau / 8, 0.05)
    g = np.arange(t.min() - n_pad * tau, t.max() + n_pad * tau + step, step)
    from scipy.ndimage import gaussian_filter1d
    w = gaussian_filter1d(rng.normal(size=len(g)), tau / step / 2)
    w = (w - w.mean()) / (w.std() + 1e-12)
    return lambda tt: np.interp(tt, g, w)


def systematics(rng, t, quarter, starts, scale=1.0):
    """PDCSAP residuals: exponential thermal-settling ramps after every gap and a slow
    quadratic drift per quarter."""
    s = np.zeros_like(t)
    for t0 in starts:
        m = t >= t0
        s[m] += rng.choice([-1, 1]) * rng.uniform(30, 400) * 1e-6 * scale * np.exp(-(t[m] - t0) / rng.uniform(0.2, 1.2))
    for qq in np.unique(quarter):
        m = quarter == qq
        x = (t[m] - t[m].mean()) / (np.ptp(t[m]) / 2)
        s[m] += np.polyval(rng.normal(0, 150e-6 * scale, 3), x)
    return s


def spots_random(rng, t, prot, n, lifetime):
    """n spots at random longitudes/latitudes with log-normal area evolution."""
    out = []
    for _ in range(n):
        proc = smooth_process(rng, t, lifetime)
        sd = rng.uniform(0.3, 0.9)
        area = rng.uniform(0.3, 1.0)
        out.append(dict(lon=rng.uniform(0, 2 * np.pi), lat=np.arcsin(rng.uniform(-0.7, 0.7)), contrast=0.7,
                        drift=rng.normal(0, 0.01), area=(lambda tt, p=proc, s=sd, a=area: a * np.exp(s * p(tt) - s * s / 2))))
    return out


def stellar_variability(rng, t, amp, prot, n_spots=None, lifetime=None):
    """Spot modulation scaled to a peak-to-peak amplitude ~ amp (relative flux)."""
    n_spots = n_spots or int(rng.integers(1, 4))
    lifetime = lifetime or rng.uniform(1.5, 5.0) * prot
    inc = np.arccos(rng.uniform(0.0, 0.9))
    sp = spots_random(rng, t, prot, n_spots, lifetime)
    f = M.spot_flux(t, prot, inc, sp)
    v = f - np.median(f)
    return amp * v / (np.percentile(v, 95) - np.percentile(v, 5) + 1e-12)


def finish(rng, t, quarter, starts, signal, sigma_w, red=None, sys_scale=1.0, cosmic=True):
    """Observed PDCSAP-like flux: signal x (1 + variability) + noise, normalised per quarter."""
    red = red if red is not None else ou_noise(rng, t, rng.uniform(10, 60) * 1e-6, rng.uniform(0.05, 0.3))
    flux = signal + red + systematics(rng, t, quarter, starts, sys_scale) + rng.normal(0, sigma_w, len(t))
    if cosmic:
        hit = rng.random(len(t)) < 5e-4
        flux[hit] += rng.uniform(3, 10, hit.sum()) * sigma_w
    err = np.empty_like(t)
    for qq in np.unique(quarter):
        m = quarter == qq
        med = np.median(flux[m])
        flux[m] /= med
        err[m] = sigma_w * rng.uniform(0.9, 1.05) * (1 + 0.02 * rng.normal(size=m.sum())) / med
    return pd.DataFrame({"time": np.round(t, 6), "flux": np.round(flux, 7), "flux_err": np.round(err, 7),
                         "quarter": quarter.astype(int)})


def log_uniform(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


# ----------------------------------------------------------------------------- families (DGPs)
def limb_darkening(rng, teff):
    """Kepler-band quadratic coefficients following the Claret (2011) trend with Teff."""
    u1 = 0.40 - 1.1e-4 * (teff - 5800) + rng.normal(0, 0.02)
    u2 = 0.26 + 6.0e-5 * (teff - 5800) + rng.normal(0, 0.02)
    return round(float(np.clip(u1, 0.1, 0.8)), 4), round(float(np.clip(u2, 0.0, 0.45)), 4)


SNR_MIN = 300.0      # transit white-noise S/N floor (see gen_transit)
VAR_MAX = 0.03       # peak-to-peak stellar variability cap for transit hosts
RPRS_ERR_MAX = 0.25  # the reference's linearised sigma(rp_rs) must be < this x tolerance


def gen_transit(rng, k_var: float):
    """Planet on a circular orbit around a main-sequence/slightly evolved star; k_var scales
    the stellar variability (the no-detrending trap's knob)."""
    for _ in range(400):
        t, quarter, starts = time_grid(rng)
        teff = rng.uniform(4300, 6600)
        mass = float(np.clip((teff / 5772) ** 1.8 * rng.uniform(0.9, 1.1), 0.6, 1.5))
        rad = mass ** 0.85 * rng.uniform(0.9, 1.35)
        logg = 4.438 + np.log10(mass) - 2 * np.log10(rad)
        rho = mass / rad ** 3
        u1, u2 = limb_darkening(rng, teff)
        P = log_uniform(rng, 0.8, 30.0)
        k = rng.uniform(0.03, 0.2)
        b = rng.uniform(0.0, 0.85)
        aRs = 4.2064 * P ** (2 / 3) * rho ** (1 / 3)
        if aRs < 2.0 or b + k > 0.97:
            continue
        t0 = t.min() + rng.uniform(0, P)
        clean = M.transit_flux(t, t0, P, k, aRs, b, u1, u2, n_sub=21)
        depth = 1 - clean.min()
        dur = A.t14(P, k, aRs, b)
        ep = np.round((t - t0) / P)
        n_tr = len(np.unique(ep[clean < 1 - 0.5 * depth]))
        n_in = int(np.sum(clean < 1 - 0.5 * depth))
        if n_tr < 3 or dur < 2 * M.CADENCE:
            continue
        # cheap pre-filter for the no-limb-darkening trap (sqrt of the central depth): its bias is
        # ~+10% at b = 0 and vanishes near b ~ 0.7, where the chord samples the disk-mean brightness;
        # make() then requires it outside the 5% tolerance on the data, so live draws have b <~ 0.55
        if abs(np.sqrt(depth) / k - 1) < 0.065:
            continue
        # white-noise S/N of the transit, depth sqrt(n_in) / sigma >= SNR_MIN: with 29.4-min
        # integrations the ingress is unresolved and b trades off against k, so rp_rs needs
        # far more S/N than detection does
        s_max = depth * np.sqrt(n_in) / SNR_MIN
        if s_max < 50e-6:
            continue
        sigma_w = log_uniform(rng, 50e-6, min(600e-6, s_max))
        prot = log_uniform(rng, 3.0, 35.0)
        amp = min(log_uniform(rng, 5e-4, 1.5e-2) * k_var, VAR_MAX)
        var = stellar_variability(rng, t, amp, prot)
        # knob: the variability left in must bias a global-baseline depth (a realistic cap of
        # 3% peak-to-peak; a draw that cannot reach the bias is redrawn)
        in_tr = clean < 1 - 0.5 * depth
        bias = abs(np.mean(var[in_tr]) - np.median(var)) / (2 * depth)
        while bias < 0.09 and amp * 1.4 <= VAR_MAX:
            var, amp = var * 1.4, amp * 1.4
            bias *= 1.4
        if bias < 0.09:
            continue
        lc = finish(rng, t, quarter, starts, clean * (1 + var), sigma_w)
        star = {"teff": round(float(teff)), "logg": round(float(logg), 3), "feh": round(float(rng.normal(0, 0.2)), 2),
                "u1": u1, "u2": u2}
        latent = {"period": P, "rp_rs": k, "t0": t0, "a_rs": aRs, "b": b, "t14": dur, "u1": u1, "u2": u2,
                  "depth": depth, "sigma_w": sigma_w, "var_amp": amp, "prot_star": prot, "n_transits": n_tr,
                  "mass": mass, "radius": rad}
        return lc, star, latent, set(CORE_TRAPS["transit"])
    raise RuntimeError("gen_transit: no acceptable draw")


SB_SHALLOW = 0.30       # share of EB draws with a faint companion (depth ratio ~0.03-0.3, log-uniform)
SNR_SEC_MIN = 20.0     # the secondary eclipse's white-noise S/N floor


def gen_eb(rng, k: float):
    """Detached binary of two limb-darkened spheres; eccentric on ~60% of draws.

    The secondary/primary depth ratio tracks the surface-brightness ratio sb, drawn so that
    ~30% of binaries have a faint, cool, low-mass companion (sb log-uniform in 0.03-0.3; the
    real Kepler set has 31% of its depth ratios below 0.3) and the rest sb uniform in 0.3-0.97.
    The companion's mass follows sb (q ~ sb^0.25) and its temperature T2 ~ T1 sb^0.2. The
    white noise is capped so the secondary eclipse keeps S/N >= SNR_SEC_MIN."""
    for _ in range(400):
        t, quarter, starts = time_grid(rng)
        P = log_uniform(rng, 0.8, 20.0)
        sb = log_uniform(rng, 0.03, 0.3) if rng.random() < SB_SHALLOW else rng.uniform(0.3, 0.97)
        m1 = rng.uniform(0.7, 1.6)
        q = float(np.clip(sb ** 0.25 * rng.uniform(0.85, 1.15), 0.25, 1.0))
        R1, R2 = m1 ** 0.8 * rng.uniform(0.9, 1.25), (q * m1) ** 0.8 * rng.uniform(0.9, 1.1)
        a = 4.2064 * (m1 * (1 + q)) ** (1 / 3) * P ** (2 / 3)
        r1, r2 = R1 / a, R2 / a
        if r1 + r2 > 0.45:
            continue
        ecc_draw = rng.random() < (0.2 if P < 1.5 else 0.65)
        if ecc_draw:
            e = rng.uniform(0.03, min(0.45, 0.08 + 0.1 * P))
            w = rng.uniform(0, 2 * np.pi)
        else:
            e, w = 0.0, rng.uniform(0, 2 * np.pi)
        # both conjunctions must eclipse, neither grazing
        rc = [(1 - e * e) / (1 + e * np.sin(w)), (1 - e * e) / (1 - e * np.sin(w))]
        cosi_max = 0.8 * (r1 + r2) / max(rc)
        inc = np.arccos(rng.uniform(0, cosi_max))
        teff1 = rng.uniform(5200, 7000)
        ld1 = limb_darkening(rng, teff1)
        ld2 = limb_darkening(rng, teff1 * sb ** 0.2)
        tperi = t.min() + rng.uniform(0, P)
        tc = M.conjunction_times(P, tperi, e, w, inc)
        dd = [1 - M.eb_flux(np.array([x]), P, tperi, e, w, inc, r1, r2, sb, ld1, ld2, n_sub=1)[0] for x in tc]
        ipri = int(np.argmax(dd))
        ratio = dd[1 - ipri] / dd[ipri]
        phase = float(np.mod((tc[1 - ipri] - tc[ipri]) / P, 1.0))
        if not (0.03 <= ratio <= 0.97) or dd[ipri] < 0.005:
            continue
        if ecc_draw and not (0.02 <= abs(phase - 0.5) <= 0.22):
            continue
        clean = M.eb_flux(t, P, tperi, e, w, inc, r1, r2, sb, ld1, ld2, n_sub=15)
        ep = np.round((t - tc[ipri]) / P)
        n_ecl = len(np.unique(ep[np.abs(A.fold(t, P, tc[ipri])) < 0.02 * P]))
        if n_ecl < 3:
            continue
        # white noise capped so the secondary keeps S/N = depth sqrt(n_in) / sigma >= SNR_SEC_MIN
        near_sec = np.abs(A.fold(t, P, tc[1 - ipri])) < 0.1 * P
        n_sec = int(np.sum(near_sec & (1 - clean > 0.5 * dd[1 - ipri])))
        s_max = dd[1 - ipri] * np.sqrt(max(n_sec, 1)) / SNR_SEC_MIN
        if n_sec < 3 or s_max < 50e-6:
            continue
        sigma_w = log_uniform(rng, 50e-6, min(600e-6, s_max))
        ph = 2 * np.pi * (t - tc[ipri]) / P
        ell = -rng.uniform(0.2, 0.6) * (r1 ** 3 + r2 ** 3) * np.cos(2 * ph) - rng.uniform(0, 0.3) * (r1 ** 2) * 0.1 * np.cos(ph)
        var = stellar_variability(rng, t, log_uniform(rng, 2e-4, 4e-3), log_uniform(rng, 1.0, 30.0))
        lc = finish(rng, t, quarter, starts, clean * (1 + ell + var), sigma_w)
        latent = {"period": P, "secondary_phase": phase, "t_primary": tc[ipri], "e": e, "omega": w,
                  "inc_deg": float(np.degrees(inc)), "r1": r1, "r2": r2, "sb_ratio": sb, "depth_primary": dd[ipri],
                  "depth_ratio": ratio, "eccentric": bool(ecc_draw), "sigma_w": sigma_w, "q": q,
                  "snr_secondary": float(dd[1 - ipri] * np.sqrt(n_sec) / sigma_w)}
        req = set(CORE_TRAPS["eb"]) | ({"phase_half", "swapped_eclipses"} if ecc_draw else set())
        return lc, None, latent, req
    raise RuntimeError("gen_eb: no acceptable draw")


def gen_rot(rng, k: float):
    """Spotted star; on 70% of draws two active longitudes roughly half a turn apart."""
    t, quarter, starts = time_grid(rng)
    P = log_uniform(rng, 1.0, 40.0)
    double = rng.random() < 0.7
    # two active longitudes only read as a double dip when both spots are seen at a similar
    # foreshortening: comparable latitudes on a fairly inclined star
    inc = np.arccos(rng.uniform(0.0, 0.6 if double else 0.85))
    life = rng.uniform(2.0, 6.0) * P
    lon0 = rng.uniform(0, 2 * np.pi)
    lat0 = rng.uniform(-0.5, 0.5)
    longs = [(lon0, 1.0, lat0)]
    if double:
        longs.append((lon0 + np.pi + rng.normal(0, 0.2), rng.uniform(0.5, 0.9), lat0 + rng.normal(0, 0.1)))
    elif rng.random() < 0.5:
        longs.append((lon0 + rng.uniform(1.0, 5.3), rng.uniform(0.05, 0.3), rng.uniform(-0.6, 0.6)))
    spots = []
    drift0 = rng.normal(0, 0.004)                  # active longitudes in one latitude band share a rate
    for lon, rel, lat in longs:
        proc = smooth_process(rng, t, life)
        sd = rng.uniform(0.15, 0.35) if double else rng.uniform(0.2, 0.6)
        drift = drift0 + rng.normal(0, 0.0015 if double else 0.006)
        spots.append(dict(lon=lon, lat=lat, contrast=0.7, drift=drift,
                          area=(lambda tt, p=proc, s=sd, a=rel: a * np.exp(s * p(tt) - s * s / 2))))
    f = M.spot_flux(t, P, inc, spots)
    v = f - np.median(f)
    amp = log_uniform(rng, 1e-3, 2e-2)
    v = amp * v / (np.percentile(v, 95) - np.percentile(v, 5) + 1e-12)
    sigma_w = log_uniform(rng, 50e-6, 600e-6)
    red = ou_noise(rng, t, rng.uniform(20, 80) * 1e-6, rng.uniform(0.05, 0.5))
    lc = finish(rng, t, quarter, starts, 1 + v, sigma_w, red=red, sys_scale=min(1.0, amp / 3e-3))
    latent = {"rotation_period": P, "double_dip": bool(double), "inc_deg": float(np.degrees(inc)),
              "spot_lifetime": life, "amplitude": amp, "sigma_w": sigma_w,
              "longitudes": [[float(l), float(r), float(b)] for l, r, b in longs]}
    # the periodogram's half-period peak is the defining trap of a two-longitude star; the ACF's
    # half-period peak is smoothed away for short periods, so that trap stays conditional
    req = {"ls_peak"} if double else set()
    return lc, None, latent, req


FAMILIES = {"transit": gen_transit, "eb": gen_eb, "rot": gen_rot}


# ----------------------------------------------------------------------------- reference analyses
def _arrays(lc: pd.DataFrame):
    d = lc[["time", "flux"]].astype(float)
    d = d[np.isfinite(d["time"]) & np.isfinite(d["flux"])].sort_values("time")
    return d["time"].to_numpy(), d["flux"].to_numpy()


def analyse_transit(t, f, u1, u2) -> dict:
    ok = A.clip_high(t, f)
    t, f = t[ok], f[ok]
    fd = f / A.running_median(t, f, 1.0)                         # search detrend
    _, _, bb = A.bls(t, fd)
    tw, fw = A.transit_windows(t, f, bb["period"], bb["t0"], bb["duration"])
    f1 = A.fit_transit(tw, fw, bb["period"], bb["t0"], np.sqrt(max(bb["depth"], 1e-6)), bb["duration"], u1, u2,
                       b_starts=(0.2, 0.7))
    T = max(f1["t14"], 2 * A.CAD)
    tw, fw = A.transit_windows(t, f, f1["period"], f1["t0"], T)
    ref = A.fit_transit(tw, fw, f1["period"], f1["t0"], f1["rp_rs"], T, u1, u2, b_starts=(f1["b"], 0.1, 0.5, 0.8))
    T = ref["t14"]
    starts = (ref["b"],)
    # traps
    dt = A.fold(tw, ref["period"], ref["t0"])
    core = np.abs(dt) < T / 6
    depth = 1 - np.mean(fw[core])
    no_ld = {"period": ref["period"], "rp_rs": float(np.sqrt(max(depth, 0.0)))}
    sel = np.isin(t, tw)                                          # same windows, flux as delivered
    raw = A.fit_transit(t[sel], f[sel], ref["period"], ref["t0"], ref["rp_rs"], T, u1, u2, b_starts=starts)
    nos = A.fit_transit(tw, fw, ref["period"], ref["t0"], ref["rp_rs"], T, u1, u2, nsub=1, b_starts=starts)
    tw2, fw2 = A.transit_windows(t, f, ref["period"], ref["t0"], T, side=1.5 * max(2 * T, 0.35))
    wide = A.fit_transit(tw2, fw2, ref["period"], ref["t0"], ref["rp_rs"], T, u1, u2, b_starts=starts)
    ans = lambda r: {"period": float(r["period"]), "rp_rs": float(r["rp_rs"])}
    return {"answer": ans(ref),
            "traps": {"no_detrending": (ans(raw), 1), "no_limb_darkening": (no_ld, 3), "no_supersampling": (ans(nos), 4),
                      "wider_baseline_window": (ans(wide), None)},
            "diag": {"bls": bb, "fit": {k: float(v) for k, v in ref.items()}, "central_depth": float(depth),
                     "n_transit_points": int(core.sum())}}


def analyse_eb(t, f) -> dict:
    ok = A.clip_high(t, f)
    t, f = t[ok], f[ok]
    fd = A.eb_detrend(t, f)
    _, _, bb = A.bls(t, fd)
    sol = A.eb_solve(t, fd, bb["period"], bb["t0"], bb["duration"])
    unref = A.eb_solve(t, fd, bb["period"], bb["t0"], bb["duration"], refine=False)
    P, ph = sol["period"], sol["secondary_phase"]
    traps = {"half_period": ({"period": P / 2, "secondary_phase": 0.5}, 2),
             "swapped_eclipses": ({"period": P, "secondary_phase": float(np.mod(1 - ph, 1.0))}, 3),
             "phase_half": ({"period": P, "secondary_phase": 0.5}, 4),
             "unrefined_period": ({"period": unref["period"], "secondary_phase": unref["secondary_phase"]}, None)}
    if sol["multiple"] >= 3:                       # BLS stacked eclipses at P/3 or P/4: its period as reported
        traps["bls_alias"] = ({"period": sol["p_bls"], "secondary_phase": 0.5}, 2)
    return {"answer": {"period": P, "secondary_phase": ph}, "traps": traps,
            "diag": {"bls": bb, **{k: v for k, v in sol.items() if k not in ("period", "secondary_phase", "groups")}}}


def analyse_rot(t, f) -> dict:
    ok = A.clip_high(t, f)
    t, f = t[ok], f[ok]
    P, first, pk = A.acf_period(t, f)
    ls = A.ls_peak(t, f)
    series = A.acf_period_series(pk, P)
    return {"answer": {"rotation_period": P},
            "traps": {"ls_peak": ({"rotation_period": ls}, 2), "first_acf_peak": ({"rotation_period": first}, 3),
                      "acf_peak_series": ({"rotation_period": series}, None)},
            "diag": {"acf_peaks": [(float(a), float(b)) for a, b in pk[:6]]}}


_CACHE: dict = {}


def _analyse(family: str, lc: pd.DataFrame, star: dict | None) -> dict:
    t, f = _arrays(lc)
    key = (family, hashlib.sha1(t.tobytes() + f.tobytes()).hexdigest(),
           None if star is None else (float(star["u1"]), float(star["u2"])))
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        if family == "transit":
            _CACHE[key] = analyse_transit(t, f, float(star["u1"]), float(star["u2"]))
        elif family == "eb":
            _CACHE[key] = analyse_eb(t, f)
        elif family == "rot":
            _CACHE[key] = analyse_rot(t, f)
        else:
            raise ValueError(family)
    return _CACHE[key]


def reference(family: str, lc: pd.DataFrame, star: dict | None = None) -> dict:
    """The correct analysis -> answer dict with ANSWER_KEYS[family]."""
    return dict(_analyse(family, lc, star)["answer"])


def trap_answers(family: str, lc: pd.DataFrame, star: dict | None = None) -> dict:
    """{name: (answer_dict, fails_at_step)} for every documented wrong analysis, with its nominal
    step; entries with step None are accepted variants of the reference (must agree with it)."""
    return {k: (dict(v), s) for k, (v, s) in _analyse(family, lc, star)["traps"].items()}


def deviation_units(family: str, ans: dict, gold: dict, tol: dict | None = None) -> float:
    tol = tol or tolerance(family, gold)
    return float(max(abs(ans[k] - gold[k]) / tol[k] for k in ANSWER_KEYS[family]))


def ablation_table(family: str, lc: pd.DataFrame, star: dict | None = None, gold: dict | None = None,
                   required=None) -> tuple[dict, list, list]:
    """The eval_config ablation table for an instance, plus (weak, loose): required traps that are
    inside tolerance, and accepted variants that are outside it. A conditional trap that
    coincides with the reference is recorded with fails_at_step None."""
    gold = gold or reference(family, lc, star)
    tol = tolerance(family, gold)
    required = set(CORE_TRAPS[family]) if required is None else set(required)
    table = {"reference": {**gold, "fails_at_step": None, "deviation": 0.0}}
    weak, loose = [], []
    for name, (ans, step) in trap_answers(family, lc, star).items():
        dev = deviation_units(family, ans, gold, tol)
        if step is not None and dev <= 1.0:
            if name in required:
                weak.append(name)
            else:
                step = None                                   # coincides with the reference here
        elif step is None and dev > 1.0:
            loose.append(name)
        table[name] = {**ans, "fails_at_step": step, "deviation": dev}
    return table, weak, loose


# ----------------------------------------------------------------------------- make / emit / mix
def make(family: str, seed: int, max_tries: int = 12):
    """Draw until the reference is within tolerance of the latent truth, every required trap is
    live and every accepted variant is inside tolerance. A dead no-detrending trap strengthens
    the stellar-variability knob; anything else just re-draws."""
    rng = np.random.default_rng(seed)
    knob = 1.0
    for tries in range(1, max_tries + 1):
        lc, star, latent, required = FAMILIES[family](rng, knob)
        t0 = time.time()
        gold = reference(family, lc, star)
        tol = tolerance(family, gold)
        insane = [k for k in ANSWER_KEYS[family] if not abs(gold[k] - latent[k]) <= tol[k]]
        if family == "transit":                     # rp_rs must not hinge on where in the b-k valley a fit lands
            if _analyse(family, lc, star)["diag"]["fit"]["rp_rs_err"] > RPRS_ERR_MAX * tol["rp_rs"]:
                insane.append("rp_rs_err")
        table, weak, loose = ablation_table(family, lc, star, gold, required)
        secs = time.time() - t0
        if not insane and not weak and not loose:
            break
        if "no_detrending" in weak:
            knob *= 1.5
    return lc, star, latent, gold, tol, table, weak + loose + [f"sanity:{k}" for k in insane], tries, secs


def _emit_one(job) -> tuple:
    family, split, i, seed, out = job
    lc, star, latent, gold, tol, table, weak, tries, secs = make(family, seed)
    pid = f"astro_{family}_{i:05d}"
    d = out / pid
    (d / "data_files").mkdir(parents=True, exist_ok=True)
    lc.to_csv(d / "data_files" / LC_FILE, index=False)
    files = [f"data_files/{LC_FILE}"]
    if star is not None:
        pd.DataFrame([star]).to_csv(d / "data_files" / STAR_FILE, index=False)
        files.append(f"data_files/{STAR_FILE}")
    cfg = {"id": pid, "task": task_text(family, star), "data_files": files,
           "ground_truth": {k: float(gold[k]) for k in ANSWER_KEYS[family]},
           "grader": {"type": "multi_numeric_tolerance",
                      "config": {"keys": {k: {"absolute_tolerance": float(tol[k])} for k in ANSWER_KEYS[family]}}},
           "_provenance": {"family": f"astro_{family}", "seed": seed, "gen_tries": tries,
                           "tol_rule": TOL_RULE[family],
                           "note": ("ground truth is the reference analysis (generators/astro_analysis.py) on the "
                                    "sampled light curve, not the latent parameter; ablation deviation is in units "
                                    "of tolerance, maxed over the answer keys"),
                           "latent": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                      for k, v in latent.items()},
                           "steps": STEPS[family],
                           "ablations": table}}
    (d / "eval_config.json").write_text(json.dumps(cfg, indent=1, default=float))
    return pid, weak, tries, secs


def emit(family: str, split: str, n: int, seed0: int, out_root: Path, workers: int = 1) -> dict:
    """Instances are seeded by index, so the output does not depend on `workers`."""
    out = out_root / f"astro_{family}" / split
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(family, split, i, seed0 + i, out) for i in range(n)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_emit_one, jobs, chunksize=1)
    else:
        res = [_emit_one(j) for j in jobs]
    weak_ids = [pid for pid, weak, _, _ in res if weak]
    man = {"family": f"astro_{family}", "split": split, "n": n, "seed0": seed0,
           "n_all_traps_valid": n - len(weak_ids), "n_weak": len(weak_ids), "weak_ids": weak_ids,
           "mean_tries": sum(r[2] for r in res) / n, "mean_reference_seconds": sum(r[3] for r in res) / n}
    (out / "_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def selfcheck() -> int:
    """Forward model against exact/brute-force occultations, and the analysis model against the
    independent forward model."""
    worst = M._self_check()
    t = np.linspace(-0.3, 0.3, 3001)
    dmax = 0.0
    for P, k, aRs, b, u1, u2 in [(0.8, 0.2, 3.3, 0.5, 0.55, 0.15), (3.0, 0.1, 9.0, 0.0, 0.4, 0.26),
                                 (12.0, 0.03, 25.0, 0.8, 0.32, 0.3), (1.5, 0.15, 5.0, 0.7, 0.45, 0.2)]:
        for ns in (1, 11):
            d = np.max(np.abs(A.transit_model(t, 0.0, P, k, aRs, b, u1, u2, ns) -
                              M.transit_flux(t, 0.0, P, k, aRs, b, u1, u2, n_sub=ns)))
            dmax = max(dmax, d)
    print(f"analysis transit_model vs forward model (12 vs 32 nodes, same sub-exposures): max |diff| = {dmax:.1e}")
    ok = worst < 1e-7 and dmax < 1e-7
    print("selfcheck:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def mix(split: str, out_root: Path) -> int:
    """generated/astro_mix/SPLIT: symlinks to every family's instances, one RL training pool."""
    out = out_root / "astro_mix" / split
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(FAMILIES):
        for d in sorted((out_root / f"astro_{f}" / split).glob(f"astro_{f}_*")):
            link = out / d.name
            if not link.exists():
                link.symlink_to(d.resolve(), target_is_directory=True)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES) + ["all"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--out", default=None, help="default: $DEEP_DATA_ROOT/generated")
    ap.add_argument("--mix", action="store_true", help="only link the emitted families into astro_mix/SPLIT")
    ap.add_argument("--workers", type=int, default=1, help="processes; the output does not depend on it")
    ap.add_argument("--selfcheck", action="store_true", help="validate the light-curve models and exit")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not args.family:
        ap.error("--family is required")
    from paths import data_root
    out = Path(args.out) if args.out else data_root() / "generated"
    if args.mix:
        print(json.dumps({"astro_mix": args.split, "n": mix(args.split, out)}))
        return 0
    fams = sorted(FAMILIES) if args.family == "all" else [args.family]
    for f in fams:
        # seed block fixed per family (not per loop position), so one family run alone and
        # the same family inside --family all draw identical instances
        j = sorted(FAMILIES).index(f)
        seed0 = args.seed0 if args.seed0 is not None else (9_000_000 + 100_000 * j + (0 if args.split == "train" else 50_000))
        man = emit(f, args.split, args.n, seed0, out, args.workers)
        print(json.dumps({k: v for k, v in man.items() if k != "weak_ids"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
