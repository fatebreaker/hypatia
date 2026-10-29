"""Signal physics: natural frequency and quality factor of a damped oscillator from a ring-down (a SAGE plug-in).

The data are the free decay of a lab oscillator after it is displaced and released -- a mass on a
spring, a torsion pendulum, a clamped cantilever or an air-track glider, each with an adjustable
eddy-current damper -- as a data logger records it: one timestamped reading per sample, unevenly
spaced (the logger polls the sensor over a shared bus) with a few drop-outs, a slow drift of the
sensor's zero as it warms up, coloured (Ornstein-Uhlenbeck) plus white noise, and a few
single-sample electrical glitches. The oscillator obeys x'' + (w0/Q) x' + w0^2 x = 0, so it rings
at the damped frequency f_d = f0 sqrt(1 - 1/(4Q^2)) and its amplitude decays as exp(-gamma t) with
gamma = pi f0 / Q.

Reference analysis (the gold, on the sampled data): nonlinear least squares, on the samples as
recorded, of exp(-gamma t) (a cos 2 pi f_d t + b sin 2 pi f_d t) + a quadratic baseline (the
drift, fitted jointly), started from the Lomb-Scargle peak of the detrended series and the best
decay rate on a grid; glitches are handled by a soft-L1 fit first, after which points more than 5
robust SDs off it are dropped and the fit is repeated by plain least squares. Then
f0 = sqrt(f_d^2 + (gamma / 2 pi)^2) and Q = pi f0 / gamma.

Traps (the step each skips; skills from the shared taxonomy) and accepted variants:

  1 qc            no_outlier_removal      plain least squares on every sample, glitches included
  2 normalise     no_detrending           a constant baseline only: the drift is left in the series
  3 model-choice  undamped_sinusoid       a constant-amplitude sinusoid; f from that fit and Q from the
                                          half-power width of the Lomb-Scargle peak (the only damping
                                          information an undamped model leaves), which the finite,
                                          gappy record broadens
  4 estimate      fft_interpolated        cleaned, detrended samples linearly interpolated onto an even
                                          grid, FFT: f from the peak bin, Q = f / half-power width;
                                          leakage and interpolation across the gaps bias both
  5 report        damped_frequency_as_f0  the fitted damped frequency f_d reported as f0 (conditional:
                  (conditional)           live where 1/sqrt(1 - 1/(4Q^2)) - 1 exceeds the f0 tolerance,
                                          i.e. at low Q; step None where it coincides)
  -  variant      soft_l1_fit             the robust soft-L1 fit's own answer, no clipping or refit
  -  variant      cubic_drift             a cubic baseline instead of the quadratic one

Tolerance: f0 max(2 SE, 0.2% of f0, 0.02/T), Q max(2 SE, 3% of Q, 0.2 Q/cycles); SE from the
reference fit's Newey-West (HAC) covariance over about two periods (the noise is coloured), T the
record length and cycles = f0 T. The resolution-scaled floors matter on few-cycle, low-Q records,
where correlated noise moves f0 by up to ~1% and Q by up to ~15% and no residual-based SE sees it
(calibrated on 150 draws: the reference then misses the latent truth on 5-8% of draws at every Q).

Real data: none shipped (checked 2026-09-15): no public ring-down or pendulum record with a published
f0 and Q was found on a stable open source (Rdatasets, NIST StRD, a Zenodo/web search).

The analysis functions are self-contained (numpy/scipy/astropy) so a demonstration embeds their
source verbatim: the trace and the ground truth are the same code.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sage.core import Cell, Family, Sample, Trap

CSV = "ringdown.csv"
NAN = float("nan")


# ----------------------------------------------------------------------------- analysis (embedded in demos)
def ringdown_fit(t, x, deg=2, damped=True, robust=True, clip=5.0):
    """Nonlinear least squares of x(t) = exp(-gamma t) (a cos 2 pi f t + b sin 2 pi f t) + a polynomial
    baseline of degree `deg` in the scaled time (the drift), on the samples as recorded. Start: the
    Lomb-Scargle peak of the detrended series and the best decay rate on a grid (for fixed f and gamma
    the model is linear in the rest). Glitches: a soft-L1 fit, then points more than `clip` robust SDs
    off it are dropped and the fit is repeated by least squares (clip=None: stop at the soft-L1 fit;
    robust=False: plain least squares on every point). Returns (f, gamma, kept-point mask, result)."""
    from astropy.timeseries import LombScargle
    from scipy.optimize import least_squares
    t = t - t[0]
    u = t / t[-1]

    def X(f, g, m):
        e, w = np.exp(-g * t[m]), 2 * np.pi * f * t[m]
        return np.column_stack([e * np.cos(w), e * np.sin(w)] + [u[m] ** j for j in range(deg + 1)])

    def res(p, m):
        return X(p[0], p[1] if damped else 0.0, m) @ p[2:] - x[m]

    def mad(r):
        return 1.4826 * np.median(np.abs(r - np.median(r)))

    m = np.ones(len(t), bool)
    detrended = x - np.polyval(np.polyfit(u, x, deg), u)
    freq, power = LombScargle(t, detrended).autopower(maximum_frequency=0.5 / np.median(np.diff(t)),
                                                      samples_per_peak=20)
    f = freq[np.argmax(power)]
    lin = lambda g: np.linalg.lstsq(X(f, g, m), x, rcond=None)[0]
    g = min(np.pi * f / np.geomspace(1, 300, 40), key=lambda g: np.sum((X(f, g, m) @ lin(g) - x) ** 2)) \
        if damped else 0.0
    p = np.r_[f, g, lin(g)]
    if robust:
        p = least_squares(res, p, args=(m,), loss="soft_l1", f_scale=mad(res(p, m)), x_scale="jac").x
        if clip is None:
            return p[0], p[1], m, None
        m = np.abs(res(p, m) - np.median(res(p, m))) < clip * mad(res(p, m))
    sol = least_squares(res, p, args=(m,), x_scale="jac")
    return sol.x[0], sol.x[1], m, sol


def natural_frequency_q(f_d, gamma):
    """x'' + (w0/Q) x' + w0^2 x = 0 rings at w_d = w0 sqrt(1 - 1/(4 Q^2)) and decays as exp(-gamma t),
    gamma = w0 / (2Q): so f0 = sqrt(f_d^2 + (gamma / 2 pi)^2) and Q = pi f0 / gamma."""
    f0 = np.hypot(f_d, gamma / (2 * np.pi))
    return float(f0), float(np.pi * f0 / gamma)


# ----------------------------------------------------------------------------- the wrong analyses
def fit_se(sol, lags):
    """SEs of (f0, Q) from the fit's Newey-West (HAC) sandwich covariance -- the noise is coloured, so the
    white-noise (J'J)^-1 s^2 understates them -- with Bartlett weights over `lags` neighbouring samples,
    propagated by the delta method. Also returns the residuals' lag-1 autocorrelation (a diagnostic)."""
    J, r = sol.jac, sol.fun
    Jr = J * r[:, None]
    S = Jr.T @ Jr
    for lag in range(1, lags + 1):
        G = Jr[lag:].T @ Jr[:-lag]
        S += (1 - lag / (lags + 1)) * (G + G.T)
    H = np.linalg.pinv(J.T @ J)
    cov = H @ S @ H
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    p = sol.x
    G = np.zeros((2, len(p)))
    for i in range(2):
        h = 1e-6 * max(abs(p[i]), 1e-12)
        up, dn = p.copy(), p.copy()
        up[i] += h
        dn[i] -= h
        G[:, i] = (np.array(natural_frequency_q(up[0], up[1])) - np.array(natural_frequency_q(dn[0], dn[1]))) / (2 * h)
    se = np.sqrt(np.maximum(np.diag(G @ cov @ G.T), 0))
    return float(se[0]), float(se[1]), rho


def half_power_q(freq, power):
    """(peak frequency, peak frequency / full width at half power), crossings interpolated linearly."""
    i = int(np.argmax(power))
    h = power[i] / 2
    lo, hi = i, i
    while lo > 0 and power[lo] > h:
        lo -= 1
    while hi < len(power) - 1 and power[hi] > h:
        hi += 1
    f_lo = np.interp(h, [power[lo], power[lo + 1]], [freq[lo], freq[lo + 1]]) if power[lo] <= h else freq[lo]
    f_hi = np.interp(h, [power[hi], power[hi - 1]], [freq[hi], freq[hi - 1]]) if power[hi] <= h else freq[hi]
    return float(freq[i]), float(freq[i] / max(f_hi - f_lo, 1e-300))


def _detrended(t, x, deg=2):
    u = (t - t[0]) / (t[-1] - t[0])
    return x - np.polyval(np.polyfit(u, x, deg), u)


def undamped_answer(t, x, keep):
    """TRAP (step 3): a constant-amplitude sinusoid + drift by least squares on the cleaned samples;
    Q from the half-power width of the Lomb-Scargle peak."""
    from astropy.timeseries import LombScargle
    t, x = t[keep], x[keep]
    f = ringdown_fit(t, x, damped=False, robust=False)[0]
    freq = np.linspace(0.5 * f, 1.5 * f, 3001)
    return {"f0": float(f), "Q": half_power_q(freq, LombScargle(t, _detrended(t, x)).power(freq))[1]}


def fft_answer(t, x, keep):
    """TRAP (step 4): cleaned, detrended samples interpolated onto an even grid at the median interval,
    FFT: f from the peak bin, Q = f / half-power width."""
    t, x = t[keep], x[keep]
    dt = np.median(np.diff(t))
    grid = np.arange(t[0], t[-1], dt)
    y = np.interp(grid, t, _detrended(t, x))
    f, q = half_power_q(np.fft.rfftfreq(len(y), dt)[1:], (np.abs(np.fft.rfft(y - y.mean())) ** 2)[1:])
    return {"f0": f, "Q": q}


# ----------------------------------------------------------------------------- the family
SETUPS = (  # (apparatus, sensor, column, quantity, unit, f0 range Hz, amplitude range)
    ("a brass mass hanging from a steel coil spring, with an adjustable eddy-current damping magnet",
     "a laser triangulation sensor", "x_mm", "vertical displacement", "mm", (0.5, 8.0), (2.0, 20.0)),
    ("a torsion pendulum (a brass disc on a steel wire) with an adjustable eddy-current damper",
     "an optical lever", "angle_mrad", "twist angle", "mrad", (0.5, 5.0), (5.0, 60.0)),
    ("a clamped aluminium cantilever with an adjustable magnetic damper near its tip",
     "a capacitive displacement probe", "x_um", "tip displacement", "um", (3.0, 40.0), (10.0, 300.0)),
    ("a glider on an air track held between two springs, with adjustable magnetic damping",
     "an ultrasonic motion sensor", "x_mm", "position", "mm", (0.5, 4.0), (5.0, 40.0)),
)


class DampedOscillator(Family):
    name = "osc_damped"
    domain = "physics"
    keys = ("f0", "Q")
    steps = {1: "reject the single-sample glitches (robust fit, then drop gross outliers)",
             2: "remove the instrumental drift (a baseline fitted jointly with the oscillation)",
             3: "model the decay: a damped, not a constant-amplitude, sinusoid",
             4: "fit in the time domain on the uneven samples (not an FFT of interpolated data)",
             5: "report the natural frequency f0, not the damped frequency f_d"}
    skills = {1: "qc", 2: "normalise", 3: "model-choice", 4: "estimate", 5: "report"}
    tol_rule = ("f0: max(2 SE, 0.2% of f0, 0.02/T); Q: max(2 SE, 3% of Q, 0.2 Q/cycles); SE from the reference fit's "
                "Newey-West (HAC) covariance over ~2 periods; T the record length, cycles = f0 T")
    note = ("ground truth is the reference analysis (soft-L1 then clipped least-squares fit of a damped sinusoid plus "
            "a quadratic drift to the uneven samples, Lomb-Scargle start; f0 = sqrt(f_d^2 + (gamma/2pi)^2), "
            "Q = pi f0/gamma) on the sampled ring-down, not the latent parameters; deviation is in units of "
            "tolerance, maxed over the answer keys")
    seed_blocks = {"train": 17_000_000, "test": 17_050_000}
    max_tries = 12
    rng_policy = "stream"
    demo_timeout = 180

    def knobs(self, rng):
        return {"spikes": 1.0, "drift": 1.0, "short": 1.0}

    def draw(self, rng, k) -> Sample:
        j = int(rng.integers(len(SETUPS)))
        setup = SETUPS[j]
        (flo, fhi), (alo, ahi) = setup[5], setup[6]
        f0 = float(np.exp(rng.uniform(np.log(flo), np.log(fhi))))
        Q = float(np.exp(rng.uniform(np.log(3.0), np.log(80.0))))
        gamma = np.pi * f0 / Q
        fd = f0 * np.sqrt(1 - 1 / (4 * Q * Q))
        n_tau = max(rng.uniform(2.5, 4.0) / k["short"], 1.8)            # record length in decay times
        T = n_tau / gamma
        n = int(min(max(rng.uniform(300, 900), 5 * f0 * T), 2500))
        dt = np.exp(rng.normal(0, 0.4, n)) * (T / n)                    # uneven polling intervals
        t = np.cumsum(dt) - dt[0] + rng.uniform(0.02, 0.2) / f0
        t = t[t <= T]
        for _ in range(int(rng.integers(2, 5))):                         # logger drop-outs
            a = rng.uniform(0.05, 0.9) * T
            t = t[(t < a) | (t > a + rng.uniform(0.02, 0.06) * T)]
        A = float(np.exp(rng.uniform(np.log(alo), np.log(ahi))))
        x = A * np.exp(-gamma * t) * np.cos(2 * np.pi * fd * t + rng.uniform(0, 2 * np.pi))
        u = t / T
        span = rng.uniform(1.0, 5.0) * A * k["drift"]                    # the zero drifts as the sensor warms
        c1, c2 = rng.uniform(-1, 1, 2)
        x += span * (c1 * u + c2 * u * u) / (abs(c1) + abs(c2) + 1e-9) + rng.normal(0, A)
        tau_n, s_red = rng.uniform(0.2, 2.0) / f0, rng.uniform(0.01, 0.04) * A
        red = np.zeros(len(t))
        for i in range(1, len(t)):                                       # Ornstein-Uhlenbeck (coloured) noise
            a = np.exp(-(t[i] - t[i - 1]) / tau_n)
            red[i] = a * red[i - 1] + s_red * np.sqrt(1 - a * a) * rng.normal()
        x += red + rng.uniform(0.005, 0.02) * A * rng.normal(size=len(t))
        ns = int(rng.integers(3, 9))
        idx = rng.choice(len(t), ns, replace=False)
        x[idx] += rng.choice([-1.0, 1.0], ns) * rng.uniform(1.5, 5.0, ns) * A * k["spikes"]
        res = 10.0 ** (np.floor(np.log10(A)) - 4)                       # sensor resolution
        df = pd.DataFrame({"time_s": np.round(t, 5), setup[2]: np.round(np.round(x / res) * res, 8)})
        meta = {"setup": j, "n": len(df), "T": float(T)}
        latent = {"f0": f0, "Q": Q, "f_d": float(fd), "gamma": float(gamma), "amplitude": A,
                  "record_decay_times": float(n_tau), "drift_span": float(span), "n_glitches": ns,
                  "glitch_idx": sorted(int(i) for i in idx), "red_noise_sd": float(s_red), "red_noise_tau": float(tau_n)}
        return Sample({"df": df, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["df"].to_csv(d / CSV, index=False)
        return [CSV]

    # ------------------------------------------------------------------ analysis
    def _tx(self, data):
        df = data["df"]
        return df["time_s"].to_numpy(float), df.iloc[:, 1].to_numpy(float)

    def _fit(self, data, **kw):
        """ringdown_fit on the draw with one stage switched by `kw` (none: the reference), cached."""
        cache = data.setdefault("_cache", {})
        key = tuple(sorted(kw.items()))
        if key not in cache:
            fd, g, keep, sol = ringdown_fit(*self._tx(data), **kw)
            f0, q = natural_frequency_q(fd, g)
            cache[key] = {"f0": f0, "Q": q, "_fd": float(fd), "_gamma": float(g), "_keep": keep, "_sol": sol}
        return cache[key]

    def reference(self, data) -> dict:
        r = self._fit(data)
        t = self._tx(data)[0]
        T = float(t[-1] - t[0])
        n = int(r["_keep"].sum())
        cycles = r["f0"] * T
        se_f, se_q, rho = fit_se(r["_sol"], int(min(max(2 * n / cycles, 2), n // 4)))   # HAC over ~2 periods
        return {"f0": r["f0"], "Q": r["Q"], "_se_f0": se_f, "_se_Q": se_q, "_record_s": T, "_cycles": float(cycles),
                "_rho": rho, "_f_d": r["_fd"], "_gamma": r["_gamma"], "_n_rejected": int((~r["_keep"]).sum())}

    def tolerance(self, gold) -> dict:
        """2 SE, with floors: 0.2% of f0 or 1/50 of the record's frequency resolution 1/T, and 3% of Q or
        Q x 0.2 / (cycles in the record) -- a few-cycle ring-down cannot pin f0 and Q to better than a
        fraction of its resolution, and there the correlated noise is under-covered by any residual SE."""
        f0, q, c = abs(gold["f0"]), abs(gold["Q"]), gold["_cycles"]
        return {"f0": max(2 * gold["_se_f0"], 0.002 * f0, 0.02 / gold["_record_s"]),
                "Q": max(2 * gold["_se_Q"], 0.03 * q, 0.2 * q / c)}

    def sanity(self, gold, tol, latent) -> list[str]:
        ok = all(np.isfinite(gold[k]) and gold[k] > 0 for k in ("f0", "Q", "_se_f0", "_se_Q"))
        return [] if ok else ["fit_not_identified"]

    def traps(self, data) -> dict:
        def safe(fn):
            try:
                a = fn()
                return {k: float(a[k]) for k in self.keys} if all(np.isfinite(a[k]) for k in self.keys) \
                    else {k: NAN for k in self.keys}
            except Exception:                                            # noqa: BLE001
                return {k: NAN for k in self.keys}
        ref = self._fit(data)
        t, x = self._tx(data)
        pick = lambda r: {"f0": r["f0"], "Q": r["Q"]}
        return {"no_outlier_removal": Trap(safe(lambda: pick(self._fit(data, robust=False))), 1),
                "no_detrending": Trap(safe(lambda: pick(self._fit(data, deg=0))), 2),
                "undamped_sinusoid": Trap(safe(lambda: undamped_answer(t, x, ref["_keep"])), 3),
                "fft_interpolated": Trap(safe(lambda: fft_answer(t, x, ref["_keep"])), 4),
                "damped_frequency_as_f0": Trap({"f0": ref["_fd"], "Q": ref["Q"]}, 5, False),
                "soft_l1_fit": Trap(safe(lambda: pick(self._fit(data, clip=None))), None),
                "cubic_drift": Trap(safe(lambda: pick(self._fit(data, deg=3))), None)}

    def strengthen(self, k, dead):
        k = dict(k)
        if "no_outlier_removal" in dead:
            k["spikes"] = round(min(k["spikes"] * 1.5, 4.0), 4)
        if "no_detrending" in dead:
            k["drift"] = round(min(k["drift"] * 1.5, 4.0), 4)
        if {"undamped_sinusoid", "fft_interpolated"} & set(dead):
            k["short"] = round(min(k["short"] * 1.2, 1.6), 4)            # a shorter record: more spectral broadening
        return k

    def shortcuts(self, data) -> dict:
        """The periodogram of the raw series: f at the Lomb-Scargle peak, Q from its half-power width."""
        from astropy.timeseries import LombScargle
        t, x = self._tx(data)
        freq, power = LombScargle(t, x).autopower(maximum_frequency=0.5 / np.median(np.diff(t)), samples_per_peak=20)
        f, q = half_power_q(freq, power)
        return {"periodogram_raw": {"f0": f, "Q": q}}

    def extra_provenance(self, inst) -> dict:
        g = inst.a.gold
        return {"knobs": inst.knobs, "design": {"setup": SETUPS[inst.sample.data["meta"]["setup"]][0],
                                                "n_samples": inst.sample.data["meta"]["n"]},
                "reference_diagnostics": {k[1:]: v for k, v in g.items() if k.startswith("_")},
                "skills": {str(s): v for s, v in self.skills.items()}}

    # ------------------------------------------------------------------ prompt
    def task_text(self, data, blind: bool = False) -> str:
        app, sensor, col, what, unit = SETUPS[data["meta"]["setup"]][:5]
        head = (f"You are an experimental physicist. {app[0].upper() + app[1:]} was displaced and released, and "
                f"{sensor} recorded its free decay (ring-down) from just after the release.\n\n"
                "Data files:\n"
                f"- data_files/{CSV}: one row per sample. Columns: time_s (time since the release, s), {col} ({what}, "
                f"{unit}).\n\n"
                "About the recording: the data logger polls the sensor over a shared bus, so the samples are unevenly "
                "spaced (each carries its own timestamp) and the logger dropped out a few times; the sensor's zero "
                "drifts slowly as it warms up; and electrical interference occasionally produces a single-sample "
                "spike.\n\n")
        q = ("Question: the oscillator obeys x'' + (2 pi f0 / Q) x' + (2 pi f0)^2 x = 0. What are its natural "
             "(undamped) frequency f0 (Hz) and its quality factor Q?\n\n")
        method = ("" if blind else
                  "Approach: fit the ring-down in the time domain, on the samples as recorded (do not interpolate onto "
                  "an even grid for an FFT): exp(-gamma t) (a cos 2 pi f_d t + b sin 2 pi f_d t) plus a quadratic "
                  "baseline for the drift, fitted jointly by nonlinear least squares and started from the Lomb-Scargle "
                  "periodogram peak. Handle the spikes with a robust (soft-L1) fit first, then drop points more than 5 "
                  "robust SDs off it and refit. The fitted frequency is the damped one: f0 = sqrt(f_d^2 + "
                  "(gamma/2 pi)^2) and Q = pi f0 / gamma.\n\n")
        return head + q + method + 'Return your answer as JSON: {"answer": {"f0": <Hz>, "Q": <dimensionless>}}'

    def demo(self, cfg) -> list[Cell]:
        inspect = (f'import numpy as np, pandas as pd\nfrom astropy.timeseries import LombScargle\n'
                   f'df = pd.read_csv("data_files/{CSV}")\nprint(df.head(3).to_string(index=False))\n'
                   't, x = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)\n'
                   'dt = np.diff(t)\n'
                   'print(f"{len(t)} samples over {t[-1] - t[0]:.4g} s; interval median {np.median(dt):.3g} s, "\n'
                   '      f"max {dt.max():.3g} s; drop-outs (> 5 x median): {(dt > 5 * np.median(dt)).sum()}")\n'
                   'u = (t - t[0]) / (t[-1] - t[0])\n'
                   'base = np.polyval(np.polyfit(u, x, 2), u)                 # a rough quadratic baseline\n'
                   'r = x - base\n'
                   'iqr = lambda v: np.subtract(*np.percentile(v, [75, 25]))\n'
                   'print(f"baseline moves by {base[-1] - base[0]:.4g}; spread (IQR) about it: first tenth "\n'
                   '      f"{iqr(r[u < 0.1]):.4g}, last tenth {iqr(r[u > 0.9]):.4g}")\n'
                   'w = (t[1:-1] - t[:-2]) / (t[2:] - t[:-2])                   # each sample vs the line through its neighbours\n'
                   'e = np.abs(np.r_[0, x[1:-1] - (x[:-2] + w * (x[2:] - x[:-2])), 0])\n'
                   'local = 1.4826 * pd.Series(e).rolling(51, center=True, min_periods=5).median().to_numpy()\n'
                   'spike = (e > 10 * local) & (e >= np.r_[0, e[:-1]]) & (e >= np.r_[e[1:], 0])\n'
                   'print("isolated spikes (> 10 local robust SDs off their neighbours):", int(spike.sum()))\n'
                   'freq, power = LombScargle(t, r).autopower(maximum_frequency=0.5 / np.median(dt), samples_per_peak=20)\n'
                   'print(f"Lomb-Scargle peak {freq[power.argmax()]:.5g} Hz: about "\n'
                   '      f"{freq[power.argmax()] * (t[-1] - t[0]):.0f} cycles in the record")')

        def prose2(outs):
            o = outs[0]
            num = lambda pat: (m.group(1) if (m := re.search(pat, o, re.M)) else None)
            n, gaps = num(r"^(\d+) samples over"), num(r"drop-outs \(> 5 x median\): (\d+)")
            move, early, late = num(r"baseline moves by (\S+);"), num(r"first tenth (\S+),"), num(r"last tenth (\S+)")
            spikes, peak = num(r"off their neighbours\): (\d+)"), num(r"peak (\S+) Hz")
            drift = (f"The baseline moves by {move} over the record while the oscillation's spread falls from {early} "
                     f"to {late}" if move and early and late else "The baseline drifts while the oscillation decays")
            glitch = (f"{spikes} isolated spike{'s stand' if spikes != '1' else ' stands'} out (electrical glitches)"
                      if spikes and spikes != "0" else "The task warns of single-sample glitches")
            return (f"{n or 'The'} unevenly spaced samples with {gaps or 'several'} drop-outs: I fit in the time domain "
                    "on the samples as recorded, because interpolating across the gaps onto an even grid for an FFT "
                    f"would bias the peak and its width. {drift}, so the drift enters the model as a quadratic baseline "
                    "fitted jointly with the oscillation, and the decay as exp(-gamma t) -- a constant-amplitude sinusoid "
                    f"would not give Q. {glitch}: a first fit with a soft-L1 loss is insensitive to them, points more "
                    "than 5 robust SDs off it are dropped, and the fit is repeated by least squares. The fit starts from "
                    f"the Lomb-Scargle peak{f' at {peak} Hz' if peak else ''}. The "
                    "fitted frequency is the damped one, f_d = f0 sqrt(1 - 1/(4Q^2)), so I convert: f0 = "
                    "sqrt(f_d^2 + (gamma/2 pi)^2) and Q = pi f0 / gamma.")
        fit = ('f_d, gamma, keep, sol = ringdown_fit(t, x)\n'
               'f0, Q = natural_frequency_q(f_d, gamma)\n'
               'print(f"{(~keep).sum()} samples rejected as glitches; residual SD {np.std(sol.fun):.3g}")\n'
               'print(f"damped frequency f_d = {f_d:.6g} Hz, decay rate gamma = {gamma:.5g} 1/s, f0/f_d = {f0 / f_d:.6f}")\n'
               'print("FINAL f0 = %.6f Q = %.6f" % (f0, Q))')
        return [Cell("I load the ring-down and look at the sampling (spacing, drop-outs), how far the baseline drifts "
                     "compared with the oscillation, whether any samples are isolated spikes, and where the periodogram "
                     "peaks.", inspect),
                Cell(prose2, fit, uses=(ringdown_fit, natural_frequency_q))]


FAMILIES = [DampedOscillator()]
