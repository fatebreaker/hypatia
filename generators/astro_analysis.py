"""Reference analyses for the Kepler light-curve families (generators/astro_kepler.py).

Every function here is self-contained numpy/scipy code: rl/traces_astro.py embeds their
source verbatim in the demonstration cells, so the trace an agent learns from and the
reference that defines the ground truth are the same code, byte for byte. Keep them free
of module state other than CAD and of anything the sandbox lacks (numpy, scipy, pandas,
statsmodels, scikit-learn only).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

CAD = 0.0204336          # Kepler long cadence = integration time, days (29.4 min)


# ============================================================================ shared
def segments(t, gap=0.5):
    """Index arrays of the continuous stretches of t (split wherever a gap exceeds `gap` days)."""
    cut = np.flatnonzero(np.diff(t) > gap) + 1
    return np.split(np.arange(len(t)), cut)


def running_median(t, f, window=1.0, mask=None):
    """Median of the unmasked flux within +/- window/2 of every point, separately in each
    continuous segment (a trend must not be smeared across a data gap or a quarter boundary).
    Masked points get the median of their unmasked neighbours, so masking the transits
    interpolates the trend underneath them instead of letting them pull it down."""
    keep = np.ones(len(t), bool) if mask is None else ~np.asarray(mask)
    trend = np.full(len(t), np.nan)
    for idx in segments(t):
        tu, fu = t[idx][keep[idx]], f[idx][keep[idx]]
        if len(tu) < 3:
            trend[idx] = np.median(f[idx])
            continue
        lo = np.searchsorted(tu, t[idx] - window / 2)
        hi = np.searchsorted(tu, t[idx] + window / 2)
        trend[idx] = [np.median(fu[a:b]) if b - a >= 3 else np.nan for a, b in zip(lo, hi)]
        bad = np.isnan(trend[idx])
        if bad.any():
            trend[idx[bad]] = np.interp(t[idx[bad]], t[idx[~bad]], trend[idx[~bad]]) if (~bad).any() else np.median(fu)
    return trend


def clip_high(t, f, window=0.11, nsig=5.0):
    """Mask of points to keep: drops upward outliers (cosmic rays) more than nsig robust
    sigmas above a running median over ~5 cadences, short enough to follow even steep
    spot modulation; dips (transits, eclipses) are never clipped."""
    r = f / running_median(t, f, window) - 1
    s = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2) / np.median(f)    # white-noise sigma
    return r < nsig * s


def bls_power(t, y, periods, durations_h, qmax=0.2):
    """Box least squares power for each trial period. The light curve is folded into bins of
    one cadence and the in-box sums for every duration come from cumulative sums over the
    binned fold (wrapping around phase 1). Power = chi-square improvement of a box dip over a
    flat line, in units of the point-to-point variance; only durations between 0.025 and
    0.3 P**(1/3) days (and below qmax P) are tried at each period."""
    sig2 = (1.4826 * np.median(np.abs(np.diff(y))) / np.sqrt(2)) ** 2
    ntot, tref = len(t), t.min()
    power = np.zeros(len(periods))
    best = dict(power=-1.0)
    tt = t - tref
    for c0 in range(0, len(periods), 16):                  # small chunks stay in cache
        P = periods[c0:c0 + 16]
        nb = int(np.ceil(P[0] / CAD))
        idx = (tt[None, :] * (nb / P)[:, None]).astype(np.int64) % nb + np.arange(len(P))[:, None] * nb
        S = np.bincount(idx.ravel(), np.tile(y, len(P)), len(P) * nb).reshape(len(P), nb)
        N = np.bincount(idx.ravel(), None, len(P) * nb).reshape(len(P), nb).astype(float)
        Ds = [D / 24 for D in durations_h if 0.025 <= D / 24 / P[0] ** (1 / 3) <= 0.3 and D / 24 <= qmax * P[0]]
        wmax = int(np.ceil(max(Ds) / (P[0] / nb))) + 1 if Ds else 1
        cS = np.concatenate([np.zeros((len(P), 1)), np.cumsum(np.concatenate([S, S[:, :wmax]], 1), 1)], 1)
        cN = np.concatenate([np.zeros((len(P), 1)), np.cumsum(np.concatenate([N, N[:, :wmax]], 1), 1)], 1)
        pw_best = np.zeros(len(P))
        for D in Ds:
            w = max(int(round(D / (P[0] / nb))), 1)
            s_in = cS[:, w:w + nb] - cS[:, :nb]
            n_in = cN[:, w:w + nb] - cN[:, :nb]
            pw = np.where((n_in >= 3) & (s_in < 0), s_in ** 2 / np.maximum(n_in * (1 - n_in / ntot), 1e-9), 0.0) / sig2
            j = np.argmax(pw, axis=1)
            pj = pw[np.arange(len(P)), j]
            pw_best = np.maximum(pw_best, pj)
            i = int(np.argmax(pj))
            if pj[i] > best["power"]:
                depth = -s_in[i, j[i]] / n_in[i, j[i]] / (1 - n_in[i, j[i]] / ntot)
                best = dict(power=float(pj[i]), period=float(P[i]), duration=float(w * P[i] / nb),
                            t0=float(tref + (j[i] + w / 2) / nb * P[i]), depth=float(depth))
        power[c0:c0 + 16] = pw_best
    return power, best


def bls(t, f, pmin=0.5, pmax=None, durations_h=(1, 1.5, 2, 3, 4, 5, 6.5, 8, 10, 13, 16)):
    """Box least squares transit/eclipse search on a detrended light curve (flux ~ 1).

    Coarse pass: frequencies uniform in f**(1/3), stepped so that over the baseline a transit
    of the shortest plausible duration (0.035 P**(1/3) days) drifts by at most its own
    length. Fine pass: 8x finer steps around the three strongest coarse peaks. Returns
    (periods, power, best) for the coarse grid, best = dict(period, t0, duration, depth, snr)."""
    t = np.asarray(t, float)
    y = np.asarray(f, float) - np.mean(f)
    span = t.max() - t.min()
    pmax = pmax or min(60.0, span / 2.5)
    dx = 0.035 / (3 * span)
    x = np.arange(pmax ** (-1 / 3), pmin ** (-1 / 3), dx)
    periods = 1.0 / x ** 3
    power, best = bls_power(t, y, periods, durations_h)
    order = np.argsort(power)[::-1]
    peaks = []
    for i in order:                                    # three strongest, >1% apart in period
        if all(abs(periods[i] / periods[j] - 1) > 0.01 for j in peaks):
            peaks.append(i)
        if len(peaks) == 3:
            break
    for i in peaks:
        xf = x[i] + np.arange(-16, 17) * dx / 8
        pf, bf = bls_power(t, y, 1.0 / xf ** 3, durations_h)
        if bf["power"] > best["power"]:
            best = bf
    best["snr"] = float(np.sqrt(max(best["power"], 0.0)))
    return periods, power, best


def fold(t, P, t0):
    """Time from the nearest event, in days, for a linear ephemeris."""
    return (t - t0 + 0.5 * P) % P - 0.5 * P


# ============================================================================ transit
_GX, _GW = np.polynomial.legendre.leggauss(12)
_TH = (_GX + 1) * np.pi / 2
_WT = _GW * np.pi / 2 * np.sin(_TH)


def occult(z, k, u1, u2):
    """Fraction of a quadratically limb-darkened star's flux hidden by a dark disk of radius k
    at centre distance z (both in stellar radii). Rings of radius r lose the arc 2*kappa*r
    inside the planet's disk; the fully covered core has a closed form, the partial rings are
    integrated by Gauss-Legendre after the map r = a + (b-a)(1-cos th)/2, which removes the
    square-root singularities at the ring-tangency radii and at the stellar limb."""
    z = np.maximum(np.asarray(z, float), 1e-12)
    c = np.clip(k - z, 0, 1)
    mc = np.sqrt(1 - c * c)
    F = lambda m: (1 - u1 - u2) * m ** 2 / 2 + (u1 + 2 * u2) * m ** 3 / 3 - u2 * m ** 4 / 4
    core = np.where(k > z, 2 * np.pi * (F(1.0) - F(mc)), 0.0)
    a = np.clip(np.abs(z - k), 0, 1)[:, None]
    b = np.clip(z + k, 0, 1)[:, None]
    r = a + (b - a) * (1 - np.cos(_TH)) / 2
    kap = np.arccos(np.clip((r * r + z[:, None] ** 2 - k * k) / (2 * r * z[:, None]), -1, 1))
    mu = np.sqrt(np.clip(1 - r * r, 0, 1))
    ring = 2 * kap * r * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)
    part = (b - a)[:, 0] / 2 * (ring * _WT).sum(1)
    return (core + part) / (np.pi * (1 - u1 / 3 - u2 / 6))


def transit_model(t, t0, P, k, aRs, b, u1, u2, nsub=11):
    """Circular-orbit transit light curve. With nsub > 1 every point is the average over the
    29.4-min integration (nsub sub-exposures); nsub = 1 is the instantaneous flux."""
    off = ((np.arange(nsub) + 0.5) / nsub - 0.5) * CAD
    ts = np.asarray(t, float)[:, None] + off[None, :]
    ph = 2 * np.pi * (ts - t0) / P
    z = aRs * np.sqrt(np.sin(ph) ** 2 + (b / aRs * np.cos(ph)) ** 2)
    flux = np.ones(z.shape)
    m = (np.cos(ph) > 0) & (z < 1 + k)
    if m.any():
        flux[m] = 1 - occult(z[m], k, u1, u2)
    return flux.mean(1)


def t14(P, k, aRs, b):
    """Total transit duration (first to fourth contact), days."""
    s = np.sqrt(max((1 + k) ** 2 - b * b, 0.0)) / aRs / np.sqrt(1 - (b / aRs) ** 2)
    return P / np.pi * np.arcsin(min(s, 1.0))


def transit_windows(t, f, P, t0, dur, side=None, order=2):
    """Local detrending: keep +/- (dur/2 + side) days around every transit and divide each
    window by a polynomial fitted to its OUT-of-transit points only (|dt| > 0.6 dur + one
    cadence). Transits without baseline on both sides are dropped. Returns (t, f) of the
    normalised windows."""
    side = max(2 * dur, 0.35) if side is None else side
    dt = fold(t, P, t0)
    ep = np.round((t - t0) / P).astype(int)
    tt, ff = [], []
    for e in np.unique(ep[np.abs(dt) < dur / 2 + side]):
        w = (ep == e) & (np.abs(dt) < dur / 2 + side)
        oot = w & (np.abs(dt) > 0.6 * dur + CAD)
        n_in = np.sum(w & (np.abs(dt) < 0.5 * dur))
        if n_in < 2 or np.sum(oot & (dt < 0)) < 4 or np.sum(oot & (dt > 0)) < 4:
            continue
        c = np.polyfit(dt[oot], f[oot], order)
        tt.append(t[w])
        ff.append(f[w] / np.polyval(c, dt[w]))
    return np.concatenate(tt), np.concatenate(ff)


def fit_transit(t, f, P, t0, k0, dur, u1, u2, nsub=11, b_starts=(0.1, 0.45, 0.75)):
    """Least-squares fit of (t0, P, k, a/R*, b) with fixed quadratic limb darkening, started
    from several impact parameters (the b - a/R* - k valley is shallow at low S/N); returns
    the best solution as a dict. Only points within one duration of mid-transit enter: farther
    out the model is exactly 1 and the normalised flux carries no information."""
    near = np.abs(fold(t, P, t0)) < max(dur, 3 * CAD)
    t, f = t[near], f[near]
    best = None
    for b0 in b_starts:
        a0 = P * np.sqrt(max((1 + k0) ** 2 - b0 * b0, 0.05)) / (np.pi * max(dur, 0.02))
        a0 = float(np.clip(a0, 1.6, 250))
        x0 = [t0, P, k0, a0, b0]
        lo = [t0 - 0.1, P * 0.99, 0.2 * k0, 1.5, 0.0]
        hi = [t0 + 0.1, P * 1.01, min(0.6, 3 * k0), 300.0, 1.2]
        res = least_squares(lambda p: transit_model(t, *p, u1, u2, nsub) - f, x0, bounds=(lo, hi),
                            x_scale=[0.002, 1e-5 * P, 0.1 * k0, 0.1 * a0, 0.1])
        if best is None or res.cost < best.cost:
            best = res
    t0, P, k, aRs, b = best.x
    J = best.jac
    cov = np.linalg.pinv(J.T @ J) * np.sum(best.fun ** 2) / max(len(t) - 5, 1)
    return dict(t0=float(t0), period=float(P), rp_rs=float(k), a_rs=float(aRs), b=float(b), t14=float(t14(P, k, aRs, b)),
                rp_rs_err=float(np.sqrt(max(cov[2, 2], 0.0))), rms=float(np.std(best.fun)), n=len(t))


# ============================================================================ eclipsing binary
def eb_detrend(t, f, window=1.5, n_iter=3):
    """Out-of-eclipse trend by an iterated running median: points far below the trend
    (the eclipses) are masked, the mask grown by 0.1 d each side, and the median recomputed
    from the remaining points."""
    mask = np.zeros(len(t), bool)
    for _ in range(n_iter):
        trend = running_median(t, f, window, mask)
        r = f / trend - 1
        s = 1.4826 * np.median(np.abs(r[~mask] - np.median(r[~mask])))
        low = r < -5 * s
        mask = np.zeros(len(t), bool)
        if low.any():
            tl = t[low]
            j = np.searchsorted(tl, t)
            d1 = np.abs(t - tl[np.clip(j - 1, 0, len(tl) - 1)])
            d2 = np.abs(t - tl[np.clip(j, 0, len(tl) - 1)])
            mask = np.minimum(d1, d2) < 0.1
    return f / running_median(t, f, window, mask)


def dip_fit(dt, f, width0):
    """Symmetric dip f = c - A exp(-(dt - tc)^2 / 2 s^2) fitted to points near an eclipse;
    returns (tc, A, s, core_depth) where core_depth = 1 - median flux within 0.3 s of tc."""
    lo, hi = [-width0, 0, CAD / 4, 0.5], [width0, 1.0, max(2 * width0, CAD), 1.5]
    p0 = np.clip([0.0, 1 - np.min(f), width0 / 4, 1.0], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    res = least_squares(lambda p: p[3] - p[1] * np.exp(-0.5 * ((dt - p[0]) / p[2]) ** 2) - f, p0, bounds=(lo, hi))
    tc, A, s, c = res.x
    core = np.abs(dt - tc) < max(0.3 * s, CAD)
    depth = 1 - np.median(f[core]) if core.sum() >= 3 else A
    return float(tc), float(A), float(s), float(depth)


def eclipse_times(t, f, P, t0, dur):
    """Individual eclipse mid-times near t0 + n P from dip fits to each event."""
    ep = np.round((t - t0) / P).astype(int)
    dt = t - (t0 + ep * P)
    out = []
    for e in np.unique(ep):
        w = (ep == e) & (np.abs(dt) < 1.5 * dur)
        if np.sum(w & (np.abs(dt) < 0.3 * dur)) < 3 or np.sum(w) < 8:
            continue
        tc, A, s, _ = dip_fit(dt[w], f[w], dur)
        if abs(tc) < 0.5 * dur and A > 0:
            out.append((e, t0 + e * P + tc))
    return np.array(out)


def secondary_search(t, f, P, t_pri, dur, excl=None):
    """Most significant dip in the fold at P away from the primary eclipse -- a matched
    filter. Boxes of 0.5, 1 and 1.5 primary durations slide on a phase grid of half a
    cadence; each is scored by how far its mean falls below the average of its two flanking
    boxes of the same width (both must hold data), in units of the noise, so a shallow
    secondary is found by its
    significance against the LOCAL baseline (smooth out-of-eclipse modulation left by the
    detrending cancels) rather than by one noisy bin. Bins within `excl` days of the
    primary's centre (default: the BLS duration) are removed first and no box may reach
    them: the primary's wings are deeper than a shallow secondary, so a box that grazes them
    always wins. Returns the dip's phase (0.5 if no box qualifies)."""
    excl = dur if excl is None else excl
    ph = np.mod((t - t_pri) / P, 1.0)
    nb = int(np.ceil(P / (CAD / 2)))
    b = (ph * nb).astype(int) % nb
    S = np.bincount(b, f - 1, nb)
    N = np.bincount(b, None, nb).astype(float)
    cb = (np.arange(nb) + 0.5) / nb
    inside = np.abs(((cb + 0.5) % 1.0) - 0.5) * P <= excl           # the primary's bins
    S[inside], N[inside] = 0.0, 0.0
    best, best_ph = 0.0, 0.5
    for frac in (0.5, 1.0, 1.5):
        w = max(int(round(frac * dur / (P / nb))), 1)
        Sw = np.convolve(np.concatenate([S, S[:w]]), np.ones(w), "valid")[:nb]
        Nw = np.convolve(np.concatenate([N, N[:w]]), np.ones(w), "valid")[:nb]
        SL, NL, SR, NR = np.roll(Sw, w), np.roll(Nw, w), np.roll(Sw, -w), np.roll(Nw, -w)
        centre = (np.arange(nb) + w / 2) / nb
        near_edge = (np.abs(((centre + 0.5) % 1.0) - 0.5) - w / 2 / nb) * P   # box edge to primary, days
        ok = (near_edge > excl) & (Nw >= max(3, 0.5 * np.median(Nw[Nw > 0]))) & (NL >= Nw / 3) & (NR >= Nw / 3)
        # baseline = mean of the two flank means (equal weights), so a linear slope cancels
        base = 0.5 * (SL / np.maximum(NL, 1) + SR / np.maximum(NR, 1))
        contrast = base - Sw / np.maximum(Nw, 1)                            # > 0 for a dip
        var = 1 / np.maximum(Nw, 1) + 0.25 * (1 / np.maximum(NL, 1) + 1 / np.maximum(NR, 1))
        z = np.where(ok, contrast / np.sqrt(var), 0.0)
        j = int(np.argmax(z))
        if z[j] > best:
            best, best_ph = z[j], float(centre[j] % 1.0)
    return best_ph


def fit_away(t, f, P, tc, dur, t_avoid, avoid):
    """Dip fit to the fold at P around tc that cannot land on another eclipse: only points
    within 1.5 dur of tc AND more than `avoid` days from t_avoid enter, and if the fitted
    centre still runs into that zone (or more than one duration away) the fit is rejected
    and tc kept. Returns (centre, core depth, width)."""
    d = fold(t, P, tc)
    w = (np.abs(d) < 1.5 * dur) & (np.abs(fold(t, P, t_avoid)) > avoid)
    if w.sum() < 8:
        return tc, 0.0, dur / 4
    tcf, A, s, depth = dip_fit(d[w], f[w], dur)
    if A <= 0 or abs(tcf) > dur or abs(fold(tc + tcf, P, t_avoid)) <= avoid:
        core = w & (np.abs(d) < max(dur / 4, CAD))
        return tc, float(1 - np.median(f[core])) if core.sum() >= 3 else 0.0, s
    return tc + tcf, depth, s


def event_depths(t, f, P, tc, s):
    """Depth of every individual eclipse near tc + n P: amplitude of the Gaussian template
    exp(-dt^2 / 2 s^2) fitted to that event alone (linear least squares). Returns
    {epoch: depth}."""
    ep = np.round((t - tc) / P).astype(int)
    dt = t - (tc + ep * P)
    g = np.exp(-0.5 * (dt / s) ** 2)
    use = np.abs(dt) < 3 * s
    out = {}
    for e in np.unique(ep[use]):
        w = use & (ep == e)
        if np.sum(g[w] > 0.5) >= 2:
            out[int(e)] = float(np.sum(g[w] * (1 - f[w])) / np.sum(g[w] ** 2))
    return out


# Deliberate: near-equal twins whose eclipse depths drift slowly can fail this F test and return
# P/2; a paired odd/even test was considered and not adopted -- those real targets stay dropped.
def alias_multiple(ep, dv, max_m=4):
    """Smallest m such that grouping the events by epoch mod m explains their depths.

    Events spaced by the BLS period are the same eclipse type only if the true period IS the
    BLS period. If the true period is m times longer, the events fall into m groups (primary,
    secondary, or no eclipse at all). For each m the within-group sum of squares is compared
    with m = 1 by an F statistic; m is accepted when the groups differ significantly (F > 30)
    and by more than 2% of the eclipse depth. Of those, the one with the smallest within-group
    variance is refined back to its smallest accepted divisor unless the finer grouping is itself
    significant (nested F > 30): grouping by 4 always fits a true period of 2 BLS periods at
    least as well, by chance. Also returns {m: (within-group rms, group means)} for inspection."""
    n = len(dv)
    def ss(m):
        return sum(np.sum((dv[ep % m == g] - dv[ep % m == g].mean()) ** 2) for g in range(m) if np.any(ep % m == g))
    info, good = {}, {}
    S1 = ss(1)
    for m in range(1, max_m + 1):
        if n < 2 * m:
            break
        Sm = ss(m)
        means = [float(dv[ep % m == g].mean()) for g in range(m) if np.any(ep % m == g)]
        info[m] = (float(np.sqrt(Sm / n)), means)
        F = ((S1 - Sm) / (m - 1)) / max(Sm / (n - m), 1e-30) if m > 1 else 0.0
        if m > 1 and F > 30 and max(means) - min(means) > 0.02 * max(means):
            good[m] = Sm / (n - m)
    if not good:
        return 1, info
    best = min(good, key=good.get)
    for m in sorted(good):
        if best % m == 0:
            F = ((ss(m) - ss(best)) / max(best - m, 1)) / max(ss(best) / (n - best), 1e-30)
            if m == best or F < 30:
                return m, info
    return best, info


def eb_solve(t, f, P_bls, t0_bls, dur, refine=True):
    """Resolve the period alias, identify the primary (deeper) eclipse, and time both.

    BLS stacks eclipses of different kinds whenever they are spaced by a common fraction of
    the orbit -- a secondary at phase 0.5 gives half the true period, one near 1/3 or 2/3 of
    an eccentric orbit a third. Every event at the BLS spacing is fitted with one template
    and its depth recorded; if the events split into m groups by epoch mod m (odd/even for
    m = 2), the true period is m times the BLS period. Depth differences are judged against
    the scatter between events, not the white noise, because spots and residual trends change
    eclipse depths from one event to the next; a timing offset between odd and even events
    also means m = 2. At the adopted period the deeper eclipse is the primary, and the
    secondary is the other eclipse group, or else the deepest dip elsewhere in the fold."""
    ev = []
    for off in (0.0, P_bls):                          # odd and even events, folded at 2 P_bls
        d = fold(t - off, 2 * P_bls, t0_bls)
        w = np.abs(d) < 1.5 * dur
        ev.append(dip_fit(d[w], f[w], dur))
    (tA, AA, sA, dA), (tB, AB, sB, dB) = ev
    dep = event_depths(t, f, P_bls, t0_bls + tA, sA)
    ep = np.array(list(dep.keys()))
    dv = np.array(list(dep.values()))
    m, groups = alias_multiple(ep, dv)
    if m == 1 and abs(tA - tB) > max(3 * CAD, 0.1 * dur):
        m = 2                                         # odd and even events offset in time
    P = m * P_bls
    gm = {g: dv[ep % m == g].mean() for g in range(m) if np.any(ep % m == g)}
    gp = max(gm, key=gm.get)                          # deepest group
    others = {g: v for g, v in gm.items() if g != gp}
    scatter = np.std(dv[ep % m == gp]) if np.sum(ep % m == gp) > 1 else 0.0
    # the BLS eclipse first: its fitted width sets how much of the fold it owns (+/- 4 sigma
    # of the dip fit, never less than 0.75 of the BLS duration, which can undershoot it)
    d = fold(t, P, t0_bls + tA + gp * P_bls)
    w = np.abs(d) < 1.5 * dur
    tcf, A1, s1, dep1 = dip_fit(d[w], f[w], dur)
    tc1 = t0_bls + tA + gp * P_bls + tcf
    ext1 = max(4 * s1, 0.75 * dur)
    if others and max(others.values()) > max(5 * scatter, 0.05 * gm[gp]):
        gs = max(others, key=others.get)
        tc2 = t0_bls + tA + gs * P_bls
    else:
        tc2 = tc1 + secondary_search(t, f, P, tc1, dur, ext1) * P
    tc2, dep2, s2 = fit_away(t, f, P, tc2, dur, tc1, ext1)
    ecl = sorted([(tc1, dep1, ext1), (tc2, dep2, max(4 * s2, 0.75 * dur))], key=lambda e: -e[1])
    (t_pri, d_pri, ext_pri), (t_sec, d_sec, _) = ecl  # primary = the deeper eclipse
    if refine:                                        # linear ephemeris from the primary times
        et = eclipse_times(t, f, P, t_pri, dur)
        if len(et) >= 3:
            P, t_pri = np.polyfit(et[:, 0], et[:, 1], 1)
            t_sec = fit_away(t, f, P, t_sec, dur, t_pri, ext_pri)[0]
    phase = float(np.mod((t_sec - t_pri) / P, 1.0))
    return dict(period=float(P), secondary_phase=phase, t_primary=float(t_pri), t_secondary=float(t_sec),
                depth_primary=d_pri, depth_secondary=d_sec, multiple=int(m), p_bls=float(P_bls),
                group_depths={int(g): float(v) for g, v in gm.items()}, n_events=int(len(dv)), groups=groups,
                odd_even_time_offset=float(tB - tA))


# ============================================================================ rotation
def uniform_grid(t, f):
    """Flux minus 1 on a uniform grid at the Kepler cadence, zeros in the gaps."""
    n = int(round((t.max() - t.min()) / CAD)) + 1
    i = np.round((t - t.min()) / CAD).astype(int)
    x = np.zeros(n)
    x[i] = f - 1.0
    return x


def acf(t, f, max_lag_days=None):
    """Autocorrelation of the gap-filled uniform series (McQuillan et al. 2013, 2014):
    r_k = sum x_i x_{i+k} / sum x_i^2, then smoothed with a Gaussian of sigma = 9 lags.
    Returns (lags in days, raw acf, smoothed acf)."""
    x = uniform_grid(t, f)
    x = np.where(x != 0, x - np.mean(x[x != 0]), 0.0)
    n = len(x)
    F = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(F * np.conj(F))[:n]
    r = r / r[0]
    m = int((max_lag_days or 0.5 * (t.max() - t.min())) / CAD)
    r = r[:m]
    return np.arange(m) * CAD, r, gaussian_filter1d(r, 9, truncate=3.1)


def acf_peaks(lags, s, min_prom=0.02):
    """Local maxima of the smoothed ACF beyond its first zero-lag decline, with their
    parabola-refined lags and heights."""
    pk, props = find_peaks(s, prominence=min_prom)
    out = []
    for p in pk:
        if p < 2 or p > len(s) - 2:
            continue
        a, b, c = s[p - 1], s[p], s[p + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        out.append((float(lags[p] + d * (lags[1] - lags[0])), float(b)))
    return out


def acf_period(t, f):
    """Rotation period: the first ACF peak, unless the second peak is higher (two active
    longitudes make a lower peak at half the period); returns (period, first-peak lag, peaks)."""
    lags, r, s = acf(t, f)
    pk = acf_peaks(lags, s)
    if not pk:
        return float("nan"), float("nan"), pk
    P = pk[1][0] if len(pk) > 1 and pk[1][1] > pk[0][1] else pk[0][0]
    return float(P), float(pk[0][0]), pk


def acf_period_series(pk, P):
    """Cross-check: slope of a line through the lags of the peaks at multiples of P."""
    lag = np.array([p[0] for p in pk])
    n = np.round(lag / P)
    ok = (n >= 1) & (np.abs(lag - n * P) < 0.2 * P)
    if ok.sum() < 2:
        return float(P)
    return float(np.sum(n[ok] * lag[ok]) / np.sum(n[ok] ** 2))


def lomb_scargle(t, y, f0, df, n):
    """Classical Lomb-Scargle power (Scargle 1982) on the uniform grid f0 + j df, j < n.
    exp(i w t) is advanced from one frequency to the next by a single complex multiply
    (re-anchored every 256 steps), which keeps 8,000 points x 4,000 frequencies well under
    a second."""
    y = y - np.mean(y)
    tc = t - t.mean()
    step = np.exp(2j * np.pi * df * tc)
    out = np.empty(n)
    for j in range(n):
        if j % 256 == 0:
            e = np.exp(2j * np.pi * (f0 + j * df) * tc)
        z1 = e @ y                                    # sum y exp(i w t)
        z2 = np.sum(e * e)                            # sum exp(2 i w t)
        c, s = np.cos(np.angle(z2) / 2), np.sin(np.angle(z2) / 2)
        yc, ys = z1.real * c + z1.imag * s, z1.imag * c - z1.real * s
        out[j] = 0.5 * (yc ** 2 / (len(t) / 2 + abs(z2) / 2) + ys ** 2 / (len(t) / 2 - abs(z2) / 2))
        e = e * step
    return out


def ls_peak(t, f, pmin=0.5, pmax=70.0):
    """Period of the highest Lomb-Scargle peak (a naive alternative to the ACF)."""
    df = 0.1 / (t.max() - t.min())
    n = int((1 / pmin - 1 / pmax) / df)
    p = lomb_scargle(t, f, 1 / pmax, df, n)
    return float(1 / (1 / pmax + np.argmax(p) * df))
