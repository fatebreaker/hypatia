"""Forward models for the Kepler light-curve families (generators/astro_kepler.py).

Numpy only. Three pieces, each validated in `_self_check()`:

  occulted_fraction(z, k, u1, u2)
      Fraction of a quadratically limb-darkened star's flux hidden by an opaque disk of
      radius k (stellar radii) whose centre is z stellar radii from the star's centre.
      Integral over the stellar disk in polar rings: the occulted arc of the ring of radius r
      is 2*kappa(r)*r with kappa = arccos((r^2 + z^2 - k^2) / (2 r z)). The fully covered
      core (r < k - z) is integrated analytically; the partial ring zone [|z - k|, min(1, z + k)]
      by Gauss-Legendre after the cosine map r = a + (b - a)(1 - cos theta)/2, which turns the
      square-root endpoint singularities (ring tangent to the planet's limb, stellar limb
      mu -> 0) into smooth integrands. Accurate to < 1e-8 of the stellar flux with 32 nodes
      (checked against the exact uniform-disk overlap and a brute-force 2-D pixel sum).

  transit_flux / eb_flux
      Planet (circular orbit) and detached binary (Keplerian, eccentric) light curves,
      averaged over the 29.4-min Kepler long-cadence integration by supersampling.

  spot_flux
      Rotational modulation from evolving spots on a limb-darkened sphere.
"""
from __future__ import annotations

import numpy as np

CADENCE = 0.020433598        # Kepler long cadence, days (29.4244 min)
T_INT = 0.020433598          # integration time modelled by supersampling, days
BKJD0 = 2454833.0

_NQ = 32
_X, _W = np.polynomial.legendre.leggauss(_NQ)
_TH = (_X + 1.0) * np.pi / 2.0           # theta nodes on [0, pi]
_WTH = _W * np.pi / 2.0 * np.sin(_TH)    # d r = (b - a)/2 sin(theta) d theta


def _core(c, u1, u2):
    """Analytic integral of 2 pi r I(r) dr over r in [0, c] (I(0) = 1)."""
    mc = np.sqrt(np.clip(1.0 - c * c, 0.0, 1.0))
    a0, a1, a2 = 1.0 - u1 - u2, u1 + 2.0 * u2, -u2

    def F(m):
        return a0 * m ** 2 / 2.0 + a1 * m ** 3 / 3.0 + a2 * m ** 4 / 4.0
    return 2.0 * np.pi * (F(1.0) - F(mc))


def occulted_fraction(z, k, u1=0.0, u2=0.0) -> np.ndarray:
    """Blocked fraction of the stellar flux (0 = no occultation)."""
    z = np.atleast_1d(np.asarray(z, float))
    k = np.broadcast_to(np.asarray(k, float), z.shape)
    out = np.zeros(z.shape)
    m = z < 1.0 + k
    if not m.any():
        return out
    zz, kk = np.maximum(z[m], 1e-12), k[m]
    c = np.clip(kk - zz, 0.0, 1.0)                      # fully covered core radius
    blocked = np.where(kk > zz, _core(c, u1, u2), 0.0)
    a = np.clip(np.abs(zz - kk), 0.0, 1.0)[:, None]
    b = np.clip(zz + kk, 0.0, 1.0)[:, None]
    r = a + (b - a) * (1.0 - np.cos(_TH)) / 2.0
    arg = (r * r + zz[:, None] ** 2 - kk[:, None] ** 2) / (2.0 * r * zz[:, None])
    kap = np.arccos(np.clip(arg, -1.0, 1.0))
    mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
    inten = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    blocked = blocked + ((b - a)[:, 0] / 2.0) * np.sum(2.0 * kap * r * inten * _WTH, axis=1)
    out[m] = blocked / (np.pi * (1.0 - u1 / 3.0 - u2 / 6.0))
    return out


def _supersample(t, n):
    if n <= 1:
        return t[:, None]
    off = (np.arange(n) + 0.5) / n - 0.5
    return t[:, None] + off[None, :] * T_INT


def transit_flux(t, t0, P, k, aRs, b, u1, u2, n_sub=15) -> np.ndarray:
    """Relative flux of a star transited by a planet on a circular orbit, averaged over the
    long-cadence integration (n_sub subsamples; n_sub=1 is the instantaneous flux)."""
    ts = _supersample(np.asarray(t, float), n_sub)
    ph = 2.0 * np.pi * ((ts - t0) / P)
    cosi = b / aRs
    x = aRs * np.sin(ph)
    y = aRs * cosi * np.cos(ph)
    z = np.sqrt(x * x + y * y)
    front = np.cos(ph) > 0
    f = np.ones(z.shape)
    m = front & (z < 1.0 + k)
    if m.any():
        f[m] = 1.0 - occulted_fraction(z[m], k, u1, u2)
    return f.mean(axis=1)


# ----------------------------------------------------------------------------- binaries
def kepler_E(M, e):
    E = M + e * np.sin(M) / (1.0 - e * np.cos(M) + 1e-12) if e > 0 else M.copy()
    for _ in range(30):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < 1e-12:
            break
    return E


def eb_geometry(t, P, tperi, e, w, inc):
    """Sky separation (units of a) and the sign of the line-of-sight coordinate of star 2
    (positive: star 2 is in front of star 1)."""
    M = 2.0 * np.pi * (t - tperi) / P
    E = kepler_E(np.mod(M, 2.0 * np.pi), e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = (1.0 - e * e) / (1.0 + e * np.cos(nu))
    sep = r * np.sqrt(np.cos(w + nu) ** 2 + (np.sin(w + nu) * np.cos(inc)) ** 2)
    front2 = np.sin(w + nu) > 0
    return sep, front2


def eb_flux(t, P, tperi, e, w, inc, r1, r2, sb, ld1, ld2, n_sub=15) -> np.ndarray:
    """Normalised flux of two limb-darkened spheres. r1, r2: radii in units of the semi-major
    axis; sb: ratio of disk-averaged surface brightness (star 2 / star 1). Star 2 in front of
    star 1 is the primary conjunction."""
    ts = _supersample(np.asarray(t, float), n_sub)
    sep, front2 = eb_geometry(ts.ravel(), P, tperi, e, w, inc)
    L1, L2 = 1.0, sb * (r2 / r1) ** 2
    loss = np.zeros(sep.shape)
    m1 = front2 & (sep < r1 + r2)            # star 1 occulted by star 2
    if m1.any():
        loss[m1] = L1 * occulted_fraction(sep[m1] / r1, r2 / r1, *ld1)
    m2 = (~front2) & (sep < r1 + r2)         # star 2 occulted by star 1
    if m2.any():
        loss[m2] = L2 * occulted_fraction(sep[m2] / r2, r1 / r2, *ld2)
    f = 1.0 - loss / (L1 + L2)
    return f.reshape(ts.shape).mean(axis=1)


def conjunction_times(P, tperi, e, w, inc):
    """Times of minimum sky separation near the primary and secondary conjunctions."""
    out = []
    for nu_c in (np.pi / 2 - w, 3 * np.pi / 2 - w):
        E = 2.0 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu_c / 2))
        M = E - e * np.sin(E)
        tc = tperi + np.mod(M, 2 * np.pi) / (2 * np.pi) * P
        tt = tc + np.linspace(-0.02, 0.02, 4001) * P
        sep, _ = eb_geometry(tt, P, tperi, e, w, inc)
        out.append(float(tt[np.argmin(sep)]))
    return out


# ----------------------------------------------------------------------------- spots
def spot_flux(t, prot, inc, spots, u1=0.5, u2=0.2) -> np.ndarray:
    """Relative flux of a rotating star with small circular spots.

    spots: list of dicts with lon (rad), lat (rad), contrast (0-1), drift (fractional period
    offset, differential rotation), and area(t) (callable, fraction of the visible hemisphere).
    Each spot removes contrast * area * foreshortening(mu) * I(mu) / <I> of the flux."""
    t = np.asarray(t, float)
    norm = 1.0 - u1 / 3.0 - u2 / 6.0
    dF = np.zeros_like(t)
    for s in spots:
        ph = 2.0 * np.pi * t / (prot * (1.0 + s.get("drift", 0.0))) + s["lon"]
        mu = np.cos(inc) * np.sin(s["lat"]) + np.sin(inc) * np.cos(s["lat"]) * np.cos(ph)
        mu = np.clip(mu, 0.0, None)
        limb = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        dF += s["contrast"] * s["area"](t) * 4.0 * mu * limb / norm * (mu > 0)
    return 1.0 - dF


# ----------------------------------------------------------------------------- checks
def _uniform_overlap(z, k):
    """Exact overlap area of the unit disk and a disk of radius k at distance z, over pi."""
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    full = z <= abs(1 - k)
    out[full] = min(k * k, 1.0)
    part = (z > abs(1 - k)) & (z < 1 + k)
    zz = z[part]
    k0 = np.arccos((k * k + zz * zz - 1) / (2 * k * zz))
    k1 = np.arccos((1 - k * k + zz * zz) / (2 * zz))
    out[part] = (k * k * k0 + k1 - 0.5 * np.sqrt(4 * zz * zz - (1 + zz * zz - k * k) ** 2)) / np.pi
    return out


def _pixel_occultation(z, k, u1, u2, n=6000):
    """Brute force: sum I(r) over an n x n pixel grid covering the occulting disk."""
    xs = z - k + (np.arange(n) + 0.5) * (2 * k / n)
    ys = -k + (np.arange(n) + 0.5) * (2 * k / n)
    dA = (2 * k / n) ** 2
    tot = 0.0
    for i0 in range(0, n, 500):
        X, Y = np.meshgrid(xs, ys[i0:i0 + 500])
        r2 = X * X + Y * Y
        inside = (r2 < 1.0) & ((X - z) ** 2 + Y ** 2 < k * k)
        mu = np.sqrt(np.clip(1 - r2, 0, 1))
        inten = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
        tot += np.sum(inten * inside) * dA
    return tot / (np.pi * (1 - u1 / 3 - u2 / 6))


def _self_check(verbose=True) -> float:
    worst = 0.0
    zs = np.linspace(0.0, 1.25, 400)
    for k in (0.03, 0.1, 0.2, 0.7, 1.3):
        exact = _uniform_overlap(zs, k)
        num = occulted_fraction(zs, k, 0.0, 0.0)
        err = np.max(np.abs(num - exact))
        worst = max(worst, err)
        if verbose:
            print(f"uniform disk  k={k:<5} max|numeric - exact| = {err:.2e}")
    for z, k, u1, u2 in [(0.0, 0.1, 0.45, 0.20), (0.5, 0.1, 0.40, 0.25), (0.93, 0.1, 0.40, 0.25),
                         (1.05, 0.12, 0.60, 0.10), (0.3, 0.03, 0.35, 0.28), (0.85, 0.2, 0.50, 0.20)]:
        num = float(occulted_fraction(z, k, u1, u2)[0])
        pix = _pixel_occultation(z, k, u1, u2)
        worst = max(worst, abs(num - pix))
        if verbose:
            print(f"quadratic LD  z={z:<5} k={k:<5} u=({u1},{u2})  numeric={num:.9f}  pixel={pix:.9f}  "
                  f"diff={num - pix:+.2e}")
    return worst


if __name__ == "__main__":
    w = _self_check()
    print("worst absolute difference:", f"{w:.2e}")
