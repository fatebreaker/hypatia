"""How large a difference can the decision set actually resolve?

WHY THIS EXISTS
---------------
The backtest decision set is 1,000 targets with **215 positives** -- and that
215 is every positive available at this cutoff/final pair, not a sample, so it
is a hard ceiling that no additional download raises. ROADMAP already warns
that "AP differences under ~0.05 are probably not real". This turns that
warning into a number, BEFORE the GPU time is spent rather than after.

The failure mode being prevented is specific and common: run the agent, observe
AP 0.66 against a 0.628 baseline, and report a win that a bootstrap would have
shown is indistinguishable from zero.

WHAT IT MEASURES
----------------
Two regimes, because they have very different power:

* **ranking (AP)** -- driven by the 215 positives. Weak.
* **calibration (ECE)** -- uses all 1,000 predictions, and every classifier
  here is badly calibrated (0.154-0.222). Better powered, and it is the axis
  the project's own novelty check calls most defensible.

The agent and the classifiers score the SAME targets, so the comparison is
paired. That matters enormously: a paired bootstrap resamples targets and
takes the per-resample difference, cancelling the shared difficulty of the
draw. The higher the two scorers correlate, the more it cancels -- and since
the agent currently sees exactly the classifier's features rendered as text,
correlation should be high. Power is therefore reported across a range of
correlations rather than at one optimistic value.

MODES
-----
Real scores if the decision set has been built, otherwise a simulation matched
to its known composition (215/785) and the published baseline AP. The
simulation answers the design question -- "what gap is detectable at all?" --
which does not depend on the particular score values.

Run:
  venv/bin/python analysis/power.py                     # design mode
  venv/bin/python analysis/power.py --target-ap 0.628   # genetics-only bar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_POS_DEFAULT = 215
N_NEG_DEFAULT = 785


# --------------------------------------------------------------------------- #
# metrics -- implemented directly because the bootstrap calls them ~10^5 times
# --------------------------------------------------------------------------- #
def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP, delegating to sklearn because TIES MUST NOT DEPEND ON INPUT ORDER.

    This was a hand-rolled cumulative-precision loop. It agreed with sklearn to
    4e-16 on continuous scores -- which is exactly the case that cannot expose
    the bug -- and diverged badly the moment scores were tied: sorting picks
    *an* order for equal scores and credits whatever it happens to put first.

    Measured on real agent output with 11 distinct scores over 1,000 targets:

        sklearn        0.7615, spread 0.00000 over 200 random re-orderings
        hand-rolled    0.6950, spread 0.04096 over the same re-orderings

    sklearn's precision_recall_curve groups equal scores at one threshold, so
    it is invariant by construction. An LLM asked for a probability emits a
    handful of round numbers, so ties are the normal case here, not an edge
    case -- and the hand-rolled version understated AP by 0.066 and produced a
    metric that moved with row order.
    """
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if y.sum() else float("nan")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins. Mirrors agent_backtest."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def _scores(rng, y, mu, rho=None, base=None):
    """Latent-Gaussian scores: positives shifted by `mu`, negatives at 0.

    `rho` correlates this scorer's noise with `base`'s, which is how two models
    reading the same features behave. Independent scorers (rho=0) are the
    pessimistic case for a paired test.
    """
    z = rng.standard_normal(len(y))
    if base is not None and rho:
        z = rho * base + np.sqrt(1 - rho**2) * z
    return z + mu * y, z


def _mu_for_ap(target_ap: float, y: np.ndarray, seed: int = 0, reps: int = 80) -> float:
    """Binary-search the signal strength that yields `target_ap` in expectation."""
    rng = np.random.default_rng(seed)

    def ap_at(mu):
        return np.mean([average_precision(y, _scores(rng, y, mu)[0])
                        for _ in range(reps)])

    lo, hi = 0.0, 8.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if ap_at(mid) < target_ap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_ci(y, s_a, s_b, n_boot, rng, metric=average_precision):
    """Bootstrap CI for metric(a) - metric(b), resampling TARGETS (paired)."""
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            d[i] = np.nan
            continue
        d[i] = metric(yb, s_a[idx]) - metric(yb, s_b[idx])
    d = d[~np.isnan(d)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def power_for_gap(y, mu_base, mu_agent, rho, n_sim, n_boot, seed):
    """Fraction of simulated experiments whose 95% paired CI excludes 0.

    `mu_agent` is calibrated once by the caller -- it depends only on the gap,
    not on rho, and recalibrating per cell dominates the runtime.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        s_b, z = _scores(rng, y, mu_base)
        s_a, _ = _scores(rng, y, mu_agent, rho=rho, base=z)
        lo, _hi = paired_ci(y, s_a, s_b, n_boot, rng)
        hits += lo > 0
    return hits / n_sim


def _ap_of(y, mu, rng, reps=200):
    return float(np.mean([average_precision(y, _scores(rng, y, mu)[0])
                          for _ in range(reps)]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pos", type=int, default=N_POS_DEFAULT)
    ap.add_argument("--n-neg", type=int, default=N_NEG_DEFAULT)
    ap.add_argument("--target-ap", type=float, default=0.628,
                    help="baseline AP to power against (0.628 genetics-only)")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    y = np.r_[np.ones(args.n_pos, int), np.zeros(args.n_neg, int)]
    rng = np.random.default_rng(args.seed)
    print(f"decision set: {args.n_pos} positives / {args.n_neg} negatives "
          f"(base rate {y.mean():.1%})")
    print(f"baseline AP  : {args.target_ap}")
    print()

    mu = _mu_for_ap(args.target_ap, y, seed=args.seed)
    realised = _ap_of(y, mu, rng)
    print(f"calibrated signal mu={mu:.3f} -> AP {realised:.3f}")
    print()

    # ---- 1. how wide is a single AP estimate? ----
    s, _ = _scores(rng, y, mu)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        boot.append(average_precision(y[idx], s[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("SINGLE-MODEL PRECISION")
    print(f"  AP {realised:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}")
    print(f"  -> a lone AP number carries +-{(hi - lo) / 2:.3f}. Any headline")
    print("     comparison must be a PAIRED difference, not two separate CIs.")
    print()

    # ---- 2. paired power to detect a gap ----
    print("PAIRED POWER TO DETECT AN AP GAP  (95% CI excludes 0)")
    print("  rho = correlation between agent and classifier scores")
    print()
    gaps = [0.02, 0.04, 0.06, 0.08, 0.10]
    rhos = [0.0, 0.5, 0.8]
    print("  gap    " + "".join(f"rho={r:<7.1f}" for r in rhos))
    detect = {}
    for g in gaps:
        # Calibrated once per gap: mu_agent depends on the gap, not on rho.
        mu_agent = _mu_for_ap(min(0.999, realised + g), y, seed=args.seed)
        row = f"  +{g:.2f}  "
        for r in rhos:
            p = power_for_gap(y, mu, mu_agent, r, args.n_sim, args.n_boot, args.seed)
            row += f"{p:>6.0%}     "
            if p >= 0.80 and r not in detect:
                detect[r] = g
        print(row)
    print()
    for r in rhos:
        d = detect.get(r)
        msg = f"{d:+.2f}" if d else f">{gaps[-1]:+.2f}"
        print(f"  rho={r:.1f}: smallest gap detectable at 80% power = {msg} AP")
    print()

    # ---- 3. calibration, which uses all 1000 points ----
    # Built so the reported ECE is the induced bias, not an accident of the
    # construction: q is a genuinely calibrated posterior (labels are DRAWN
    # from it), then a fixed offset makes the model overconfident by `delta`.
    print("CALIBRATION (ECE) PRECISION, n=%d" % len(y))
    for delta in (0.05, 0.15):
        q = np.clip(rng.beta(1.2, 4.0, len(y)), 0.001, 0.999)   # ~21% mean
        yc = (rng.random(len(y)) < q).astype(int)
        p = np.clip(q + delta, 0, 1)
        e = ece(yc, p)
        b = []
        for _ in range(2000):
            idx = rng.integers(0, len(yc), len(yc))
            b.append(ece(yc[idx], p[idx]))
        elo, ehi = np.percentile(b, [2.5, 97.5])
        print(f"  induced bias {delta:.2f} -> ECE {e:.3f}  "
              f"95% CI [{elo:.3f}, {ehi:.3f}]  width {ehi - elo:.3f}")
    print("  -> ECE uses all 1,000 predictions, not just the 215 positives,")
    print("     so it resolves a smaller absolute difference than AP does.")
    print("     Published classifier ECE spans 0.154-0.222; a gap of that size")
    print("     is comfortably outside this width.")
    print()

    # ---- 4. the tie tax ----
    # agent_backtest asks for a single `probability` float. LLMs answer in
    # round numbers, so 1,000 targets collapse onto a handful of distinct
    # values and AP -- a ranking metric -- pays for every tie. This is a
    # measurement artefact that looks exactly like a worse model.
    print("TIE TAX: cost of coarse probability buckets (ties broken at random)")
    print("  buckets        mean AP    loss")
    cont = None
    for k in (None, 20, 10, 7, 5):
        aps = []
        for _ in range(300):
            s, _ = _scores(rng, y, mu)
            p = 1.0 / (1.0 + np.exp(-s))
            if k:
                p = np.round(p * k) / k
            p = p + 1e-9 * rng.standard_normal(len(p))    # random tie-breaking
            aps.append(average_precision(y, p))
        m = float(np.mean(aps))
        if cont is None:
            cont, label = m, "continuous"
        else:
            label = f"{k} buckets"
        print(f"  {label:<13} {m:.3f}     {m - cont:+.3f}")
    print("  -> an agent emitting ~5-7 distinct probabilities loses as much AP")
    print("     to ties as the smallest gap this set can detect. Ask for a")
    print("     finer score (0-100 integer) or carry a continuous tie-breaker,")
    print("     otherwise a genuinely competitive agent reads as a loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
