"""Read a verl GRPO log and say whether the reward curve is going anywhere.

    venv/bin/python scripts/watch_grpo.py $DEEP_DATA_ROOT/logs/verl_grpo_run1.log

WHY THIS EXISTS
---------------
"Watch the reward curve" is easy to do badly. A 500-step run produces a noisy
line that the eye will happily read a trend into, and this project has already
been burned once by exactly that: the hand-rolled pilot's flat curve was
briefly reported as "no learning at this scale" when it was really a missing
weight sync. So this prints the curve, but it also prints the three things that
decide whether the curve means anything:

  1. A slope with a bootstrap CI, not a slope alone. If the CI spans zero, the
     honest statement is "no detectable trend yet", not "it is not learning".
  2. rollout_probs_diff -- the rollout/actor logprob gap. If this drifts up,
     the policy being served has come apart from the policy being trained and
     NOTHING else on the page can be trusted. It is the pilot's bug as a number.
  3. The fraction of steps whose advantages are all-but-zero. GRPO's advantage
     is group-relative, so a group whose rollouts all score the same produces
     no gradient regardless of how good or bad that score is. A curve can sit
     flat because the task is saturated (or impossible) rather than because
     learning failed, and those look identical in the reward alone.

TRAIN REWARD IS NOT THE RESULT. It is whatever family is in train.parquet.
The families are NOT hardcoded here any more: these labels once read
"cis-MVMR" and "held-out scRNA" unconditionally, which made a scRNA-on-scRNA
continuation run look like a cross-family transfer experiment to anyone
reading the output. The experiment is the val-core line, which
is measured far less often. Both are printed; do not confuse them.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

STEP_RE = re.compile(r"step:(\d+) - (.*)")
# verl prints "key:value" pairs joined by " - ", and values themselves contain
# things like "np.float64(0.02)", so split on the separator rather than on "-".
# The key charset MUST include "-": the held-out metric is
# "val-core/scrna_1k/reward/mean@1", and without the hyphen this silently
# matched "core/scrna_1k/..." instead, so the validation section never printed
# and the run looked like it had no held-out numbers at all. Spaces are still
# excluded, so the " - " separator between pairs cannot be absorbed.
PAIR_RE = re.compile(r"([a-zA-Z_][\w/@.-]*):((?:np\.\w+\()?[-\d.e+]+\)?)")


def _num(raw: str) -> float | None:
    """Parse a verl metric value, which may be bare or numpy-wrapped.

    verl prints some metrics as "np.float64(0.0272)" / "np.int64(23)". The
    wrapper must be stripped BEFORE scanning for digits: searching the raw
    string finds the 64 in "float64" first, which silently turned every
    held-out validation score into 64.0.
    """
    inner = re.match(r"np\.\w+\((.*)\)?$", raw)
    if inner:
        raw = inner.group(1)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    return float(m.group()) if m else None


def parse(path: Path) -> list[dict]:
    """One dict per logged step. Later lines win, so a resumed run is fine."""
    steps: dict[int, dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        row = steps.setdefault(step, {"step": step})
        for key, raw in PAIR_RE.findall(m.group(2)):
            val = _num(raw)
            if val is not None:
                row[key] = val
    return [steps[k] for k in sorted(steps)]


def sparkline(vals: list[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return blocks[0] * len(vals)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def slope_ci(xs: list[float], ys: list[float], n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float, float]:
    """OLS slope per 100 steps, with a bootstrap CI over resampled points.

    The CI is what keeps this honest: a positive point estimate whose interval
    covers zero is not evidence of learning.
    """
    def fit(pairs) -> float:
        n = len(pairs)
        mx = sum(p[0] for p in pairs) / n
        my = sum(p[1] for p in pairs) / n
        den = sum((p[0] - mx) ** 2 for p in pairs)
        if den == 0:
            return 0.0
        return sum((p[0] - mx) * (p[1] - my) for p in pairs) / den

    pairs = list(zip(xs, ys))
    point = fit(pairs)
    rng = random.Random(seed)
    boots = sorted(fit([rng.choice(pairs) for _ in pairs]) for _ in range(n_boot))
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return point * 100, lo * 100, hi * 100


def _family(which: str, log: Path | None = None) -> str:
    """Family name for the train/val side of THE RUN THIS LOG BELONGS TO.

    Read it out of the log's own header, which prints

        train   : /path/to/train.parquet   (rare_burden_1k)

    Reading the parquet at a fixed path instead is wrong twice over. It assumes
    every run uses $DEEP_DATA_ROOT/verl_data, which stopped being true when
    VERL_DATA was added so a second family could train without clobbering the
    first -- and it reports whatever is on disk NOW rather than what this run
    actually loaded, so a later rebuild silently relabels an old log. Both
    failures look like a correct label. The burden run was announced as
    "scrna_1k" for exactly this reason.
    """
    if log is not None:
        try:
            head = log.read_text(errors="ignore")[:4000]
        except OSError:
            head = ""
        m = re.search(rf"^{which}\s+:\s+\S+\s+\(([^)]*?)(?:,\s*HELD OUT)?\)\s*$",
                      head, re.M)
        if m and m.group(1).strip():
            return m.group(1).strip()

    import os
    root = Path(os.environ.get("VERL_DATA")
                or (data_root()
                    / "verl_data"))
    f = root / f"{which}.parquet"
    try:
        import pandas as pd
        u = sorted(set(pd.read_parquet(f, columns=["data_source"])["data_source"]))
    except Exception:
        return f"unreadable:{f.name}"
    return "+".join(u) if u else f"empty:{f.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--tail", type=int, default=40,
                    help="how many recent steps to draw (default 40)")
    args = ap.parse_args()

    if not args.log.is_file():
        print(f"no such log: {args.log}", file=sys.stderr)
        return 1

    rows = parse(args.log)
    train = [r for r in rows if "critic/score/mean" in r]
    val = [(r["step"], r["val-core/scrna_1k/reward/mean@1"])
           for r in rows if "val-core/scrna_1k/reward/mean@1" in r]

    if not train:
        print("no completed training steps in this log yet")
        if val:
            print(f"initial held-out validation: {val[0][1]:.4f} (step {val[0][0]})")
        return 0

    ys = [r["critic/score/mean"] for r in train]
    xs = [float(r["step"]) for r in train]

    print(f"{len(train)} steps  |  train reward ({_family('train', args.log)}, "
          f"the family being trained)")
    shown = train[-args.tail:]
    print(f"  {sparkline([r['critic/score/mean'] for r in shown])}")
    print(f"  step {shown[0]['step']} -> {shown[-1]['step']}   "
          f"first {ys[0]:.3f}   last {ys[-1]:.3f}   "
          f"min {min(ys):.3f}   max {max(ys):.3f}")

    if len(train) >= 8:
        pt, lo, hi = slope_ci(xs, ys)
        verdict = ("rising" if lo > 0 else
                   "falling" if hi < 0 else
                   "no detectable trend (CI covers zero)")
        print(f"  slope {pt:+.4f} per 100 steps  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    else:
        print("  slope: need >=8 steps before a trend is worth quoting")

    # --- the two things that decide whether the curve above means anything ---
    diffs = [r["training/rollout_probs_diff_mean"] for r in train
             if "training/rollout_probs_diff_mean" in r]
    if diffs:
        recent = diffs[-10:]
        flag = "  <-- CHECK: rollout has drifted from the actor" if max(recent) > 0.05 else ""
        print(f"\nsync   rollout_probs_diff_mean  first {diffs[0]:.5f}  "
              f"last {diffs[-1]:.5f}  max(last 10) {max(recent):.5f}{flag}")

    amax = [r["critic/advantages/max"] for r in train if "critic/advantages/max" in r]
    if amax:
        dead = sum(1 for a in amax if abs(a) < 1e-6)
        print(f"signal advantages/max: {dead}/{len(amax)} steps ~zero "
              f"(zero-variance groups give GRPO no gradient)")

    # Entropy is the leading indicator for the above. There is no KL term in
    # this run, so nothing but the clip ratio holds the policy near its SFT
    # init; if entropy collapses the rollouts within a group become identical,
    # advantages go to zero, and the reward curve flattens for a reason that
    # has nothing to do with the task.
    # With a KL term the question is whether it is actually binding. kl_loss
    # rising then levelling off is the policy moving away from its reference
    # and being held; kl_loss pinned at ~0 while train reward is also flat
    # means the penalty is too strong and nothing can move.
    kl = [r["actor/kl_loss"] for r in train if "actor/kl_loss" in r]
    coef = next((r["actor/kl_coef"] for r in train if "actor/kl_coef" in r), None)
    if kl and coef:
        note = ""
        if len(kl) >= 10 and max(kl[-10:]) < 1e-6:
            note = ("  <-- pinned at zero: either nothing has moved yet, or "
                    "the penalty is too strong")
        print(f"       kl_loss (coef {coef:g}): first {kl[0]:.5f}  "
              f"last {kl[-1]:.5f}  max {max(kl):.5f}{note}")

    ent = [r["actor/entropy"] for r in train if "actor/entropy" in r]
    if ent:
        drop = (ent[-1] - ent[0]) / ent[0] * 100 if ent[0] else 0.0
        flag = "  <-- CHECK: near-deterministic, groups will stop varying" if ent[-1] < 0.005 else ""
        print(f"       entropy: first {ent[0]:.4f}  last {ent[-1]:.4f}  "
              f"({drop:+.0f}%){flag}")

    if val:
        vf, tf = _family("val", args.log), _family("train", args.log)
        kind = "transfer" if vf != tf else "IN-FAMILY held out, not transfer"
        print(f"\nheld-out {vf} ({kind}) -- not the curve above")
        for step, v in val[-8:]:
            print(f"  step {step:>4}  {v:.4f}")
        if len(val) >= 2:
            print(f"  baselines: SFT 0.022 (RESULTS §15), in-corpus ceiling 0.606 (§16)")
            print(f"  NOTE: repeat measurements of the same untrained policy spanned "
                  f"0.0065-0.0282, so single points are not comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
