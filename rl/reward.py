"""Dense step-level reward for the cis-MVMR family.

WHY THIS EXISTS
---------------
Qwen3-8B passes 0/10 GeneBench-Pro problems. Under GRPO, advantage is computed
within a group of rollouts on the same prompt; if every rollout scores 0 the
advantage is identically 0 and there is NO GRADIENT. Binary pass/fail is
therefore unusable as a training signal in this regime -- you would burn
GPU-weeks on a flat loss curve.

Each generated instance carries an ablation table: the numeric output of every
known wrong analysis, tagged with the pipeline step at which it breaks. A
submitted answer that lands on one of those outputs tells us *which* mistake
was made, not merely that one was. That converts ~1 bit of almost-always-zero
signal into graded partial credit.

    step 1  scale       usable protein SD from batch-structured measurements
    step 2  alleles     align phase-2 betas to the disease effect allele
    step 3  phase       holdout betas, not winner's-cursed screening betas
    step 4  joint       LD-aware GLS over both proteins jointly
    step 5  residual    prune the latent-third-protein cluster, then refit

REWARD SHAPE
------------
    valid answer within tolerance of truth            -> 1.0
    matches a known ablation, failing at step k        -> (k - 1) / n_steps
    no clean attribution                              -> small proximity term
    malformed / non-finite / out of range              -> 0.0

DESIGN JUDGEMENTS WORTH DISAGREEING WITH
----------------------------------------
* Credit is monotone in `fails_at_step`, following the benchmark's own framing
  ("the stage at which each incomplete analysis fails"). Note this is not
  self-evidently the right ordering: `no_unit_conversion` fails at step 1 yet
  performs the full LD-aware joint fit correctly, so by work-done it is closer
  to competent than a step-5 shortcut. We keep monotone-in-step because for
  shaping purposes we want gradient toward doing step 1 first -- an agent that
  moves from failing-at-1 to failing-at-3 has genuinely fixed the scale bug.

* The reward-hacking surface is SPURIOUS ATTRIBUTION, not proximity. A
  constant predictor emitting the population-mean theta lands within a loose
  attribution radius of some ablation surprisingly often (23% at 2.0x
  tolerance) and collects step credit for nothing. `audit()` measures that
  floor explicitly and asserts on it. The radius is 0.5x tolerance as a result.

* `no_unit_conversion` outputs exceed the grader's max_value of 1.0, so step-1
  failures are rejected by the range gate rather than attributed. Both paths
  give 0 -- (1-1)/n_steps is 0 anyway -- so this is consistent, but it means
  the step-1 branch is never exercised in attribution.

* Reward is computed from the numeric answer only, never from the reasoning
  text. Judging prose would need an LLM judge: slower, unverifiable, and
  trivially hackable by asserting the right steps without performing them.

Usage:
    from rl.reward import StepReward
    r = StepReward()
    res = r.score(eval_config, {"theta_PROTA": 0.31, "theta_PROTB": 0.22})
    res.reward, res.fails_at_step, res.attribution

    venv/bin/python rl/reward.py --audit    # validate the reward itself
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

# Canonical pipeline length for this family. Read from the data where possible
# so other families with different depth work unchanged.
DEFAULT_N_STEPS = 5

# Ablations that are accepted variants rather than mistakes; they carry
# fails_at_step = None and should not be treated as failure attributions.
PASSING_ABLATIONS = ("reference", "threshold_5")



# Vendored benchmark graders usable as RL rewards (see StepReward.score). Loaded lazily, once.
_BENCH_GRADERS = {
    "medcalc_official": Path(__file__).resolve().parent.parent / "scripts" / "medcalc_reference_grader.py",
    "qrdata_official": Path(__file__).resolve().parent.parent / "benchmarks" / "qrdata" / "reference_grader.py",
    # executes against benchmarks/ehrsql2024/data/mimic_iv.sqlite (read-only), the test set's own database
    "ehrsql2024_official": Path(__file__).resolve().parent.parent / "benchmarks" / "ehrsql2024" / "reference_grader.py",
    # 10-17 (H-GEN27): MedAgentBench-style training tasks (generated/mabgen; the benchmark's own grader, recorded GETs) and
    # BioDSBench-style training tasks (generated/bdsgen; the BioDSBench grader reading generated/bdsgen/data)
    "medagentbench_refsol": Path(__file__).resolve().parent.parent / "benchmarks" / "medagentbench" / "reference_grader.py",
    "bdsgen_tests": Path(__file__).resolve().parent.parent / "scripts" / "bdsgen_reference_grader.py",
    # BioCoder training tasks (generated/biocoder; MedAgentGym's output-equality check, offline)
    "biocoder_stdout": Path(__file__).resolve().parent.parent / "scripts" / "biocoder_reference_grader.py",
}
_BENCH_MODS: dict = {}


def _bench_grader(gtype: str):
    if gtype not in _BENCH_MODS:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"_bench_grader_{gtype}", _BENCH_GRADERS[gtype])
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        _BENCH_MODS[gtype] = mod
    return _BENCH_MODS[gtype]

@dataclass
class RewardResult:
    reward: float
    passed: bool
    err: float                      # max abs error vs ground truth
    fails_at_step: int | None       # attributed step, None if unattributed
    attribution: str | None         # ablation name, or None
    attribution_err: float | None   # distance to that ablation
    component: str                  # which branch produced the reward
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["detail"] = dict(self.detail)
        return d


@dataclass
class StepReward:
    tolerance: float = 0.025        # the grader's absolute tolerance
    # Weight on the unattributed proximity branch. Small on purpose: this is
    # the only branch a data-ignoring policy can farm. See audit().
    proximity_weight: float = 0.15
    # An answer must be within this multiple of tolerance of an ablation to be
    # attributed to it. Calibrated by sweep: a data-ignoring constant predictor
    # lands inside a 2.0x radius 23% of the time (floor 0.161) but only 2% at
    # 0.5x (floor 0.052), while genuine ablation outputs sit at distance ~0 by
    # construction and are unaffected. Tight is free here.
    attribution_slack: float = 0.5
    # Require the nearest ablation to be this much closer than the runner-up,
    # otherwise the attribution is ambiguous and we fall back to proximity.
    margin_ratio: float = 1.5
    # THE MOST AN UNSOLVED ANSWER CAN EARN.
    #
    # Step credit was (step-1)/n_steps, so an analysis that got four of five
    # stages right and produced a WRONG number scored 0.80, and finishing was
    # worth only +0.20 of the range. That pricing is backwards for an agenda
    # about teaching agents to complete an analysis, and it is not theoretical:
    # RESULTS §13 vs the fixed-trace re-measurement shows 34 of 48 rollouts
    # from the old policy landing at exactly 0.800 -- every one of them the
    # same step-5 omission. A quarter of GRPO groups had zero spread and
    # therefore zero gradient, and the rest were mostly comparing 0.8 to 0.8.
    #
    # Scaling every unsolved branch by this ceiling keeps the signal dense
    # (steps 3/4/5 stay ordered and separated) while making the terminal step
    # worth more than all the others combined. 1.0 reproduces the old
    # behaviour exactly, so §19 and earlier numbers remain reproducible.
    partial_ceiling: float = 1.0
    n_steps: int = DEFAULT_N_STEPS

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _fields(cfg: dict) -> list[str]:
        """The NUMERIC graded fields.

        Two families answer with a string category -- wf_selection's
        selected_locus ("A"/"B") and mpp_qtl's high_founder ("F1".."F8") -- and
        including them here made `_vec` call float("B"), so scoring the EXACT
        ground truth raised ValueError and every rollout was recorded as
        `error:ValueError` -> 0.000. Both families therefore read as complete
        failures in every evaluation: base, bespoke, leave-one-out and the
        combined model. They are graded by `_exact` instead.
        """
        return sorted(k for k in cfg["ground_truth"]
                      if k not in StepReward._exact_keys(cfg))

    @staticmethod
    def _exact_keys(cfg: dict) -> list[str]:
        """Fields matched as strings rather than numbers.

        Taken from the instance's own composite grader when it declares them,
        and otherwise inferred from the ground truth being non-numeric, so a
        family that omits the declaration still grades correctly.
        """
        cc = ((cfg.get("grader") or {}).get("config") or {})
        declared = list(cc.get("exact_match_keys") or {})
        out = list(declared)
        for k, v in (cfg.get("ground_truth") or {}).items():
            if k in out:
                continue
            if isinstance(v, str):
                out.append(k)
        return sorted(out)

    @staticmethod
    def _exact_ok(cfg: dict, submission: dict | None) -> bool:
        keys = StepReward._exact_keys(cfg)
        if not keys:
            return True
        obj = submission or {}
        if "answer" in obj and isinstance(obj["answer"], dict):
            obj = obj["answer"]
        gt = cfg["ground_truth"]
        return all(str(obj.get(k)) == str(gt[k]) for k in keys)

    @staticmethod
    def _vec(obj: dict, keys: list[str]) -> list[float] | None:
        """Extract the graded fields as finite floats, or None if unusable."""
        if not isinstance(obj, dict):
            return None
        # accept either {"answer": {...}} or a bare answer object
        if "answer" in obj and isinstance(obj["answer"], dict):
            obj = obj["answer"]
        out = []
        for k in keys:
            if k not in obj:
                return None
            try:
                v = float(obj[k])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(v):
                return None
            out.append(v)
        return out

    def _scales(self, cfg: dict, keys: list[str]) -> list[float]:
        """Per-key grading tolerance, falling back to the scalar `tolerance`.

        The three simulated families set every key's `absolute_tolerance` to
        the same number their caller passes in as `tolerance`, so for them this
        returns a constant vector and nothing below changes by a float.
        `gwas_loci` is the first family whose keys live on different scales --
        a locus count and a mean on the log-odds scale -- and there no single
        scalar can grade both: `instance_tol` takes the MIN across keys, which
        would have graded a count of 53 loci to +/-0.02.
        """
        cc = ((cfg.get("grader") or {}).get("config") or {})
        # Composite graders declare their tolerances under `numeric_keys`, not
        # `keys`. Reading only `keys` silently fell back to the caller's blanket
        # `--tol` for every composite family -- struct_inversion, crispri,
        # txr1, wf_selection, mpp_qtl -- so their PUBLISHED per-field
        # tolerances (0.055, 0.030, 3.0 cM, ...) were never applied.
        kc = dict(cc.get("keys") or {})
        kc.update(cc.get("numeric_keys") or {})
        out = []
        for k in keys:
            t = (kc.get(k) or {}).get("absolute_tolerance")
            out.append(float(t) if t else float(self.tolerance))
        return out

    @staticmethod
    def _linf(a: list[float], b: list[float],
              scales: list[float] | None = None) -> float:
        """Max deviation IN UNITS OF EACH KEY'S TOLERANCE.

        Matches the grader's all-fields-must-pass semantics, where one bad
        field fails the submission. Normalising by the per-key tolerance means
        the pass threshold is a plain 1.0 everywhere, so a family whose keys
        have different scales is graded on each key's own terms rather than on
        the tightest one. With a constant scale vector this is the old raw
        L-infinity divided by that constant, which is why every threshold below
        that used to read `x * self.tolerance` now reads plain `x`.
        """
        if scales is None:
            return max(abs(x - y) for x, y in zip(a, b))
        return max(abs(x - y) / s for x, y, s in zip(a, b, scales))

    def _in_range(self, cfg: dict, keys: list[str], vals: list[float]) -> bool:
        cc = ((cfg.get("grader") or {}).get("config") or {})
        # Composite graders declare their tolerances under `numeric_keys`, not
        # `keys`. Reading only `keys` silently fell back to the caller's blanket
        # `--tol` for every composite family -- struct_inversion, crispri,
        # txr1, wf_selection, mpp_qtl -- so their PUBLISHED per-field
        # tolerances (0.055, 0.030, 3.0 cM, ...) were never applied.
        kc = dict(cc.get("keys") or {})
        kc.update(cc.get("numeric_keys") or {})
        for k, v in zip(keys, vals):
            c = kc.get(k) or {}
            if c.get("min_value") is not None and v < c["min_value"]:
                return False
            if c.get("max_value") is not None and v > c["max_value"]:
                return False
        return True

    # -------------------------------------------------------------------- score
    def score(self, cfg: dict, submission: dict | None) -> RewardResult:
        # 2026-09-30: instances graded by a VENDORED BENCHMARK GRADER (MedCalc-Bench / QRData training pools, the
        # healthcare RL of docs/RL_PLAN.md) are rewarded by that grader's own verdict -- the fraction of keys it passes,
        # no proximity credit (it has no ablations). Every other instance takes the path below, unchanged.
        gtype = (cfg.get("grader") or {}).get("type")
        if gtype in _BENCH_GRADERS:
            try:
                r = _bench_grader(gtype).evaluate(cfg, submission or {})
            except Exception:                                   # noqa: BLE001 -- a malformed answer earns 0
                return RewardResult(0.0, False, float("inf"), None, None, None, "malformed")
            frac = float(r.get("score") or 0.0)
            return RewardResult(round(frac, 6), bool(r.get("passed")), 0.0 if r.get("passed") else float("inf"),
                                None, "reference" if r.get("passed") else None, None,
                                "pass" if r.get("passed") else "bench_fail")
        keys = self._fields(cfg)
        exact_ok = self._exact_ok(cfg, submission)

        # EXACT-ONLY configs (BixBench-style multiple choice): every graded
        # field is categorical, so the numeric path below has nothing to do --
        # and previously crashed on max() of an empty error vector. Reward is
        # the FRACTION of questions answered correctly, matching the vendored
        # reference grader (which reports 0.667 for 2-of-3) and giving GRPO
        # within-group variance instead of an all-or-nothing signal.
        if not keys:
            ex = self._exact_keys(cfg)
            obj = submission or {}
            if "answer" in obj and isinstance(obj["answer"], dict):
                obj = obj["answer"]
            if not isinstance(obj, dict) or not ex:
                return RewardResult(0.0, False, float("inf"), None, None, None,
                                    "malformed")
            gt = cfg["ground_truth"]
            frac = sum(1 for k in ex if str(obj.get(k)) == str(gt[k])) / len(ex)
            return RewardResult(round(frac, 6), frac == 1.0,
                                0.0 if frac == 1.0 else float("inf"),
                                None, "reference" if frac == 1.0 else None,
                                None, "pass" if frac == 1.0 else "exact_partial")

        truth = [float(cfg["ground_truth"][k]) for k in keys]
        scales = self._scales(cfg, keys)

        vals = self._vec(submission or {}, keys)
        if vals is None:
            return RewardResult(0.0, False, float("inf"), None, None, None,
                                "malformed")
        if not self._in_range(cfg, keys, vals):
            return RewardResult(0.0, False, self._linf(vals, truth, scales),
                                None, None, None, "out_of_range")

        # `err` is now in units of tolerance: 1.0 means "exactly at the edge".
        err = self._linf(vals, truth, scales)
        if err <= 1.0:
            # A categorical miss fails the instance however close the numbers
            # are: naming the wrong locus or the wrong founder is a wrong
            # answer, not a near one.
            if not exact_ok:
                return RewardResult(0.0, False, err, None, None, None,
                                    "exact_mismatch")
            return RewardResult(1.0, True, err, None, "reference", 0.0, "pass")

        # --- attribute to a known wrong analysis -------------------------------
        abl = ((cfg.get("_provenance") or {}).get("ablations")) or {}
        ranked = []
        for name, a in abl.items():
            if name in PASSING_ABLATIONS or a.get("fails_at_step") is None:
                continue
            try:
                av = [float(a[k]) for k in keys]
            except (KeyError, TypeError, ValueError):
                continue
            ranked.append((self._linf(vals, av, scales), name,
                           a["fails_at_step"]))
        ranked.sort()

        n_steps = max([r[2] for r in ranked], default=self.n_steps)
        n_steps = max(n_steps, self.n_steps)

        if ranked:
            d0, name0, step0 = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else float("inf")
            close_enough = d0 <= self.attribution_slack
            unambiguous = runner_up >= self.margin_ratio * max(d0, 1e-12)
            if close_enough and unambiguous:
                credit = self.partial_ceiling * (step0 - 1) / n_steps
                return RewardResult(
                    round(credit, 6), False, err, step0, name0, d0,
                    "step_attribution",
                    {"runner_up_dist": None if runner_up == float("inf") else runner_up,
                     "n_steps": n_steps})

        # --- no clean attribution: weak, bounded proximity credit -------------
        # 1 / (1 + err) with err in tolerance units is scale-free and hits 0.5 at
        # err == one tolerance, decaying fast beyond it.
        prox = 1.0 / (1.0 + err)
        return RewardResult(
            round(self.partial_ceiling * self.proximity_weight * prox, 6), False, err,
                            None, ranked[0][1] if ranked else None,
                            ranked[0][0] if ranked else None,
                            "proximity", {"prox": round(prox, 6)})

    # -------------------------------------------------------------------- batch
    def score_many(self, cfgs: list[dict], subs: list[dict | None]) -> list[RewardResult]:
        return [self.score(c, s) for c, s in zip(cfgs, subs)]


# --------------------------------------------------------------------------- #
# validation: the reward function itself needs testing before anything trains
# --------------------------------------------------------------------------- #
def _load(split: str, limit: int, corpus: str = "cis_mvmr_1k") -> list[dict]:
    root = data_root() / "generated" / corpus / split
    if not root.exists():          # real-data corpora ship flat, not split
        root = data_root() / "generated" / corpus
    files = sorted(glob.glob(str(root / "*/eval_config.json")))[:limit]
    if not files:
        raise FileNotFoundError(f"no instances under {root}")
    return [json.loads(Path(f).read_text()) for f in files]


def audit(partial_ceiling: float = 1.0, split: str = "val", limit: int = 100,
          corpus: str = "cis_mvmr_1k") -> int:
    """Score reference policies. Establishes both ceiling and hacking floor."""
    import statistics as st

    cfgs = _load(split, limit, corpus)
    r = StepReward(partial_ceiling=partial_ceiling)
    keys = StepReward._fields(cfgs[0])
    print(f"auditing on {len(cfgs)} instances from split '{split}'\n")

    rows: list[tuple[str, list[float], list[str]]] = []

    # oracle -> must be exactly 1.0 everywhere
    rows.append(("oracle (ground truth)",
                 [r.score(c, dict(c["ground_truth"])).reward for c in cfgs], []))

    # each ablation -> must recover its own step credit
    names = [n for n in (cfgs[0].get("_provenance", {}).get("ablations") or {})
             if n not in PASSING_ABLATIONS]
    for n in names:
        vals, comps = [], []
        for c in cfgs:
            a = c["_provenance"]["ablations"][n]
            res = r.score(c, {k: a[k] for k in keys})
            vals.append(res.reward)
            comps.append(res.component)
        rows.append((f"ablation: {n}", vals, comps))

    # constant predictor -> THE reward-hacking floor
    mu = {k: st.mean(float(c["ground_truth"][k]) for c in cfgs) for k in keys}
    rows.append((f"constant = split mean {[round(mu[k],3) for k in keys]}",
                 [r.score(c, dict(mu)).reward for c in cfgs], []))

    # degenerate outputs
    rows.append(("all zeros", [r.score(c, {k: 0.0 for k in keys}).reward for c in cfgs], []))
    # A submission missing a REQUIRED key. Built as "all keys but the first",
    # which for a two-key family drops one -- but for a SINGLE-key family that
    # expression is `{keys[0]: 0.3}`, a perfectly well-formed constant
    # submission, and the audit then failed itself for scoring it above zero.
    # Drop every key instead, so the probe is malformed at any arity.
    malformed = {k: 0.3 for k in keys[1:]}
    rows.append((f"malformed (missing key{'s' if len(keys) > 2 else ''})",
                 [r.score(c, malformed).reward for c in cfgs], []))
    rows.append(("out of range (+5)",
                 [r.score(c, {k: 5.0 for k in keys}).reward for c in cfgs], []))

    print(f"{'policy':<44}{'mean':>8}{'min':>7}{'max':>7}  dominant branch")
    print("-" * 84)
    for name, vals, comps in rows:
        dom = ""
        if comps:
            dom = max(set(comps), key=comps.count)
            dom = f"{dom} ({comps.count(dom)}/{len(comps)})"
        print(f"{name:<44}{st.mean(vals):>8.3f}{min(vals):>7.3f}{max(vals):>7.3f}  {dom}")

    # assertions -- a broken reward must not silently ship
    oracle = st.mean(rows[0][1])
    const = st.mean([v for n, v, _ in rows if n.startswith("constant")][0])
    bad = st.mean([v for n, v, _ in rows if n.startswith("malformed")][0])
    # Decompose the hacking floor by branch -- lumping "not passed" together
    # hid that spurious attribution, not proximity, was the dominant term.
    cres = [r.score(c, dict(mu)) for c in cfgs]
    by: dict[str, list[float]] = {}
    for x in cres:
        by.setdefault(x.component, []).append(x.reward)
    print("\nconstant-predictor floor by branch:")
    for b, vs in sorted(by.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {b:<20} n={len(vs):>3}  contributes {sum(vs)/len(cres):.3f}")

    print()
    ok = True
    if abs(oracle - 1.0) > 1e-9:
        print(f"FAIL: oracle mean {oracle:.4f} != 1.0"); ok = False
    if bad != 0.0:
        print(f"FAIL: malformed mean {bad:.4f} != 0"); ok = False
    if const > 0.15:
        print(f"FAIL: constant-predictor floor {const:.3f} is too high -- "
              f"a policy could ignore the data and still be rewarded"); ok = False
    print("reward audit:", "PASS" if ok else "FAIL")
    print(f"  ceiling (oracle)              {oracle:.3f}")
    print(f"  hacking floor (constant)      {const:.3f}")
    print(f"  usable dynamic range          {oracle - const:.3f}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--partial-ceiling", type=float, default=1.0,
                    help="most an unsolved answer can earn; 1.0 = pre-2026-08 behaviour")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--corpus", default="cis_mvmr_1k")
    args = ap.parse_args()
    if args.audit:
        return audit(args.partial_ceiling, args.split, args.limit, args.corpus)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
