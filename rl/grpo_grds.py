"""H-GRDS -- GROUNDED-ONLY ANSWER CREDIT + DYNAMIC SAMPLING (docs/RL_PLAN.md 2026-09-29 05:35 UTC, "NEXT RL BUILD registered
as H-GRDS (before build)"). Question: is execution discipline, trained directly as the reward, enough for PR-level transfer?

REWARD, per rollout (no process bonus: --process-lambda must be 0, --process-sham off):
    r = r_answer   if the answer is GROUNDED
    r = 0          otherwise
  r_answer  = what rl/grpo_v3.py passes through at lambda 0: grpo_onpolicy's reward (StepReward at --c), and 0 under
              --require-code when no cell ran -- never recomputed.
  GROUNDED  = harness/process_signals.ProcessScorer().signals(messages, cfg, data_dir, answer=answer, n_prefix=...)
              .fired["grounded"] == 1.0: every numeric answer value appears (rounding-aware, sign included) in the STDOUT of
              an executed cell and was not typed into it as a literal. It is the value rl/grpo_v3.ProcessReward.score_group
              ALREADY computes with exactly that call (its record's "fired"), read here, never re-scored.
  A rollout whose process grpo_v3 does not score (no executed cell, or no messages) is UNGROUNDED -> 0 (already 0 under
  --require-code). An instance with NO numeric answer key (the rule does not apply, grounded None; none in causal_rdd)
  keeps r_answer: the execution gate's own n/a convention (experiments/harness_v3/exec_forcing.py), logged gr_na.
DYNAMIC SAMPLING: rl/grpo_ds.py, unchanged (--dyn-sample N required): the gate is applied inside grpo_v3's scoring, i.e.
BEFORE selection, so a group is alive (and kept) only if its GATED reward varies.

HOW, WITH NO PINNED FILE EDITED (rl/grpo_onpolicy.py, rl/grpo_v3.py, rl/grpo_ds.py are hash-pinned / running):
grpo_ds.install (which runs grpo_v3.install) is called unchanged; the reward object it returns (grpo_ds.DSReward, a
grpo_v3.ProcessReward whose score_group grpo_v3's rollout wrapper calls through `self`) is re-classed to GRDSReward, whose
changes are: score_group = grpo_v3's score_group (lambda 0), then the gate; summary / ds_log / log add the gate's counts.

LOGS (in --out). process.jsonl: r_answer = the ungated answer reward, reward = the gated one, bonus = reward - r_answer
(-r_answer where the gate zeroed it), fired.grounded as grpo_v3 logs it. process_history.jsonl gains, per step, over the
KEPT batch (and with _drawn over all N x K drawn rollouts -- read these, as for arm B):
    gr_grounded            rollouts whose answer is grounded (reward = r_answer)
    gr_ungrounded          rollouts with an answer that is NOT grounded (reward := 0) -- "the gate fired"
    gr_ungrounded_passed   of those, answers the grader PASSED (a correct answer that lost its credit)
    gr_zeroed              rollouts whose reward was > 0 before the gate and 0 after
    gr_no_answer           rollouts with no answer (reward 0 either way)
    gr_na                  rollouts on an instance where the rule does not apply (reward = r_answer)
harness_gr.json: the reward's record; a resume with another record, or of a dir with steps trained without it, is refused.

    GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_grds.py --dyn-sample 12 \
        --process-lambda 0 --process-sham off <every rl/grpo_onpolicy.py argument>
Launch through experiments/harness_v3/run_v3gr.sh (run_v3.sh's configuration and naming; run dir -hv3gr-ds<N>).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from rl import grpo_ds as DS  # noqa: E402

V3 = DS.V3
GR_VERSION = 1
GR_REWARD = ("r = r_answer iff process_signals 'grounded' fired (== 1.0, grpo_v3's own ProcessScorer.signals call), else 0; "
             "no process bonus (lambda 0); no executed cell -> ungrounded; an instance with no numeric answer key -> "
             "r_answer (n/a)")
LABELS = ("grounded", "ungrounded", "no_answer", "na")


def numeric_keys(cfg: dict) -> list:
    """The keys the grounded rule grounds (process_signals._grounded's own test: rl.reward.StepReward._fields)."""
    from rl.reward import StepReward
    try:
        return list(StepReward._fields(cfg))
    except (KeyError, TypeError):
        return []


def gate(pre: float, grounded, has_answer: bool, numeric: bool) -> tuple[str, float]:
    """(label, gated reward) of one rollout: pre = its reward at lambda 0 (r_answer, or 0 under --require-code)."""
    if not numeric:
        return "na", pre
    if grounded is not None and float(grounded) >= 1.0:
        return "grounded", pre
    if not has_answer:
        return "no_answer", 0.0
    return "ungrounded", 0.0


class GRDSReward(DS.DSReward):
    """grpo_ds's DSReward (the instance is re-classed by install): the gate after grpo_v3's scoring, and its counts."""
    _gr_drawn = None
    _gr_labels = None          # {instance name: [(label, passed, zeroed)]} for the step being scored

    def score_group(self, inst: Path, outs: list, step: int) -> list:
        recs = super().score_group(inst, outs, step)
        if self._gr_labels is None:
            self._gr_labels = {}
        cfg = json.loads((inst / "eval_config.json").read_text())
        numeric = bool(numeric_keys(cfg))
        labels = []
        for o, r in zip(outs, recs):
            pre = float(r["reward"])
            lab, total = gate(pre, (r.get("fired") or {}).get("grounded"), o.get("answer") is not None, numeric)
            r.update(reward=total, bonus=round(total - r["r_answer"], 6))
            o["reward"] = total
            o["grounded_gate"] = {"label": lab, "pre": pre}
            labels.append((lab, bool(r["passed"]), pre > 0.0 and total == 0.0))
        self._gr_labels[inst.name] = labels
        return recs

    def _gr_counts(self, insts, suffix: str) -> dict:
        labs = [x for n in insts for x in (self._gr_labels or {}).get(n, [])]
        c = {f"gr_{k}{suffix}": sum(1 for lab, _, _ in labs if lab == k) for k in LABELS}
        c[f"gr_ungrounded_passed{suffix}"] = sum(1 for lab, p, _ in labs if lab == "ungrounded" and p)
        c[f"gr_zeroed{suffix}"] = sum(1 for _, _, z in labs if z)
        return c

    def summary(self, step: int, lines: list, stats: dict, n_groups: int) -> dict:
        rec = super().summary(step, lines, stats, n_groups)
        rec.update(self._gr_counts(list(dict.fromkeys(x["inst"] for x in lines)), ""))
        if self._gr_drawn is not None:
            rec.update(self._gr_counts(self._gr_drawn, "_drawn"))
        return rec

    def ds_log(self, step: int, lines: list, drawn: list, alive: list, kept: list, handed: list, p: int) -> dict:
        self._gr_drawn = list(drawn)
        try:
            return super().ds_log(step, lines, drawn, alive, kept, handed, p)
        finally:
            self._gr_drawn = None
            self._gr_labels = {}

    def log(self, step: int, lines: list, stats: dict, n_groups: int) -> None:
        held = self._held is not None
        super().log(step, lines, stats, n_groups)
        if not held:
            self._gr_labels = {}


def config_record() -> dict:
    ps = ROOT / "harness" / "process_signals.py"
    return {"version": GR_VERSION, "reward": GR_REWARD,
            "signals_sha256": hashlib.sha256(ps.read_bytes()).hexdigest()[:16],
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}


def install(grpo_mod, dispersion_mod, v3args, dsargs, rest: list) -> DS.DynSampler:
    """grpo_ds.install (grpo_v3's harness + lambda-0 reward + dynamic sampling), then the grounded gate."""
    import argparse
    if float(v3args.process_lambda) != 0.0 or v3args.process_sham != "off":
        raise SystemExit(f"rl/grpo_grds.py: H-GRDS has NO process bonus -- --process-lambda must be 0 and --process-sham "
                         f"off (got {v3args.process_lambda} / {v3args.process_sham})")
    if not int(dsargs.dyn_sample or 0):
        raise SystemExit("rl/grpo_grds.py: H-GRDS is registered with dynamic sampling -- --dyn-sample N is required")
    pa = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pa.add_argument("--out")
    g, _ = pa.parse_known_args(rest)
    out = Path(g.out) if g.out else None
    rec = config_record()
    cfgf = out / "harness_gr.json" if out is not None else None
    if cfgf is not None:
        if cfgf.exists():
            old = json.loads(cfgf.read_text())
            if (old.get("version"), old.get("reward")) != (rec["version"], rec["reward"]):
                raise SystemExit(f"rl/grpo_grds.py: {out} was trained with {old} -- refusing to resume it as {rec}")
        elif (out / "history.jsonl").exists():
            raise SystemExit(f"rl/grpo_grds.py: {out} has steps trained WITHOUT the grounded gate (no harness_gr.json) -- "
                             "refusing to resume it")
    ds = DS.install(grpo_mod, dispersion_mod, v3args, dsargs, rest)
    if type(ds.pr) is not DS.DSReward:
        raise SystemExit(f"rl/grpo_grds.py: grpo_ds.install left a {type(ds.pr).__name__}, not grpo_ds's DSReward")
    ds.pr.__class__ = GRDSReward
    ds.pr._gr_labels = {}
    if cfgf is not None and not cfgf.exists():
        out.mkdir(parents=True, exist_ok=True)
        cfgf.write_text(json.dumps(rec, indent=1))
    print("H-GRDS (RL_PLAN 2026-09-29 05:35): answer credit ONLY when the answer is grounded (process_signals 'grounded'), "
          "else 0; no process bonus; dynamic sampling on the gated reward", flush=True)
    return ds


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(sys.argv[1:])
    dsargs, rest = DS.parse_ds_args(rest)
    sys.argv = [sys.argv[0]] + rest
    install(grpo_onpolicy, dispersion, v3args, dsargs, rest)
    raise SystemExit(grpo_onpolicy.main())
