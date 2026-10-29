"""DYNAMIC SAMPLING for the harness-v3 GRPO runner -- RECIPE PILOT arm B (docs/RL_PLAN.md 2026-09-29 02:20 / 02:30 UTC).

DAPO-style dynamic sampling: a group whose rollouts all earn the same reward has zero advantage and gives no gradient
(rl/grpo_train.advantages); once the policy collapses per prompt (epi_r0 PR s1: 7-8 of 8 groups dead from step 60) most
of each step is wasted. Arm B over-draws and trains on the groups that can still move the policy:

    step t: the trainer draws its P prompts (8; rl/grpo_onpolicy.train's seeded stream, untouched) -> this wrapper adds
    N - P more (N = --dyn-sample, 12) -> all N x K are rolled out and scored with the TOTAL reward (r_answer + process
    term, rl/grpo_v3) -> the first P groups IN DRAW ORDER whose total reward varies go to the trainer; the rest are dropped.

  * alive = the trainer's own test: >= 2 rewards (errored rollouts left out under --overflow-neutral) with pstdev >= 1e-8
    -- exactly the groups grpo_onpolicy.train would NOT count in dead_groups.
  * fewer than P alive -> the trainer gets only the alive ones (no dead padding: a dead group adds no gradient, only rows
    to the step's logged means); printed, and logged as ds_kept < P.
  * NONE alive -> the first P drawn (all dead) are handed over: zero gradient, a dead step exactly as without dynamic
    sampling. An EMPTY step would trip the trainer's instrument check (0 errored of 0 rollouts == "every rollout
    errored") and stop the run; this way the check still sees the real rollouts. Logged as ds_dead_fill = P.
  * the N - P extras: a seeded draw without replacement from the trainer's pool (every instance of the chosen instances'
    train dir, as grpo_onpolicy.main lists it) minus the step's P, keyed (seed, step): deterministic given --seed, and a
    resumed run draws what the uninterrupted run would have (no stream to replay).
  * --smoke (the trainer's 2 prompts x 4): the configured draw:keep ratio N:P (P = --prompts-per-step, default 8), at
    least one extra -> N = 12 draws 3 and keeps up to 2.
  * cost: N / P = 1.5x rollouts per step (the same --workers in flight); the update is still at most P groups.

rl/grpo_onpolicy.py and rl/grpo_v3.py are NOT edited (both are hash-pinned by the harness-v3 build tests, and the pilot's
arms A/C run the files they were registered with): this file installs grpo_v3 (v2 harness + process reward) and wraps its
rollouts, as rl/grpo_prog.py does. experiments/harness_v3/run_v3.sh launches it when EXTRA_ARGS has --dyn-sample N (run
dir -ds<N>).

    GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_ds.py --dyn-sample 12 \
        --process-lambda 0.3 <every rl/grpo_onpolicy.py argument>

LOGS (in --out). grpo_v3's files keep their meaning for the batch HANDED TO THE TRAINER: process_history.jsonl's groups,
groups_alive_answer/total, mean_p, mean_r_answer, mean_reward, cells_per_rollout and rate_* are over the kept groups, like
the trainer's history.jsonl / rollouts.jsonl (the trainer only ever sees the kept groups; its dead_groups is 0 except on a
ds_dead_fill step). Added to every process_history.jsonl line (history.jsonl is written by the pinned trainer):
    ds_drawn                    N actually drawn (12; fewer only if the pool is smaller)
    ds_alive                    groups alive among the N drawn (the selection test above)
    ds_kept                     alive groups handed to the trainer = min(P, ds_alive)
    ds_dead_fill                0, or P on a step with no alive group (see above)
    groups_alive_drawn          grpo_v3's groups_alive_total test (max - min of the total reward > 1e-12) over the N drawn
    groups_alive_answer_drawn   the same on r_answer alone
    groups_alive_drawn_scaled   groups_alive_drawn * P / ds_drawn: the alive count an ordinary P-prompt draw shows
    mean_r_answer_drawn, mean_reward_drawn, pass_rate_drawn         over all N x K drawn rollouts
process.jsonl carries every DRAWN rollout, with "ds_batch": true iff its group went to the trainer.

READING THE REGISTERED ALIVE RULE FOR ARM B. groups_alive_total is over the KEPT batch, which dynamic sampling fills with
alive groups by construction (it equals ds_kept: 8 whenever >= 8 of the 12 are alive), so it cannot show a collapse and
must NOT be read for arm B. The policy's state is read on the DRAWN batch, sampled exactly as arms A/C's 8 prompts:
    clause 1 (groups):  mean over steps 51-100 of groups_alive_drawn_scaled  (= 8 * groups_alive_drawn / ds_drawn) >= 3,
                        i.e. the alive fraction of drawn groups >= 3/8 -- A/C's own threshold, on 12 draws instead of 8;
    clause 3 (answer):  mean over steps 91-100 of mean_r_answer_drawn >= 0.30 (the kept batch's mean_r_answer is biased by
                        the selection, which drops all-pass and all-fail groups);
    clause 2 (nll):     history.jsonl's nll, as for A/C (in every arm it is over the sequences that got a gradient).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from rl import grpo_v3 as V3  # noqa: E402

DS_VERSION = 1
DS_SALT = 0x4453          # "DS": the extra draw's stream, apart from grpo_v3.sham_perm's (seed, step, crc(instance))
DEFAULT_P = 8             # rl/grpo_onpolicy.py's --prompts-per-step default
ALIVE_SD = 1e-8           # rl/grpo_train.advantages: a group with pstdev below this gets zero advantages


def n_draw(n: int, p: int, p_cfg: int) -> int:
    """Prompts to draw when the trainer asks for p: n at the configured batch (p == p_cfg, n > p_cfg); otherwise (the
    trainer's --smoke, p = 2) the same draw:keep ratio, at least one extra (12 at 8 -> 3 at 2)."""
    return max(p + 1, math.ceil(n * p / p_cfg))


def pool_of(insts) -> list:
    """The trainer's pool: every instance (a dir with eval_config.json) of the chosen instances' train dir(s), sorted --
    for one train dir exactly grpo_onpolicy.main's `insts`."""
    parents = sorted({Path(i).parent for i in insts})
    return sorted(q for d in parents for q in d.iterdir() if (q / "eval_config.json").is_file())


def draw_extra(chosen, pool, n_extra: int, seed: int, step: int) -> list:
    """n_extra instances of the pool, none of the step's chosen, without replacement, in a seeded order keyed
    (seed, step): deterministic, and the same on a resumed run."""
    import numpy as np
    names = {Path(c).name for c in chosen}
    cands = [q for q in pool if Path(q).name not in names]
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, max(int(step), 0), DS_SALT])
    return [cands[int(i)] for i in rng.permutation(len(cands))[:max(int(n_extra), 0)]]


def group_alive(outs: list, overflow_neutral: bool = False) -> bool:
    """grpo_onpolicy.train's test: the rewards the group is standardised over (errored rollouts left out under
    --overflow-neutral) give a non-zero advantage iff there are >= 2 of them with pstdev >= 1e-8."""
    rs = [float(o["reward"]) for o in outs
          if not (overflow_neutral and str(o.get("attribution") or "").startswith("error"))]
    return len(rs) >= 2 and statistics.pstdev(rs) >= ALIVE_SD


def select(groups: dict, p: int, alive_fn=group_alive) -> tuple:
    """(kept, alive, handed): alive = the drawn groups (draw order) that pass alive_fn; kept = the first p of them;
    handed = what the trainer gets -- kept, or, when nothing is alive, the first p drawn (all dead: zero gradient)."""
    alive = [name for name, outs in groups.items() if alive_fn(outs)]
    kept = alive[:p]
    return kept, alive, (kept if kept else list(groups)[:p])


class DSReward(V3.ProcessReward):
    """grpo_v3's ProcessReward (its instance is re-classed after grpo_v3.install): while a dynamic-sampling step is
    rolled out, grpo_v3's per-step log is held back; ds_log writes it once the kept groups are known."""
    _held = None

    def log(self, step: int, lines: list, stats: dict, n_groups: int) -> None:
        if self._held is None:
            return super().log(step, lines, stats, n_groups)
        self._held.update(step=step, lines=lines)

    def ds_log(self, step: int, lines: list, drawn: list, alive: list, kept: list, handed: list, p: int) -> dict:
        by: dict = {}
        for x in lines:
            by.setdefault(x["inst"], []).append(x)

        def n_alive(names, key):     # grpo_v3's groups_alive_* test
            return sum(1 for n in names if by.get(n)
                       and max(x[key] for x in by[n]) - min(x[key] for x in by[n]) > 1e-12)

        hs = set(handed)
        kl = [x for x in lines if x["inst"] in hs]
        rec = self.summary(step, kl, {"answer_alive": n_alive(handed, "r_answer"),
                                      "total_alive": n_alive(handed, "reward")}, len(handed))
        ga = n_alive(drawn, "reward")

        def mean(key):
            return round(statistics.fmean([float(x[key]) for x in lines]), 4) if lines else 0.0
        rec.update({"ds_drawn": len(drawn), "ds_alive": len(alive), "ds_kept": len(kept),
                    "ds_dead_fill": 0 if kept else len(handed),
                    "groups_alive_drawn": ga, "groups_alive_answer_drawn": n_alive(drawn, "r_answer"),
                    "groups_alive_drawn_scaled": round(ga * p / len(drawn), 4) if drawn else 0.0,
                    "mean_r_answer_drawn": mean("r_answer"), "mean_reward_drawn": mean("reward"),
                    "pass_rate_drawn": mean("passed")})
        print("process " + json.dumps(rec), flush=True)
        if self.out is not None:
            with self.lock:
                with open(self.out / "process.jsonl", "a") as f:
                    for x in lines:
                        f.write(json.dumps({**x, "ds_batch": x["inst"] in hs}, default=str) + "\n")
                with open(self.out / "process_history.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return rec


class DynSampler:
    """Installed around grpo_v3's rollouts (so every drawn group is scored with the total reward before selection)."""

    def __init__(self, pr: DSReward, n: int, p_cfg: int, seed: int, overflow_neutral: bool = False):
        self.pr, self.n, self.p_cfg, self.seed = pr, int(n), int(p_cfg), int(seed)
        self.overflow_neutral = bool(overflow_neutral)
        self._pools: dict = {}

    def pool(self, insts) -> list:
        key = tuple(sorted({str(Path(i).parent) for i in insts}))
        if key not in self._pools:
            self._pools[key] = pool_of(insts)
        return self._pools[key]

    def wrap(self, orig_rollouts):
        def ds_rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c, require_code=False, demo_for=None):
            insts = list(insts)
            p, step = len(insts), V3.step_of(spec)
            drawn = insts + draw_extra(insts, self.pool(insts), n_draw(self.n, p, self.p_cfg) - p, self.seed, step)
            self.pr._held = {}
            try:
                groups = orig_rollouts(drawn, spec, k, temperature, timeout, n_steps, workers, c,
                                       require_code=require_code, demo_for=demo_for)
                held = self.pr._held
            finally:
                self.pr._held = None
            kept, alive, handed = select(groups, p, lambda outs: group_alive(outs, self.overflow_neutral))
            self.pr.ds_log(held.get("step", step), held.get("lines", []), list(groups), alive, kept, handed, p)
            if len(kept) < p:
                print(f"dyn-sample step {step}: {len(alive)}/{len(groups)} drawn groups alive -> "
                      + (f"training on {len(kept)} < {p} groups (no dead padding)" if kept else
                         f"NONE alive: the first {len(handed)} drawn (all dead, zero gradient) handed over"),
                      flush=True)
            return {name: groups[name] for name in handed}

        ds_rollouts.dyn_sampler = self
        ds_rollouts.process_reward = self.pr
        return ds_rollouts


def config_record(n: int, p_cfg: int) -> dict:
    return {"version": DS_VERSION, "dyn_sample": n, "prompts_per_step": p_cfg, "salt": DS_SALT,
            "select": "first P groups in draw order with >= 2 rewards and pstdev >= 1e-8 (rl/grpo_train.advantages); "
                      "none alive -> the first P drawn",
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}


def parse_ds_args(argv: list) -> tuple:
    # allow_abbrev=False: no trainer option may be read as an abbreviation of --dyn-sample
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--dyn-sample", type=int, default=0)
    return ap.parse_known_args(argv)


def install(grpo_mod, dispersion_mod, v3args, dsargs, rest: list) -> DynSampler:
    """grpo_v3.install (v2 harness + process reward) on grpo_mod, then dynamic sampling around its rollouts."""
    n = int(dsargs.dyn_sample)
    pa = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pa.add_argument("--out")
    pa.add_argument("--seed", type=int, default=1)
    pa.add_argument("--prompts-per-step", type=int, default=DEFAULT_P)
    pa.add_argument("--overflow-neutral", action="store_true")
    g, _ = pa.parse_known_args(rest)
    if n <= g.prompts_per_step:
        raise SystemExit(f"rl/grpo_ds.py: --dyn-sample {n} must exceed --prompts-per-step {g.prompts_per_step} (N is the "
                         "number of prompts DRAWN per step; without dynamic sampling run rl/grpo_v3.py)")
    out = Path(g.out) if g.out else None
    rec = config_record(n, g.prompts_per_step)
    cfgf = out / "harness_ds.json" if out is not None else None
    if cfgf is not None:
        if cfgf.exists():
            old = json.loads(cfgf.read_text())
            if (old.get("dyn_sample"), old.get("prompts_per_step")) != (rec["dyn_sample"], rec["prompts_per_step"]):
                raise SystemExit(f"rl/grpo_ds.py: {out} was trained with {old} -- refusing to resume it as {rec}")
        elif (out / "history.jsonl").exists():
            raise SystemExit(f"rl/grpo_ds.py: {out} has steps trained WITHOUT dynamic sampling (no harness_ds.json) -- "
                             "refusing to resume it with --dyn-sample")
    pr = V3.install(grpo_mod, dispersion_mod, v3args, rest)
    if type(pr) is not V3.ProcessReward:
        raise SystemExit(f"rl/grpo_ds.py: grpo_v3.install returned {type(pr).__name__}, not grpo_v3's ProcessReward")
    if cfgf is not None and not cfgf.exists():
        out.mkdir(parents=True, exist_ok=True)
        cfgf.write_text(json.dumps(rec, indent=1))
    pr.__class__ = DSReward
    ds = DynSampler(pr, n, g.prompts_per_step, g.seed, g.overflow_neutral)
    grpo_mod.rollouts = ds.wrap(grpo_mod.rollouts)
    print(f"dynamic sampling (RL_PLAN 2026-09-29 recipe pilot arm B): draw {n} prompts per step (the trainer's "
          f"{g.prompts_per_step} + {n - g.prompts_per_step} seeded extras), train on the first {g.prompts_per_step} "
          f"whose total reward varies (fewer if fewer are alive); smoke: {n_draw(n, 2, g.prompts_per_step)} for 2",
          flush=True)
    return ds


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(sys.argv[1:])
    dsargs, rest = parse_ds_args(rest)
    sys.argv = [sys.argv[0]] + rest
    install(grpo_onpolicy, dispersion, v3args, dsargs, rest)
    raise SystemExit(grpo_onpolicy.main())
