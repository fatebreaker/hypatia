"""HARNESS V3 TRAINER: rl/grpo_onpolicy.py (unchanged) with its rollouts in the v2 harness the GBP_* switches describe
(harness/v2.py: KERNEL, science prompt, notebook echo -- NO library), plus a PROCESS REWARD (docs/RL_PLAN.md s.15).

    GBP_KERNEL=1 GBP_SYSTEM=science GBP_ECHO_LAST=1 venv_vllm/bin/python rl/grpo_v3.py \
        --process-lambda 0.3 [--process-sham perm] <every rl/grpo_onpolicy.py argument>

Launch through experiments/harness_v3/run_v3.sh, which names the run directory -hv3pr<lambda>[sham] and writes
harness_v2.json + harness_v3.json.

REWARD, per rollout (the group statistics, clipping and update are grpo_onpolicy.train's, untouched):
    r = r_answer + bonus,   bonus = lambda * p                               (passed)
                            bonus = min(lambda * p, 1 - MARGIN - r_answer)   (not passed; MARGIN = 0.01)
  r_answer = the reward grpo_onpolicy.rollouts already computed (StepReward at --c, and 0 under --require-code when no
  cell ran) -- used as it is, never recomputed. p in [0, 1] = harness/process_signals.py's score: the mean over the
  APPLICABLE signals of {inspect, strata, check, spec, grounded}, each verified from the executed cells, their STDOUT and
  the task's data files. A passed rollout earns >= 1.0 and an unpassed one < 1.0 whatever p is, so a correct answer
  always out-scores an incorrect one; with the registered c = 0.5 an unpassed r_answer is < 0.5 and the cap never binds
  at lambda <= 0.49. lambda = 0 returns r_answer itself (bit-identical to rl/grpo_v2.py).
  A rollout with no executed cell has p = 0 (every signal needs executed code), so --require-code's zero stands.

PROCESS-SHAM (--process-sham perm, the registered control arm): the same detectors and lambda, but each rollout gets
the p of a UNIFORMLY RANDOM rollout of its own GRPO group (a permutation drawn from (seed, step, instance)). The group's
multiset of bonuses -- hence the extra reward variance, the groups kept alive after the answer saturates, the scale --
is exactly the real arm's at that policy, but which rollout earns it is independent of what that rollout did. A gain of
PR over SHAM is therefore the process CONTENT, not "any extra reward signal". (--process-sham cols scores the detectors
against a decoy column set the task does not have: measured ~0 on stored rollouts, i.e. it would be a lambda = 0 arm, so
it is available for measurement only.)

LOGS (in --out): process.jsonl -- one line per rollout: step, inst, r_answer, passed, p (real), p_used (after the sham),
bonus, reward, cells, the value of every signal (None = not applicable), the trap modes checked, and the evidence
locations; process_history.jsonl -- one line per step: signal rates, mean p, groups alive on the answer alone vs on the
total reward. rollouts.jsonl / history.jsonl are grpo_onpolicy's and carry the TOTAL reward.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

MARGIN = 0.01
SHAM_MODES = ("off", "perm", "cols")
V3_VERSION = 1


def compose(r_answer: float, passed: bool, p: float, lam: float, margin: float = MARGIN) -> float:
    """Answer reward + bounded process term (module docstring). lam == 0 returns r_answer unchanged."""
    if not lam:
        return r_answer
    bonus = lam * max(0.0, min(1.0, float(p)))
    if not passed:
        bonus = min(bonus, max(0.0, 1.0 - margin - r_answer))
    return r_answer + bonus


def step_of(spec: str) -> int:
    """grpo_onpolicy.train samples step t from the policy published as ...-pol<t-1>."""
    m = re.search(r"-pol(\d+)(?::|$)", spec or "")
    return int(m.group(1)) + 1 if m else -1


def sham_perm(seed: int, step: int, inst: str, n: int) -> list:
    import numpy as np
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, max(step, 0), zlib.crc32(inst.encode())])
    return [int(i) for i in rng.permutation(n)]


class ProcessReward:
    """Installed around grpo_onpolicy.rollouts: scores every rollout's process, composes the reward, logs."""

    def __init__(self, lam: float, sham: str, out: Path | None, seed: int, require_code: bool = True):
        from process_signals import ProcessScorer, decoy_vocab
        if sham not in SHAM_MODES:
            raise ValueError(f"--process-sham must be one of {SHAM_MODES}")
        self.lam, self.sham, self.out, self.seed = float(lam), sham, out, int(seed)
        self.scorer = ProcessScorer(vocab_override=decoy_vocab if sham == "cols" else None)
        self.lock = threading.Lock()
        self.require_code = require_code

    def score_group(self, inst: Path, outs: list, step: int) -> list:
        from process_signals import SIGNALS
        cfg = json.loads((inst / "eval_config.json").read_text())
        recs = []
        for o in outs:
            r_answer = float(o["reward"])
            cells = int(o.get("n_cells") or 0)
            if o.get("messages") and cells > 0:
                s = self.scorer.signals(o["messages"], cfg, inst / "data_files", answer=o.get("answer"),
                                        n_prefix=int(o.get("n_prefix") or 0))
                p, fired, modes = s.score, s.fired, s.modes
                where = {k: (v if isinstance(v, str) else (v.get("where") if isinstance(v, dict) else None))
                         for k, v in s.detail.items() if k in SIGNALS}
            else:
                p, fired, modes, where = 0.0, {k: None for k in SIGNALS}, [], {}
            recs.append({"r_answer": r_answer, "passed": bool(o.get("passed")), "p": round(p, 6), "fired": fired,
                         "modes": modes, "where": where, "cells": cells,
                         "eligible": bool(o.get("messages")) and (cells > 0 or not self.require_code)})
        used = [r["p"] for r in recs]
        if self.sham == "perm":
            elig = [i for i, r in enumerate(recs) if r["eligible"]]
            perm = sham_perm(self.seed, step, inst.name, len(elig))
            for dst, k in zip(elig, perm):
                used[dst] = recs[elig[k]]["p"]
        for o, r, pu in zip(outs, recs, used):
            total = compose(r["r_answer"], r["passed"], pu, self.lam)
            if self.require_code and not r["cells"]:
                total = 0.0
            r.update(p_used=round(pu, 6), bonus=round(total - r["r_answer"], 6), reward=total)
            o["reward"] = total
            o["r_answer"] = r["r_answer"]
            o["process"] = {"p": r["p"], "p_used": r["p_used"], "fired": r["fired"]}
        return recs

    def wrap(self, orig_rollouts):
        def v3_rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c, require_code=False, demo_for=None):
            if demo_for is not None:
                raise SystemExit("rl/grpo_v3.py: an in-context library (--library / demo_for) is not harness v3")
            groups = orig_rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c,
                                   require_code=require_code, demo_for=demo_for)
            step = step_of(spec)
            by_name = {i.name: i for i in insts}
            lines, stats = [], {"answer_alive": 0, "total_alive": 0}
            for name, outs in groups.items():
                recs = self.score_group(by_name[name], outs, step)
                ra = [r["r_answer"] for r in recs]
                rt = [r["reward"] for r in recs]
                stats["answer_alive"] += (max(ra) - min(ra)) > 1e-12 if ra else 0
                stats["total_alive"] += (max(rt) - min(rt)) > 1e-12 if rt else 0
                for r in recs:
                    lines.append({"step": step, "inst": name, **{k: r[k] for k in (
                        "r_answer", "passed", "p", "p_used", "bonus", "reward", "cells", "fired", "modes", "where")}})
            self.log(step, lines, stats, len(groups))
            return groups

        v3_rollouts.process_reward = self
        return v3_rollouts

    def summary(self, step: int, lines: list, stats: dict, n_groups: int) -> dict:
        from process_signals import SIGNALS
        rec = {"step": step, "lambda": self.lam, "sham": self.sham, "groups": n_groups,
               "groups_alive_answer": stats["answer_alive"], "groups_alive_total": stats["total_alive"],
               "mean_p": round(statistics.fmean([x["p"] for x in lines]), 4) if lines else 0.0,
               "mean_p_used": round(statistics.fmean([x["p_used"] for x in lines]), 4) if lines else 0.0,
               "mean_r_answer": round(statistics.fmean([x["r_answer"] for x in lines]), 4) if lines else 0.0,
               "mean_reward": round(statistics.fmean([x["reward"] for x in lines]), 4) if lines else 0.0,
               "cells_per_rollout": round(statistics.fmean([x["cells"] for x in lines]), 2) if lines else 0.0}
        for s in SIGNALS:
            vals = [x["fired"][s] for x in lines if x["fired"].get(s) is not None]
            rec[f"rate_{s}"] = round(sum(vals) / len(vals), 4) if vals else None
            rec[f"n_{s}"] = len(vals)
        return rec

    def log(self, step: int, lines: list, stats: dict, n_groups: int) -> None:
        rec = self.summary(step, lines, stats, n_groups)
        print("process " + json.dumps(rec), flush=True)
        if self.out is None:
            return
        with self.lock:
            with open(self.out / "process.jsonl", "a") as f:
                for x in lines:
                    f.write(json.dumps(x, default=str) + "\n")
            with open(self.out / "process_history.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


def config_record(lam: float, sham: str) -> dict:
    import hashlib
    ps = ROOT / "harness" / "process_signals.py"
    return {"version": V3_VERSION, "process_lambda": lam, "process_sham": sham, "margin": MARGIN,
            "signals_sha256": hashlib.sha256(ps.read_bytes()).hexdigest()[:16],
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}


def parse_v3_args(argv: list) -> tuple[argparse.Namespace, list]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--process-lambda", type=float, default=0.3)
    ap.add_argument("--process-sham", choices=SHAM_MODES, default="off")
    return ap.parse_known_args(argv)


def install(grpo_mod, dispersion_mod, v3: argparse.Namespace, rest: list):
    """Install the v2 harness (as rl/grpo_v2.py) and the process reward into rl.grpo_onpolicy; returns the reward."""
    import v2
    hc = v2.HarnessConfig.from_env()
    if hc.library:
        raise SystemExit("rl/grpo_v3.py: GBP_LIBRARY is set -- harness v3 has NO library in training prompts")
    if hc.siblings or hc.profile or hc.apidocs or hc.checks:
        raise SystemExit("rl/grpo_v3.py: harness v3 is KERNEL + science prompt + echo only; unset GBP_SIBLINGS/"
                         "GBP_PROFILE/GBP_APIDOCS/GBP_CHECKS")
    # --init-adapter only through run_v3.sh INIT_ADAPTER (sets HV3_INIT_ADAPTER=1): RL from an SFT policy, the
    # H-MC-DISTILL stage-2 arm (docs/RL_PLAN.md 2026-09-30 21:52 UTC). Any other use is still refused.
    allow_init = os.environ.get("HV3_INIT_ADAPTER") == "1"
    for bad in ("--library", "--transfer-check", "--siblings", "--init-adapter"):
        if bad == "--init-adapter" and allow_init:
            continue
        if any(a == bad or a.startswith(bad + "=") for a in rest):
            raise SystemExit(f"rl/grpo_v3.py: {bad} is not part of harness v3")
    if not 0.0 <= v3.process_lambda < 1.0:
        raise SystemExit("rl/grpo_v3.py: --process-lambda must be in [0, 1)")
    pa = argparse.ArgumentParser(add_help=False)
    pa.add_argument("--out")
    pa.add_argument("--seed", type=int, default=1)
    pa.add_argument("--c", type=float, default=0.5)
    pa.add_argument("--require-code", action="store_true")
    g, _ = pa.parse_known_args(rest)
    out = Path(g.out) if g.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cfgf = out / "harness_v3.json"
        rec = config_record(v3.process_lambda, v3.process_sham)
        if cfgf.exists():
            old = json.loads(cfgf.read_text())
            if (old.get("process_lambda"), old.get("process_sham")) != (rec["process_lambda"], rec["process_sham"]):
                raise SystemExit(f"rl/grpo_v3.py: {out} was trained with {old} -- refusing to resume it as {rec}")
            if old.get("signals_sha256") != rec["signals_sha256"]:
                print(f"rl/grpo_v3.py: WARNING process_signals.py changed since this run started "
                      f"({old.get('signals_sha256')} -> {rec['signals_sha256']})", flush=True)
        else:
            cfgf.write_text(json.dumps(rec, indent=1))
    v2.install_grpo(grpo_mod, dispersion_mod, hc)
    pr = ProcessReward(v3.process_lambda, v3.process_sham, out, g.seed, require_code=g.require_code)
    grpo_mod.rollouts = pr.wrap(grpo_mod.rollouts)
    print(v2.banner(hc) + f" | harness v3: process reward lambda={v3.process_lambda} sham={v3.process_sham}",
          flush=True)
    return pr


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = parse_v3_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    install(grpo_onpolicy, dispersion, v3args, rest)
    raise SystemExit(grpo_onpolicy.main())
