"""DISTILLATION TRAINER -- H-DISTILL27-9 stage B (docs/RL_PLAN.md, registered 2026-10-08 15:29 UTC): rl/grpo_v3.py installed
exactly as it installs itself (--process-lambda 0 --process-sham off), with two changes, both in this file:

  1. ROLLOUTS -> TEACHER EPISODES. Each step's chosen training instances get the TEACHER's episode for that instance instead
     of k policy samples: the RL-trained 27B's transcript, correct by the grader and grounded by the audit label v2, rebuilt
     into training messages by scratchpad/build_distill_set.py with rl/dispersion.conversation (system prompt and task
     exactly as one_rollout builds them for the policy). One episode per instance; the pool (generated/<family>/train)
     must hold only instances that have one -- any other instance stops the run.
  2. ADVANTAGE a = 1 for every teacher episode, so each step is maximum likelihood on the teacher's assistant tokens (the
     trainer's ratio is 1 by construction, so the clip never binds; the loss is the token-mean NLL per episode).

Nothing else changes: same LoRA, render, masking (assistant turns only), max length, optimizer and checkpointing as the RL
runs, so stage C (rl/grpo_rg.py from this adapter, INIT_ADAPTER) continues in the same parameterisation. The policy server
is still published every step (checkpoints, smoke) but nothing is sampled from it.

    venv_vllm/bin/python rl/grpo_distill.py --teacher-set <jsonl> --process-lambda 0 --process-sham off \
        <every rl/grpo_onpolicy.py argument>                         (experiments/harness_v3/run_v3distill.sh launches it)

The teacher set is JSONL, one line per instance: {"instance": "mct-0000", "messages": [{"role": ..., "content": ...}, ...]}.
LOGS (in --out): harness_distill.json -- this file's sha256 and the teacher set's path, sha256 and size; resuming a run dir
with a different teacher set, or one trained without this file, is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "harness"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_set(path: Path) -> dict[str, list[dict]]:
    """{instance name: training messages}; every episode must be system + task + at least one assistant turn."""
    out: dict[str, list[dict]] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r["messages"]
        roles = [m.get("role") for m in msgs]
        if len(msgs) < 3 or roles[0] != "system" or roles[1] != "user" or "assistant" not in roles[2:]:
            raise SystemExit(f"rl/grpo_distill.py: {path}:{n} ({r.get('instance')}) is not system + task + turns: {roles[:4]}")
        if r["instance"] in out:
            raise SystemExit(f"rl/grpo_distill.py: {path}:{n} repeats instance {r['instance']}")
        out[r["instance"]] = msgs
    if not out:
        raise SystemExit(f"rl/grpo_distill.py: empty teacher set {path}")
    return out


def teacher_rollouts(teacher: dict[str, list[dict]]):
    """A drop-in for grpo_onpolicy.rollouts: the teacher's one episode per chosen instance, scored as full credit."""
    def rollouts(insts, spec, k, temperature, timeout, n_steps, workers, c, require_code=False, demo_for=None):
        groups = {}
        for inst in insts:
            msgs = teacher.get(inst.name)
            if msgs is None:
                raise SystemExit(f"rl/grpo_distill.py: no teacher episode for {inst.name} -- the pool must hold only "
                                 "instances of the teacher set")
            cells = sum(1 for m in msgs[2:] if m.get("role") == "assistant")
            groups[inst.name] = [{"reward": 1.0, "passed": True, "messages": msgs, "n_prefix": 0,
                                  "attribution": "teacher", "n_cells": max(1, cells), "answered": True,
                                  "answer": None, "seconds": 0.0}]
        return groups
    return rollouts


def ones(rewards: list[float]) -> list[float]:
    """a = 1 for every teacher episode: maximum likelihood on the teacher's tokens."""
    return [1.0 for _ in rewards]


def adopt(grpo_mod, set_path: Path, teacher: dict[str, list[dict]], out: Path | None) -> None:
    if out is not None:
        f = out / "harness_distill.json"
        rec = {"objective": "distillation (a = 1 on teacher episodes; no policy samples)",
               "trainer_distill_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
               "teacher_set": str(set_path), "teacher_set_sha256": hashlib.sha256(set_path.read_bytes()).hexdigest()[:16],
               "teacher_episodes": len(teacher)}
        if f.exists():
            old = json.loads(f.read_text())
            if old.get("teacher_set_sha256") != rec["teacher_set_sha256"]:
                raise SystemExit(f"rl/grpo_distill.py: {out} was trained on teacher set {old.get('teacher_set')} "
                                 f"({old.get('teacher_set_sha256')}) -- refusing to resume it on {set_path}")
        else:
            if (out / "history.jsonl").exists():
                raise SystemExit(f"rl/grpo_distill.py: {out} has training history but no harness_distill.json -- "
                                 "refusing to resume another objective as distillation")
            f.write_text(json.dumps(rec, indent=1))
    grpo_mod.rollouts = teacher_rollouts(teacher)
    grpo_mod.advantages = ones


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    from rl import grpo_v3 as V3
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--teacher-set", required=True)
    da, rest = ap.parse_known_args(sys.argv[1:])
    v3args, rest = V3.parse_v3_args(rest)
    if v3args.process_lambda != 0.0 or v3args.process_sham != "off":
        raise SystemExit("rl/grpo_distill.py: distillation uses no reward -- --process-lambda 0 --process-sham off")
    sys.argv = [sys.argv[0]] + rest
    V3.install(grpo_onpolicy, dispersion, v3args, rest)
    po = argparse.ArgumentParser(add_help=False)
    po.add_argument("--out")
    g, _ = po.parse_known_args(rest)
    set_path = Path(da.teacher_set)
    teacher = load_set(set_path)
    adopt(grpo_onpolicy, set_path, teacher, Path(g.out) if g.out else None)
    print(f"grpo_distill: DISTILLATION on {len(teacher)} teacher episodes from {set_path} (a = 1, no policy samples)",
          flush=True)
    raise SystemExit(grpo_onpolicy.main())
