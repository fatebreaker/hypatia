"""ANSWER-SHAM TRAINER (docs/RL_PLAN.md H-ANSWER-SHAM, registered 2026-10-03): rl/grpo_v3.py UNCHANGED (installed
exactly as it installs itself, at --process-lambda 0 = the answer-only reward) with ONE change to the reward: within
each GRPO group, the rewards of the ELIGIBLE rollouts -- a transcript and, under --require-code, at least one executed
cell (grpo_v3's own 'eligible') -- are PERMUTED, by a permutation drawn from (seed, step, instance + '#answer')
(grpo_v3.sham_perm, the PROCESS-SHAM's generator with its own salt). The group's multiset of rewards -- its mean, its
variance, whether it is alive -- is exactly the real arm's at that policy, and a rollout with no executed cell still
earns 0 (the code requirement stands), but WHICH coded rollout earns a reward is independent of whether its answer was
right. It is the placebo for "RL in the code harness, with the code requirement, at matched reward variance" WITHOUT
the correctness signal (RLVR with random or spurious rewards has been reported to lift Qwen bases by eliciting code
reasoning). A gain of the real arm over this one is the correctness signal's.

    venv_vllm/bin/python rl/grpo_ash.py --process-lambda 0 --process-sham off <every rl/grpo_onpolicy.py argument>
                                                                                  (run_v3ash.sh launches it)

LOGS (in --out): answer_sham.jsonl, one line per rollout: step, inst, passed, eligible, r_real (the answer reward it
earned), reward (the one it was given); process.jsonl keeps grpo_v3's fields with r_answer = real and reward = given.
harness_ash.json records this file's sha256; a run dir trained without the sham is refused.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "harness"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl import grpo_v3 as V3  # noqa: E402

SALT = "#answer"


def permute_group(recs: list, seed: int, step: int, inst_name: str) -> list:
    """The rewards each rollout is GIVEN: the eligible rollouts' real rewards permuted among themselves; the others keep
    their own (0 under --require-code). Pure function of (recs, seed, step, instance)."""
    given = [r["reward"] for r in recs]
    elig = [i for i, r in enumerate(recs) if r["eligible"]]
    perm = V3.sham_perm(seed, step, inst_name + SALT, len(elig))
    for dst, k in zip(elig, perm):
        given[dst] = recs[elig[k]]["reward"]
    return given


class AnswerShamReward(V3.ProcessReward):
    """grpo_v3's ProcessReward, re-classed after V3.install (as rl/grpo_rg.py does): permutes the eligible rollouts'
    rewards within the group (module docstring)."""

    def score_group(self, inst: Path, outs: list, step: int) -> list:
        recs = super().score_group(inst, outs, step)
        given = permute_group(recs, self.seed, step, inst.name)
        lines = []
        for o, r, g in zip(outs, recs, given):
            lines.append({"step": step, "inst": inst.name, "passed": r["passed"], "eligible": r["eligible"],
                          "r_real": r["reward"], "reward": g})
            r["reward"] = g
            r["bonus"] = round(g - r["r_answer"], 6)
            o["reward"] = g
        if self.out is not None:
            with self._a_lock:
                with open(self.out / "answer_sham.jsonl", "a") as fh:
                    for x in lines:
                        fh.write(json.dumps(x) + "\n")
        return recs


def adopt(pr) -> AnswerShamReward:
    if pr.lam != 0.0 or pr.sham != "off":
        raise SystemExit("rl/grpo_ash.py: the answer sham is defined on the answer-only reward -- "
                         f"--process-lambda 0 --process-sham off (got {pr.lam} / {pr.sham})")
    pr.__class__ = AnswerShamReward
    pr._a_lock = threading.Lock()
    if pr.out is not None:
        f = pr.out / "harness_ash.json"
        if not f.exists():
            if (pr.out / "history.jsonl").exists():
                raise SystemExit(f"rl/grpo_ash.py: {pr.out} has training history but no harness_ash.json -- it was "
                                 "trained without the answer sham; refusing to resume it as a sham run")
            rec = {"sham": "answer-perm", "salt": SALT,
                   "trainer_ash_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
            f.write_text(json.dumps(rec, indent=1))
    return pr


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    adopt(V3.install(grpo_onpolicy, dispersion, v3args, rest))
    print("grpo_ash: ANSWER SHAM on (the coded rollouts' rewards are permuted within each group)", flush=True)
    raise SystemExit(grpo_onpolicy.main())
