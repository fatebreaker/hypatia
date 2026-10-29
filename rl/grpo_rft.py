"""RFT TRAINER -- online rejection-sampling fine-tuning, the 'is RL needed?' baseline (docs/RL_PLAN.md H-RFT-BASELINE,
registered 2026-10-04): rl/grpo_v3.py UNCHANGED (installed exactly as it installs itself, at --process-lambda 0 = the
answer-only reward, --require-code as given) with ONE change to the update: the group-normalised GRPO advantage
(rl/grpo_train.advantages, (r - mean) / std within each group) is replaced by a = 1 for a rollout that earned FULL answer
credit (reward >= 1) and a = 0 otherwise. The update is then maximum likelihood on the policy's own correct rollouts,
sampled on-policy each step from the same instances, with the same group size, step budget, LR, clipping and KL as the
RL arm: no negative gradient and no baseline -- and, unlike GRPO, groups whose rollouts are ALL correct still train.
Its comparison with the GRPO arm isolates the RL objective from 'imitate your own successes'.
The --smoke run keeps GRPO's advantages: it checks the shared plumbing (centred advantages, mask, loss, logprobs), and
its assertion that advantages sum to zero does not hold for RFT by construction.

    venv_vllm/bin/python rl/grpo_rft.py --process-lambda 0 --process-sham off <every rl/grpo_onpolicy.py argument>
                                                                                  (run_v3rft.sh launches it)

LOGS (in --out): harness_rft.json records this file's sha256; a run dir trained without it is refused.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "harness"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl import grpo_v3 as V3  # noqa: E402


def rft_advantages(rewards: list[float]) -> list[float]:
    """1 for a rollout with full answer credit, 0 otherwise (partial credit and no-code rollouts get 0)."""
    return [1.0 if r >= 1.0 - 1e-9 else 0.0 for r in rewards]


def adopt(pr, grpo_mod, smoke: bool):
    if pr.lam != 0.0 or pr.sham != "off":
        raise SystemExit("rl/grpo_rft.py: RFT is defined on the answer-only reward -- "
                         f"--process-lambda 0 --process-sham off (got {pr.lam} / {pr.sham})")
    if pr.out is not None:
        f = pr.out / "harness_rft.json"
        if not f.exists():
            if (pr.out / "history.jsonl").exists():
                raise SystemExit(f"rl/grpo_rft.py: {pr.out} has training history but no harness_rft.json -- it was "
                                 "trained with GRPO advantages; refusing to resume it as an RFT run")
            rec = {"objective": "rft (a = 1[full credit], else 0)",
                   "trainer_rft_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
            f.write_text(json.dumps(rec, indent=1))
    if not smoke:
        grpo_mod.advantages = rft_advantages
    return pr


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    smoke = "--smoke" in rest
    adopt(V3.install(grpo_onpolicy, dispersion, v3args, rest), grpo_onpolicy, smoke)
    print(f"grpo_rft: RFT objective {'OFF for the smoke (GRPO advantages)' if smoke else 'ON (a = 1 for full credit, else 0)'}", flush=True)
    raise SystemExit(grpo_onpolicy.main())
