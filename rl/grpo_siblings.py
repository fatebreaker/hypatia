"""rl/grpo_onpolicy.py with K sibling datasets in every episode (docs/DIRECTIONS.md A3; H5 in docs/RL_PLAN.md s.11).

    venv_vllm/bin/python rl/grpo_siblings.py --siblings 3 <every rl/grpo_onpolicy.py argument>

A NEW entry point, so the trainer every running condition imports is untouched: it strips --siblings, replaces the
module-global `one_rollout` that grpo_onpolicy.rollouts() calls with rl.siblings.wrap(one_rollout, K) -- siblings drawn
per (target, policy spec), i.e. the same for every rollout of a GRPO group and redrawn each step -- and runs
grpo_onpolicy.main() unchanged. Reward is the target's grader, unchanged. Launch it through
experiments/directions/run_sib.sh, which names the run directory as experiments/rl_foothold/run.sh would, plus -sib<K>.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl import siblings  # noqa: E402

if __name__ == "__main__":
    sys.argv, K = siblings.strip_flag(sys.argv)
    if K <= 0:
        raise SystemExit("rl/grpo_siblings.py needs --siblings K (K >= 1); use rl/grpo_onpolicy.py for plain episodes")
    from rl import grpo_onpolicy
    grpo_onpolicy.one_rollout = siblings.wrap(grpo_onpolicy.one_rollout, K)
    print(f"siblings ON: {K} unlabeled sibling datasets per episode (rl/siblings.py); reward on the target only",
          flush=True)
    raise SystemExit(grpo_onpolicy.main())
