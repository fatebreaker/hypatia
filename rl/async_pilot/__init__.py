"""Asynchronous-RL pilot (docs/ASYNC_PILOT.md): R1 ported to AReaL v2.0.0.

Nothing here edits the files the synchronous runs use; the harness agent, the sandbox, the reward (rl/dispersion.py
one_rollout -> rl/reward.py StepReward) and R1's rollout wrapper (rl/grpo_onpolicy.py rollouts: require-code, per-instance
step scale) are imported and called unchanged.
"""
