"""R1's instance stream as a map-style dataset, so the pilot draws the SAME prompts in the SAME order as R1 seed s.

rl/grpo_onpolicy.py: insts = sorted(train dir), rng = random.Random(seed), and step t trains on rng.sample(insts, 8).
Laid out flat (step 1's 8 instances, then step 2's, ...) and read in order (shuffle off, batch 8), AReaL's dataloader
submits exactly that stream. What differs is only which finished groups share an update: an asynchronous trainer
takes the first 8 groups to FINISH, so a slow group from draw t can land in update t+1 (docs/ASYNC_PILOT.md).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from paths import data_root  # noqa: E402


def r1_pool(family: str = "rlmix6") -> list[Path]:
    root = data_root() / "generated" / family / "train"
    return sorted(p for p in root.iterdir() if (p / "eval_config.json").is_file())


def r1_stream(seed: int, steps: int, per_step: int = 8, family: str = "rlmix6") -> list[list[Path]]:
    """The instances R1 seed `seed` trains on at steps 1..steps (grpo_onpolicy.train: rng.sample(insts, per_step))."""
    insts = r1_pool(family)
    rng = random.Random(seed)
    return [rng.sample(insts, per_step) for _ in range(steps)]


class R1InstanceStream:
    """torch-style map dataset of {"inst": path}. `margin_steps` extra draws cover the groups the rollout side starts
    ahead of the trainer (staleness window + in-flight groups), so the stream never wraps around."""

    def __init__(self, seed: int = 1, steps: int = 150, per_step: int = 8, family: str = "rlmix6",
                 margin_steps: int = 50):
        self.draws = r1_stream(seed, steps + margin_steps, per_step, family)
        self.items = [{"inst": str(p)} for draw in self.draws for p in draw]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]
