"""rl/dispersion.py with K sibling datasets in every episode -- H5's evaluation (docs/DIRECTIONS.md A3; RL_PLAN.md s.11).

    venv/bin/python rl/dispersion_sib.py --siblings 3 <every rl/dispersion.py argument>

A NEW entry point (rl/dispersion.py is imported by every running trainer and stays untouched): strips --siblings and
replaces the module-global `one_rollout` that dispersion.main() calls with rl.siblings.wrap(one_rollout, K, key="eval"),
so the base and every checkpoint see the SAME siblings for each target (drawn from the target's own split, test). Every
rollout in the output JSON carries `siblings`, `sib_ref` and `sib_n` (the usage metric).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl import siblings  # noqa: E402

if __name__ == "__main__":
    sys.argv, K = siblings.strip_flag(sys.argv)
    if K <= 0:
        raise SystemExit("rl/dispersion_sib.py needs --siblings K (K >= 1); use rl/dispersion.py for plain episodes")
    from rl import dispersion
    dispersion.one_rollout = siblings.wrap(dispersion.one_rollout, K, key="eval")
    raise SystemExit(dispersion.main())
