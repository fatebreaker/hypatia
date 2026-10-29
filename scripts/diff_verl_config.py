#!/usr/bin/env python
"""Diff the hydra config two verl runs actually used, from their logs.

Resuming a trajectory means reproducing its configuration, and the launcher's
DEFAULTS are not that configuration -- they are whatever the last person left
in the script. Reproducing a run by reading its command line is how you miss
things: a 75 -> 150 resume was set up by carefully copying LR (3e-6, not the
1e-6 default) and MAX_MODEL_LEN (32768, not 16384) out of the original dump,
and still silently differed in two places that mattered:

  total_epochs        10 -> 1      600 instances / batch 8 = 75 steps per
                                   epoch, so resuming at step 75 with one
                                   epoch left NOTHING to run. The trainer
                                   validated once and exited, Ray reported only
                                   "killing leased worker because its owner
                                   died", and no traceback was written
                                   anywhere. It reads exactly like a crash.
  max_response_length 24576 -> 8192  would have truncated every episode to a
                                   third of its token budget -- the §24
                                   divergence #4 failure, re-introduced.

So: diff the whole config, don't spot-check it.

  venv/bin/python scripts/diff_verl_config.py OLD.log NEW.log
"""
from __future__ import annotations

import re
import sys

SKIP_PREFIX = ("(TaskRunnerV1", "(WorkerDict", "(AgentLoopWorkerTQ")


def config_of(path: str) -> dict[str, str]:
    raw = open(path, errors="ignore").read(900_000)
    t = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    t = re.sub(r"\((?:TaskRunnerV1|WorkerDict|AgentLoopWorkerTQ)[^)]*\)\s*", "", t)
    # ALL values seen for each key name, not the first. verl's dump contains
    # six distinct `seed` fields at different nesting depths; keeping only the
    # first compared an unrelated one that was 42 in both runs, and the differ
    # reported a seed-replication run as differing from its original in nothing
    # but the experiment name. Comparing the multiset catches it.
    out: dict[str, list[str]] = {}
    for k, v in re.findall(r"'([a-z_0-9]+)':\s*([^,\n}]+)", t):
        out.setdefault(k, []).append(v.strip().rstrip(","))
    return {k: " | ".join(sorted(v)) for k, v in out.items()}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = config_of(sys.argv[1]), config_of(sys.argv[2])
    shared = sorted(set(a) & set(b))
    diff = [(k, a[k], b[k]) for k in shared if a[k] != b[k]]
    print(f"{len(shared)} shared keys, {len(diff)} differ "
          f"({sys.argv[1]} -> {sys.argv[2]})")
    for k, x, y in diff:
        print(f"  {k:34s} {x:<26s} -> {y}")
    only = sorted(set(a) ^ set(b))
    if only:
        print(f"\n{len(only)} keys present in only one run: {only[:12]}")
    return 1 if diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
