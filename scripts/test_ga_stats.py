#!/usr/bin/env python3
"""Tests for the general-agent statistics. Runs in a second, needs no server, no GPU, no result files.

Written because ONE bug appeared TWICE on 2026-09-18: bootstrapping over instances when a family has only one
instance returns that instance every time, so the interval collapses to zero width and every difference reads as
significant. It was found and fixed in scripts/gen_compare.py, then reproduced hours later in scripts/ga_report.py
by copying the instance-level bootstrap without its guard. A fix that lives only in one file is not a fix.

    venv/bin/python scripts/test_ga_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_compare import compare as cmp_compare  # noqa: E402
from ga_report import paired as report_paired  # noqa: E402

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        fails.append(msg)


# ---- 1. a single instance must NOT produce a zero-width interval -------------------------------------------------
# One instance, 64 rollouts each: the real-data case (one SPECT file, one theophylline study).
rng = np.random.default_rng(0)
a1 = {"only": list(rng.normal(0.30, 0.25, 64))}
b1 = {"only": list(rng.normal(0.10, 0.25, 64))}

r = cmp_compare(a1, b1)
check(r is not None, "gen_compare returned None for a single instance")
check(r["hi"] - r["lo"] > 1e-6, f"gen_compare: zero-width interval on one instance: [{r['lo']}, {r['hi']}]")
check(r["unit"] == "rollouts", f"gen_compare: wrong resampling unit for one instance: {r['unit']}")

p = report_paired(a1, b1)
check(p is not None, "ga_report.paired returned None for a single instance")
check(p[2] - p[1] > 1e-6, f"ga_report: zero-width interval on one instance: [{p[1]}, {p[2]}]")

# ---- 2. many instances must still resample INSTANCES, not rollouts -----------------------------------------------
# Rollouts within an instance are correlated; treating them as independent narrows every interval by ~sqrt(k).
a2 = {f"i{i}": [0.9] * 8 if i % 2 else [0.1] * 8 for i in range(20)}
b2 = {f"i{i}": [0.5] * 8 for i in range(20)}
r2 = cmp_compare(a2, b2)
check(r2["unit"] == "instances", f"gen_compare: wrong unit with 20 instances: {r2['unit']}")
check(r2["n"] == 20, f"gen_compare: wrong instance count: {r2['n']}")
# the arms differ by 0 on average (half +0.4, half -0.4), so the interval must straddle zero
check(r2["lo"] < 0 < r2["hi"], f"gen_compare: interval should straddle zero, got [{r2['lo']}, {r2['hi']}]")

# ---- 3. a real difference must be detected ----------------------------------------------------------------------
a3 = {f"i{i}": [0.8] * 4 for i in range(30)}
b3 = {f"i{i}": [0.2] * 4 for i in range(30)}
r3 = cmp_compare(a3, b3)
check(r3["sig"], "gen_compare: failed to detect a 0.6 difference across 30 instances")
check(abs(r3["diff"] - 0.6) < 1e-9, f"gen_compare: wrong point estimate {r3['diff']}")

# ---- 4. only the instances BOTH arms ran are compared -------------------------------------------------------------
a4 = {f"i{i}": [0.7] * 4 for i in range(10)}
b4 = {f"i{i}": [0.3] * 4 for i in range(6)}          # the control ran fewer
r4 = cmp_compare(a4, b4)
check(r4["n"] == 6, f"gen_compare: compared {r4['n']} instances, expected the 6 shared")

# ---- 5. no shared instances -> None, never a silent zero -----------------------------------------------------------
check(cmp_compare({"a": [1.0]}, {"b": [0.0]}) is None, "gen_compare: should return None with no shared instances")

if fails:
    print(f"{len(fails)} FAILURE(S):")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("all general-agent statistics tests pass")
