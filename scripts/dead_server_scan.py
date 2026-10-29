#!/usr/bin/env python3
"""Which benchmark runs were scored against a server that was not there?

`harness/runner.py` records a failed model call as `error` + `grade.score = 0.0` and carries on, so a run whose vLLM
server never came up (or died mid-way) produces a complete set of result files that average to 0.000. Nothing about
the directory says the measurement did not happen -- `e2_b5_varb` read DataBench 0.000 that way, 522 of 523 records
carrying `URLError: [Errno 111] Connection refused`, while the SAME arm's QRData scored a healthy 0.403.

A zero from a dead server and a zero from a model that cannot do the task are indistinguishable by score alone, and
only one of them belongs in a paper.

    venv/bin/python scripts/dead_server_scan.py [--root results/pods] [--min-rate 0.1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the signatures of "the model was never reached", as opposed to a rollout that ran and failed
# Only signatures of "the server was never reached". NOT "maximum context length" (the model's own overflow) and
# NOT a rollout whose generated code tried to download something (astropy's IERS tables do this).
DEAD = ("Connection refused", "Max retries exceeded", "Connection reset by peer",
        "Remote end closed connection", "Failed to establish a new connection")


def dispersion_dead(path: Path) -> tuple[int, int]:
    """(errored rollouts, total) for a dispersion.py result file.

    The two result formats fail differently and the first version of this scan only understood one of them.
    `harness/runner.py` writes one file per question with an `error` string; `rl/dispersion.py` writes ONE file for
    the whole arm and records a failed rollout as an `attribution` of "error:<ExceptionName>". A control arm that
    never reached its server therefore looked clean here while every one of its 96 rollouts had failed --
    `general_agent_qwen14/control_hc_dx.json` was 96/96 errors and was being used as the baseline for a quoted
    14B result (2026-09-18).
    """
    try:
        groups = json.loads(path.read_text()).get("groups")
    except Exception:
        return 0, 0
    if not isinstance(groups, dict):
        return 0, 0
    n = dead = 0
    for v in groups.values():
        for r in v:
            n += 1
            # An "error:" attribution covers two unrelated things and only one is an infrastructure failure:
            #   - "<urlopen error [Errno 111] Connection refused>"  -- the harness never reached the server. INVALID.
            #   - "HTTP 400 ... maximum context length is 16384"    -- the model filled its context. REAL behaviour,
            #     and a legitimate zero; 85-95% of some astronomy one-demo rollouts end this way.
            # Flagging the second kind would condemn runs that measured exactly what they were meant to.
            msg = str(r.get("error_msg") or "")
            if any(k in msg for k in DEAD):
                dead += 1
    return dead, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pods")
    ap.add_argument("--min-rate", type=float, default=0.10, help="flag a directory above this error rate")
    a = ap.parse_args()
    root = ROOT / a.root
    # resolve first: results/pods is a SYMLINK and rglob does not follow one
    root = root.resolve()

    bad = []
    # A directory beginning with "." is quarantined evidence of a failure already caught; reporting it every run
    # trains the reader to ignore the tool.
    for d in sorted(p for p in root.rglob("*")
                    if p.is_dir() and not any(seg.startswith(".") for seg in p.relative_to(root).parts)):
        files = list(d.glob("*.json"))
        if len(files) < 20:                      # a benchmark directory, not a stray pair of files
            continue
        n = dead = 0
        for f in files:
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(r, dict) or "grade" not in r:
                continue
            n += 1
            e = str(r.get("error") or "")
            if any(s in e for s in DEAD):
                dead += 1
        if n and dead / n >= a.min_rate:
            bad.append((dead / n, dead, n, d.relative_to(root)))

    # dispersion-format arms: one file per arm, errors recorded as attribution "error:<Exception>"
    for f in sorted(root.rglob("*.json")):
        if any(seg.startswith(".") for seg in f.relative_to(root).parts):
            continue
        dead, n = dispersion_dead(f)
        if n >= 20 and dead / n >= a.min_rate:
            bad.append((dead / n, dead, n, f.relative_to(root)))

    if not bad:
        print(f"no result file or directory under {a.root} has a dead-server rate at or above {a.min_rate:.0%}")
        return 0
    print(f"{len(bad)} directory(ies) scored against a server that was not answering:\n")
    for rate, dead, n, rel in sorted(bad, reverse=True):
        print(f"  {rate:6.1%}  {dead:4d}/{n:<4d}  {rel}")
    print("\nThese are NOT scores. Re-run them, or exclude them -- a zero from a dead server reads exactly like a")
    print("zero from a model that cannot do the task.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
