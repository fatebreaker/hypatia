"""Read an async-pilot run: throughput (criterion 2), staleness (criterion 3), and the learning curve by policy version.

    venv/bin/python rl/async_pilot/summarize.py RUN_DIR [--r1 R1_RUN_DIR ...]

Throughput. The pilot's "one update" is R1's "one step": 8 groups x 6 episodes (48 rollouts) -> one optimizer step.
Updates per hour are measured between the first and the last update (start-up -- model load, vLLM start, the first
batch -- is reported separately). R1's reference is each seed's steps logged with the straggler cap on
(history.jsonl rows with wall_budget == 600): 3600 / mean(t_step). Criterion 2 passes if pilot >= 3x R1 (the pilot
holds two GPUs, an R1 seed one).

Staleness. history.jsonl logs, per update, how many versions old each trained episode's first (head) and last (tail)
sampled token is (0 = sampled by the policy being updated). Criterion 3 fails if more than half of the trained
episodes are > 2 versions stale at the head.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

CK = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/ckpt")


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p)] if p.exists() else []


def r1_rate(run: Path) -> tuple[float | None, int]:
    rows = [r for r in load(run / "history.jsonl") if r.get("wall_budget") == 600]
    if not rows:
        return None, 0
    return 3600.0 / st.fmean(r["t_step"] for r in rows), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--r1", nargs="*", default=[str(CK / f"rl-czero-rlmix6-base-echo-rc-lr1e-5-s{s}") for s in range(1, 6)])
    a = ap.parse_args()
    run = Path(a.run)
    hist, roll = load(run / "history.jsonl"), load(run / "rollouts.jsonl")
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
    upd = [h for h in hist if "t_end" in h]
    print(f"run {run.name}: {len(upd)} updates ({sum(bool(h.get('updated')) for h in upd)} with an optimizer step), "
          f"{len(roll)} episodes logged")
    rate = None
    if len(upd) >= 2:
        span = upd[-1]["t_end"] - upd[0]["t_end"]
        rate = 3600.0 * (len(upd) - 1) / span if span > 0 else None
        print(f"  throughput: {rate:.2f} updates/hour over updates {upd[0]['step']}..{upd[-1]['step']} "
              f"({span / 3600:.2f} h)")
    if roll and upd:
        print(f"  start-up to first update: {(upd[0]['t_end'] - min(r['t'] for r in roll if 't' in r)) / 60:.1f} min "
              f"after the first finished group (model load and vLLM start not included)")
    r1 = [(Path(p).name, *r1_rate(Path(p))) for p in a.r1]
    for name, rr, n in r1:
        print(f"  R1 {name}: " + (f"{rr:.2f} updates/hour over {n} capped steps" if rr else "no capped steps logged yet"))
    rates = [rr for _, rr, _ in r1 if rr]
    if rate and rates:
        ratio = rate / st.fmean(rates)
        print(f"  CRITERION 2 (speed): pilot / mean capped R1 = {ratio:.2f}x (needs >= 3.0x) -> "
              f"{'PASS' if ratio >= 3.0 else 'FAIL'}")
    heads, tails = Counter(), Counter()
    for h in upd:
        heads.update({int(k): v for k, v in (h.get("staleness_head") or {}).items()})
        tails.update({int(k): v for k, v in (h.get("staleness_tail") or {}).items()})
    if heads:
        n = sum(heads.values())
        gt2 = sum(v for k, v in heads.items() if k > 2) / n
        print(f"  staleness of trained episodes (versions behind the updated policy), head token: "
              f"{dict(sorted(heads.items()))}; tail token: {dict(sorted(tails.items()))}")
        print(f"  CRITERION 3 (staleness): {gt2:.1%} of trained episodes > 2 versions stale at the head -> "
              f"{'VALID' if gt2 <= 0.5 else 'INVALID'} (max_head_offpolicyness={meta.get('max_head_offpolicyness')})")
    by_v = defaultdict(list)
    for r in roll:
        if r.get("head_version") is not None or r.get("version_at_submit") is not None:
            v = r["head_version"] if r.get("head_version") is not None else r["version_at_submit"]
            by_v[int(v) // 10 * 10].append(r)
    if by_v:
        print("  learning curve by sampling policy version (10-version bins): mean reward / pass rate / cells / "
              "overflow / desync / n")
        for v in sorted(by_v):
            rs = by_v[v]
            print(f"    v{v:>3}-{v + 9:<3} {st.fmean(x['reward'] for x in rs):.3f}  "
                  f"{st.fmean(bool(x.get('passed')) for x in rs):.3f}  "
                  f"{st.fmean((x.get('cells') or 0) for x in rs):.2f}  "
                  f"{st.fmean(bool(x.get('overflow')) for x in rs):.3f}  {sum(bool(x.get('desync')) for x in rs)}  {len(rs)}")
    fam = defaultdict(list)
    for r in roll:
        if "inst" in r and "reward" in r:
            fam[r["inst"].rsplit("_", 1)[0]].append(r)
    if fam:
        print("  per family: " + ", ".join(f"{f} {st.fmean(bool(x.get('passed')) for x in rs):.2f} (n={len(rs)})"
                                           for f, rs in sorted(fam.items())))
    fails = [r for r in roll if "instrument_failure" in r or "workflow_error" in r]
    if fails:
        print(f"  {len(fails)} groups lost to instrument failures / workflow errors (see rollouts.jsonl)")
    ck = sorted(p.name for p in run.glob("step*") if (p / "adapter_model.safetensors").exists())
    print(f"  checkpoints: {ck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
