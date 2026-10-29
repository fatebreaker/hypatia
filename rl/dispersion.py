"""Is there within-group reward variance? The GRPO go/no-go.

WHY THIS IS THE DECISIVE MEASUREMENT
------------------------------------
GRPO computes advantage *within* a group of rollouts on the same prompt. If
every rollout in a group scores the same, the advantage is identically zero and
there is no gradient — regardless of how well-designed the reward is. The
project has already hit both degenerate regimes:

* **base Qwen3-8B**: ran 17 cells then submitted {0.0, 0.0} → reward 0.012, the
  proximity floor. *All rollouts fail identically.*
* **after the degenerate SFT**: recited a memorized script. *All rollouts would
  pass identically.*

So the question is not "what is the mean reward" but **"do rollouts on the same
instance differ from each other"**. A policy at mean 0.5 with every group
spanning 0.2–0.9 is trainable; a policy at mean 0.9 where every group is
exactly 0.9 is not.

WHAT IT REPORTS
---------------
Per instance, K rollouts scored by `rl/reward.py`, then:

  * **within-group spread** (max−min, and std) — the quantity GRPO consumes.
    Groups with spread 0 contribute NOTHING to the gradient.
  * **fraction of dead groups** — the headline. If most groups are flat, RL is
    blocked whatever the mean looks like.
  * reward histogram, and the step at which each rollout fails.

SAMPLING TEMPERATURE MATTERS
----------------------------
Rollouts must be sampled the way RL would sample them. The harness agent
defaults to temperature 0.2, which is near-greedy and will understate
dispersion — measuring at 0.2 and concluding "no variance" would be an artefact
of the decoding, not a property of the policy. Default here is 1.0.

Run:
  venv/bin/python rl/dispersion.py --agent llm:cismvmr-lora --n 8 --k 6
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from agents import make_agent  # noqa: E402
from paths import data_root  # noqa: E402
from rl.reward import StepReward  # noqa: E402
from sandbox import Sandbox  # noqa: E402


def instance_tol(cfg: dict, fallback: float) -> float:
    """Grading tolerance from the instance, not a flag.

    It differs by family (cis-MVMR 0.025, scRNA 0.05); a single default would
    grade one of them on the other's scale.
    """
    kc = ((cfg.get("grader") or {}).get("config") or {})
    keys = kc.get("keys") or {}
    tols = [v.get("absolute_tolerance") for v in keys.values()
            if isinstance(v, dict) and v.get("absolute_tolerance")]
    if not tols and kc.get("absolute_tolerance"):
        tols = [kc["absolute_tolerance"]]
    return float(min(tols)) if tols else fallback


def _with_demo(msgs: list[dict], demo: list | None) -> list[dict]:
    """[system, task, turns] -> [system, *demo, ICL_BRIDGE + task, turns], exactly what LLMAgent.solve sent."""
    if not demo:
        return msgs
    from agents import ICL_BRIDGE
    return [msgs[0], *demo, {"role": "user", "content": ICL_BRIDGE + msgs[1]["content"]}, *msgs[2:]]


def conversation(cfg: dict, sb, transcript: list[dict]) -> list[dict]:
    """system + task + the agent's turns, in the harness's own rendering.

    The harness stores tool output under role "tool"; the chat template and the
    SFT corpus both use "user" for it, so it is mapped here rather than at
    training time.
    """
    # The system message the agent actually SENT: LLMAgent.solve reads GBP_SYSTEM (unset -> "default" -> SYSTEM, so every
    # existing run is unchanged). Until 2026-09-24 this was hard-coded to SYSTEM, so an RL run under GBP_SYSTEM=science
    # (docs/DIRECTIONS.md A1/A2) would have trained on episodes scored in a context the policy was not sampled in.
    from agents import SYSTEMS, build_prompt
    msgs = [{"role": "system", "content": SYSTEMS[os.environ.get("GBP_SYSTEM", "default")]},
            {"role": "user", "content": build_prompt(cfg["task"], sb.listing())}]
    for m in transcript:
        role = m.get("role")
        if role == "assistant":
            msgs.append({"role": "assistant", "content": m.get("content") or ""})
        elif role in ("tool", "system"):
            # "sent": the text the model actually read, where the transcript keeps a marker instead (the forced-answer
            # turn). Training on the marker would condition the reply on a prompt the policy never saw.
            msgs.append({"role": "user", "content": m.get("sent") or m.get("content") or ""})
    return msgs


# 10-17 (H-GEN27): graders that read files the agent left in its working directory get them attached as submission["_files"]
# before the sandbox is cleaned up, exactly as harness/runner.py does at evaluation (MedAgentBench's recorded FHIR POSTs; the
# BioDSBench-style solution.py). Every other grader type is untouched.
_SUBMISSION_FILES = {"medagentbench_refsol": ("fhir_posts.jsonl",), "bdsgen_tests": ("solution.py",),
                     "biocoder_stdout": ("solution.py",),
                     "biodsbench_tests": ("solution.py",)}


def _attach_submission_files(cfg: dict, submission, sb) -> None:
    want = _SUBMISSION_FILES.get((cfg.get("grader") or {}).get("type"))
    if not want or not isinstance(submission, dict):
        return
    files = {}
    for name in want:
        p = Path(sb.workdir) / name
        if p.is_file():
            files[name] = p.read_text(errors="replace")[:200_000]
    submission["_files"] = files


def one_rollout(cfg: dict, data_dir: Path, spec: str, temperature: float,
                tol: float, timeout: int, n_steps: int = 5,
                partial_ceiling: float = 1.0, demo: list | None = None,
                demo_in_messages: bool = False) -> dict:
    """One episode, scored. `partial_ceiling` is the reward's c: 1.0 is the
    StepReward default (and what pass rates never depend on); RL trains at the
    paper's c = 0.5, as rl/verl_sandbox_loop.py does.
    `demo`: a worked example (message list, no system message) placed in context, as GBP_ICL_DEMO does -- but per
    call, so threads can use different demos (RL_PLAN.md R2's transfer checks). None = unchanged behaviour.
    `demo_in_messages`: keep the demo in the stored episode (system, *demo, bridge+task, turns) -- the context the
    policy was sampled in -- and report `n_prefix` = messages before the task, which RL must mask out of the gradient
    (RL_PLAN.md R3: the library's turns were not written by the policy)."""
    agent = make_agent(spec)
    if hasattr(agent, "temperature"):
        agent.temperature = temperature
    if demo is not None and hasattr(agent, "icl_demo"):
        agent.icl_demo = demo
    sb = Sandbox(data_dir=data_dir, timeout=timeout)
    try:
        submission, transcript = agent.solve(cfg, sb)
        _attach_submission_files(cfg, submission, sb)
        res = StepReward(tolerance=instance_tol(cfg, tol), n_steps=n_steps,
                         partial_ceiling=partial_ceiling).score(cfg, submission)
        return {"reward": res.reward, "passed": res.passed,
                # Full conversation, needed by rl/grpo_train.py to recompute
                # logprobs over the assistant spans. `transcript` records the
                # agent's turns only, so the system prompt and task have to be
                # prepended -- training on a prompt the policy never saw would
                # put the ratio on the wrong distribution.
                "messages": _with_demo(conversation(cfg, sb, transcript), demo if demo_in_messages else None),
                "fails_at_step": res.fails_at_step,
                "n_prefix": (1 + len(demo)) if (demo and demo_in_messages) else 0,
                "attribution": res.attribution, "n_cells": sb.attempts,
                "answered": submission is not None,
                # The ANSWER itself, not just its score. Without it a
                # self-consistency baseline is impossible to compute after the
                # fact, and self-consistency is the honest comparator for a
                # pass@1 claim: pass@k needs the grader to pick the winner,
                # which no deployment has.
                "answer": (submission or {}).get("answer")
                          if isinstance(submission, dict) else None}
    except Exception as exc:                                        # noqa: BLE001
        return {"reward": 0.0, "passed": False, "fails_at_step": None,
                "answer": None,
                "attribution": f"error:{type(exc).__name__}",
                # The message, not just the class: "RuntimeError" alone cannot
                # tell a context overflow (a policy property) from a dead
                # server (an instrument failure).
                "error_msg": str(exc)[:300],
                "n_cells": sb.attempts, "answered": False}
    finally:
        sb.cleanup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="llm:Qwen3-8B:nothink")
    # The instance family is a PARAMETER, not a constant. It was hardcoded to
    # cis_mvmr_1k, which meant a cross-family transfer run silently scored the
    # training family instead of the held-out one -- and reported a plausible
    # number while doing it. Transfer is the measurement that makes an RL
    # result interpretable, so the wrong family here invalidates exactly the
    # thing it exists to establish.
    ap.add_argument("--family", default="cis_mvmr_1k",
                    help="directory under generated/ (cis_mvmr_1k | scrna_1k)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8, help="instances (groups)")
    ap.add_argument("--k", type=int, default=6, help="rollouts per instance")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.025)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = data_root() / "generated" / args.family / args.split
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} missing. Mint it first, e.g.\n"
            f"  generators/emit.py --family scrna_ambient --n 100 "
            f"--out $DEEP_DATA_ROOT/generated/{args.family}/{args.split}")
    dirs = sorted(p for p in root.iterdir() if (p / "eval_config.json").exists())
    dirs = dirs[: args.n]
    # Read step depth from the data: cis-MVMR has 5 stages, scRNA has 3, and a
    # fixed n_steps would mis-scale every attributed reward on the other family.
    import json as _json
    probe = _json.loads((dirs[0] / "eval_config.json").read_text())
    steps = [v.get("fails_at_step") for v in
             (probe.get("_provenance", {}).get("ablations") or {}).values()]
    n_steps = max([s for s in steps if s], default=5)
    print(f"agent={args.agent}  family={args.family}/{args.split}  "
          f"groups={len(dirs)}  rollouts/group={args.k}  "
          f"temperature={args.temperature}  n_steps={n_steps}")

    import concurrent.futures as cf
    jobs = [(d, r) for d in dirs for r in range(args.k)]

    def run(job):
        d, _ = job
        cfg = json.loads((d / "eval_config.json").read_text())
        return d.name, one_rollout(cfg, d / "data_files", args.agent,
                                   args.temperature, args.tol, args.timeout,
                                   n_steps)

    groups: dict[str, list[dict]] = {d.name: [] for d in dirs}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, res) in enumerate(ex.map(run, jobs), 1):
            groups[name].append(res)
            if i % 10 == 0:
                print(f"  ... {i}/{len(jobs)} rollouts", flush=True)

    print("\nper-group rewards")
    print("  instance                    rewards                     spread  std")
    dead = 0
    spreads = []
    for name, rs in groups.items():
        vals = sorted(r["reward"] for r in rs)
        spread = max(vals) - min(vals)
        spreads.append(spread)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        dead += spread < 1e-9
        flag = "  <-- DEAD" if spread < 1e-9 else ""
        shown = " ".join(f"{v:.3f}" for v in vals)
        print(f"  {name:<26} {shown:<26} {spread:.3f}  {sd:.3f}{flag}")

    allr = [r["reward"] for rs in groups.values() for r in rs]
    print("\n" + "=" * 66)
    print(f"mean reward           : {statistics.fmean(allr):.3f}")
    print(f"distinct reward values: {len(set(round(v, 4) for v in allr))}")
    print(f"mean within-group spread : {statistics.fmean(spreads):.3f}")
    print(f"DEAD groups (spread 0)   : {dead}/{len(groups)} "
          f"({dead / len(groups):.0%})")
    print("=" * 66)

    # The verdict this script exists to deliver.
    frac_dead = dead / len(groups)
    if frac_dead >= 0.8:
        print("VERDICT: BLOCKED -- almost every group is flat, so GRPO advantage")
        print("         is ~0 and there is no gradient to follow. Fix the policy")
        print("         or the task distribution before spending GPU on RL.")
    elif frac_dead >= 0.4:
        print("VERDICT: MARGINAL -- a large share of groups contribute nothing.")
        print("         RL could train on the remainder but sample efficiency")
        print("         will be poor; consider filtering instances by spread.")
    else:
        print("VERDICT: VIABLE -- most groups carry within-group variance, which")
        print("         is what GRPO needs. Proceed to RL.")

    print("\nfailure attribution across all rollouts")
    for k, v in Counter(r["attribution"] for rs in groups.values()
                        for r in rs).most_common():
        print(f"  {v:>4}  {k}")
    cells = [r["n_cells"] for rs in groups.values() for r in rs]
    print(f"\ncells executed: mean {statistics.fmean(cells):.1f}, "
          f"zero-cell rollouts {sum(1 for c in cells if c == 0)}/{len(cells)}")

    out = Path(args.out) if args.out else data_root() / "rl" / "dispersion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"agent": args.agent, "family": args.family, "split": args.split,
         "temperature": args.temperature, "n_steps": n_steps,
         "k": args.k, "groups": groups, "dead_fraction": frac_dead,
         **({"icl_demo": os.environ["GBP_ICL_DEMO"]} if os.environ.get("GBP_ICL_DEMO") else {})}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
