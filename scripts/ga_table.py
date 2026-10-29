#!/usr/bin/env python3
"""The one place general-agent pass rates are computed. Every ad-hoc version of this got something wrong.

Four separate errors on 2026-09-18, each from a throwaway script, each caught only by recomputing from raw files:

  1. A "Qwen3-8B" pool that excluded only `qwen32`/`qwen14` from the DIRECTORY NAME, so general_agent_mistral_cc and
     general_agent_olmo_cc were counted as 8B. It put the control at 12.6% instead of 17.7% and made two arms look
     significant that are not (v15 p=0.011 -> p=0.714, v14 "+" -> a significant HARM).
  2. DABench quoted as x/514 when each run scores 257 problems -- every magnitude inflated by 8-10 points.
  3. A mechanism claim ("the check becomes informative, 4.2% -> 45.5%") read off one draw; the second draw reversed
     it and pooled it sits at base rate.
  4. `passed` read from a results format that does not have it (dispersion.py has `passed`, runner.py has
     `grade.score`), which silently yields zeros.

The fix for (1), which is the one that recurs: **the model is read from the `agent` field inside the file**, never
from the directory name. dispersion.py writes the full spec (`llm:Qwen3-8B:selfver17:nothink`) into every result,
so the directory is only a label and carries no meaning here.

  scripts/ga_table.py [--family FAM] [--model Qwen3-8B] [--all-models] [--self-test]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDERS = {"nothink", "train", "mtrain", "ltrain", "think"}
ARM_NAMES = {
    "control": "control (no loop)", "selfver": "loop", "selfver0": "plan-only", "selfver4": "v4 ORACLE",
    "selfver7": "v7 caution", "selfver8": "v8 state it", "selfver9": "v9 falsify",
    "selfver10": "v10 falsify+caution", "selfver11": "v11 propose+select", "selfver12": "v12 causal-order",
    "selfver13": "v13 order+select", "selfver14": "v14 timing half", "selfver15": "v15 rule half",
    "selfver16": "v16 PLACEBO(read)", "selfver17": "v17 order+caution", "selfver18": "v18 order+filecheck",
    "selfver19": "v19 select+caution", "selfver20": "v20 PLACEBO(replicate)",
}


def parse_spec(spec: str) -> tuple[str, str]:
    """`llm:Qwen3-8B:selfver17:nothink` -> ("Qwen3-8B", "selfver17"); `llm:Olmo-3-7B:train` -> (..., "control").

    The arm is whatever sits between the model and the render. An empty middle is the control: it is the same
    harness and the same model with no self-verification loop, which is exactly what every comparison here is
    against.
    """
    parts = spec.split(":")
    if not parts or parts[0] != "llm" or len(parts) < 2:
        raise ValueError(f"not an llm spec: {spec!r}")
    model, rest = parts[1], parts[2:]
    if rest and rest[-1] in RENDERS:
        rest = rest[:-1]
    return model, (rest[0] if rest else "control")


# Rollouts that never reached the server. dead_server_scan.py exists for exactly this, but it only guards the
# directories someone remembers to point it at: on 2026-09-20 four files in general_agent_gen_d2 -- 240 rollouts
# each, every one `error:URLError` with 0 cells executed -- sat in this scorer's pool and dragged the surv_tvc
# control from 5.9% to 4.5% and the `general` row from 12.5% to 6.2% before anyone thought to scan that
# directory. The scorer is the single place every number passes through, so the refusal lives here too.
_DEAD = ("Connection refused", "Max retries exceeded", "Connection reset by peer",
         "Remote end closed connection", "Failed to establish a new connection")


def resolved_spec(agent: str) -> tuple[str, str]:
    """(model, the spec as RESOLVED) from a result file's `agent` string.

    A spec NAME is not an instruction. `:general` is scale-routed in harness/agents.py -- v23
    (routed_order + no_overadjust) below 20B, v12 (bare causal_order) at 32B and above -- while the agent string
    says "general" at every size. Reading a series by the alias therefore silently mixes two instructions. That
    produced a wrong `:general` scale row on 2026-09-21, and again on 2026-09-22 when item 398 was queued as a
    third v12 draw for causal_rdd @14B and delivered a v23 arm instead; nothing flagged it but the arm's name in
    a directory listing. Every place that assembles a comparison should resolve the alias, so it lives here.
    """
    parts = agent.split(":")
    model = parts[1] if len(parts) > 1 else "?"
    spec = ":".join(parts[2:-1]) or "(bare)"
    if spec == "general":
        spec = "general=v12" if any(x in model for x in ("32B", "70B", "72B")) else "general=v23"
    return model, spec


def score(path: str, data: dict | None = None) -> tuple[int, int]:
    """(passed, rollouts) from a dispersion.py result. Raises if the file is the other format,
    or if >=10% of its rollouts never reached the server (a dead-server file is not a measurement).

    `data` lets a caller that has ALREADY parsed the file pass the dict in. These results carry full
    transcripts -- 1.7 to 11 MB each -- and draw_compare.Arm used to call this with a path and then parse the
    same file a second time for its attribution counts, so every sweep over the tree paid twice.
    """
    d = data if data is not None else json.loads(open(path).read())
    groups = d.get("groups")
    if groups is None:
        raise ValueError(f"{path}: no `groups` -- this is not a dispersion.py result")
    n = p = dead = 0
    for rollouts in groups.values():
        for r in rollouts:
            if "passed" not in r:
                raise ValueError(f"{path}: rollout has no `passed` field (wrong result format?)")
            n += 1
            p += bool(r["passed"])
            if any(s in str(r.get("error_msg") or "") for s in _DEAD):
                dead += 1
    if n and dead / n >= 0.10:
        raise ValueError(f"{path}: {dead}/{n} rollouts never reached the server -- quarantine it, do not score it")
    return p, n


def collect(model: str | None) -> dict[tuple[str, str], list[int]]:
    """(arm, family) -> [passed, rollouts], pooled over every directory, filtered by the spec inside each file."""
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for path in sorted(set(glob.glob(os.path.join(ROOT, "results/pods/general_agent*/*.json")))):
        if os.path.basename(path).startswith("."):
            continue                                            # quarantined (e.g. .x.json.dead_server)
        try:
            d = json.loads(open(path).read())
            if "groups" not in d or "agent" not in d:
                continue
            m, arm = parse_spec(d["agent"])
            if model is not None and m != model:
                continue
            p, n = score(path)
        except ValueError as e:
            # a dead-server file must be LOUD, not silently absent -- silence is how it gets requeued as "missing"
            if "never reached the server" in str(e):
                print(f"  !! EXCLUDED {e}", file=sys.stderr)
            continue
        except Exception:
            continue
        if n:
            key = (arm, d.get("family") or "?")
            agg[key][0] += p
            agg[key][1] += n
    return agg


def report(agg, model_label: str, only_family: str | None) -> None:
    fams = sorted({f for _, f in agg} if not only_family else {only_family})
    for fam in fams:
        rows = [(a, v) for (a, f), v in agg.items() if f == fam]
        if not rows:
            continue
        ctl = dict(rows).get("control")
        if not ctl:
            continue
        cp, cn = ctl
        print(f"\n{fam}  [{model_label}]   control {cp}/{cn} = {cp / cn:.1%}")
        for arm, (p, n) in sorted(rows, key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
            if arm == "control":
                continue
            pv = fisher_exact([[p, n - p], [cp, cn - cp]])[1]
            mark = "+" if (p / n > cp / cn and pv < 0.05) else ("-HARM" if (p / n < cp / cn and pv < 0.05) else "")
            print(f"    {ARM_NAMES.get(arm, arm):<24}{p:>5}/{n:<6}{p / n:>7.1%}  p={pv:<11.3g}{mark}")


def self_test() -> int:
    """Non-vacuous: each check fails loudly if the bug it guards against comes back."""
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  -- ' + detail}")
        ok = ok and cond

    chk("spec: arm+model", parse_spec("llm:Qwen3-8B:selfver17:nothink") == ("Qwen3-8B", "selfver17"))
    chk("spec: control has no arm", parse_spec("llm:Qwen3-8B:nothink") == ("Qwen3-8B", "control"))
    chk("spec: non-Qwen render", parse_spec("llm:Olmo-3-7B:selfver7:train") == ("Olmo-3-7B", "selfver7"))
    chk("spec: no render at all", parse_spec("llm:Qwen3-8B:selfver9") == ("Qwen3-8B", "selfver9"))

    q = collect("Qwen3-8B")
    o = collect("Olmo-3-7B")
    # THE regression test for error (1): the Olmo runs must not appear in the Qwen pool, and must be findable.
    chk("olmo results exist and are separable", sum(v[1] for v in o.values()) > 0,
        "no Olmo-3-7B results parsed -- the spec filter may be dropping everything")
    qcc = q.get(("control", "causal_confounded"))
    chk("8B causal_confounded control is 17.7%, not the contaminated 12.6%",
        qcc is not None and abs(qcc[0] / qcc[1] - 0.177) < 0.01,
        f"got {qcc[0] / qcc[1]:.3f}" if qcc else "missing")
    v7 = q.get(("selfver7", "causal_confounded"))
    chk("v7 pool excludes mistral/olmo (n=672, not 1152)", v7 is not None and v7[1] == 672,
        f"n={v7[1] if v7 else None}")
    # a model filter that silently matches nothing would make every table empty and every claim vacuous
    chk("nonexistent model yields nothing", sum(v[1] for v in collect("No-Such-Model-9B").values()) == 0)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    model = None if a.all_models else a.model
    report(collect(model), "every model" if model is None else model, a.family)
    return 0


if __name__ == "__main__":
    sys.exit(main())
