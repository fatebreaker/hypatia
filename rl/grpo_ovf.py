"""OVERFLOW IN THE GRADIENT -- H-OVF (docs/RL_PLAN.md, registered 2026-10-09): rl/grpo_v3.py -- or, with --grounded,
rl/grpo_rg.py, each installed exactly as it installs itself -- with ONE change, in the rollouts:

  A rollout whose next model call overflows the server's context window (vLLM: "maximum context length") ends as an
  ERROR in rl/dispersion.one_rollout: reward 0 and NO messages. It never enters the gradient; only its reward 0 enters
  its group's mean and std (rl/grpo_onpolicy.train). The notebook that grew until it overflowed is never penalised, and
  in every long 9B run notebooks grow (GR9-MC s1: 1.6 cells per rollout at steps 1-25, 6.3 at 201-225; overflowed
  rollouts about 1 of 48 per step before step 200, 13-18 after). At evaluation (32k context) its step-250 policy uses
  all 25 turns without answering on 173 of the 1,100 MedCalc problems (step 100: 7).

  Here an overflowed rollout KEEPS the conversation it was sampled in -- the messages of the model call that overflowed
  (what LLMAgent._chat was sent, after the agent's own trimming), up to the policy's last reply -- and enters the
  gradient with its reward 0, like any other failed episode. Everything else is unchanged: other errors (sandbox,
  server, network) keep no messages; grading, rewards, group statistics and evaluation are as before.

    venv_vllm/bin/python rl/grpo_ovf.py [--grounded] <every rl/grpo_v3.py argument>   (experiments/harness_v3/run_v3ovf.sh)

LOGS (in --out): harness_ovf.json -- this file's sha256 and the --grounded flag; resuming a run dir trained without this
file, or with the other flag, is refused. ovf_log.jsonl -- per rollouts() call: rollouts, errored, overflowed, kept
(entered the batch with messages), kept in groups whose rewards vary (the ones with a non-zero advantage), and the
assistant turns kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "harness"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OVERFLOW = "maximum context length"   # vLLM's 400 body when prompt + max_tokens exceeds --max-model-len
_TL = threading.local()               # per thread: the messages of the latest model call (one rollout runs per thread)


def instrument(make_agent):
    """make_agent wrapper: each agent records, for its thread, the messages of every model call it makes."""
    def make(spec, *a, **kw):
        agent = make_agent(spec, *a, **kw)
        chat = getattr(agent, "_chat", None)
        if callable(chat):
            def _chat(messages, *ca, **ckw):
                _TL.ctx = [{"role": m.get("role"), "content": m.get("content") or ""} for m in messages]
                return chat(messages, *ca, **ckw)
            object.__setattr__(agent, "_chat", _chat)
        return agent
    make.ovf_instrumented = True
    return make


def overflowed(out: dict) -> bool:
    return str(out.get("attribution") or "").startswith("error") and OVERFLOW in str(out.get("error_msg") or "")


def kept_messages(ctx: list | None) -> list | None:
    """The overflowing call's messages up to the policy's last reply (the trailing tool output, nudge or forcing turn
    is what overflowed and carries no gradient); None when the policy never replied."""
    if not ctx:
        return None
    last = max((i for i, m in enumerate(ctx) if m.get("role") == "assistant"), default=None)
    return [dict(m) for m in ctx[:last + 1]] if last is not None else None


def keep_overflow(one_rollout):
    """one_rollout wrapper: an overflowed rollout keeps the conversation it was sampled in; its reward stays 0."""
    def ovf_rollout(cfg, data_dir, spec, *a, **kw):
        _TL.ctx = None
        out = one_rollout(cfg, data_dir, spec, *a, **kw)
        ctx, _TL.ctx = getattr(_TL, "ctx", None), None
        # A demonstration in context (a v2 library; never in harness v3) would need its turns masked: not kept.
        if overflowed(out) and not out.get("messages") and not kw.get("demo"):
            msgs = kept_messages(ctx)
            if msgs:
                out["messages"] = msgs
                out["n_prefix"] = 0
                out["ovf_kept"] = sum(m["role"] == "assistant" for m in msgs)
        return out
    return ovf_rollout


def logged(rollouts, log_path: Path | None):
    """rollouts wrapper (outermost, so the rewards are final): one ovf_log.jsonl line per call."""
    def ovf_rollouts(insts, *a, **kw):
        groups = rollouts(insts, *a, **kw)
        outs = [o for v in groups.values() for o in v]
        live = [o for v in groups.values() if len({round(float(x.get("reward") or 0), 9) for x in v}) > 1 for o in v]
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rollouts": len(outs),
               "errored": sum(str(o.get("attribution") or "").startswith("error") for o in outs),
               "overflowed": sum(overflowed(o) for o in outs),
               "kept": sum(bool(o.get("ovf_kept")) for o in outs),
               "kept_live": sum(bool(o.get("ovf_kept")) for o in live),
               "kept_turns": sum(int(o.get("ovf_kept") or 0) for o in outs)}
        if log_path is not None:
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"grpo_ovf: {rec['overflowed']}/{rec['rollouts']} rollouts overflowed; {rec['kept']} kept with their "
              f"conversation ({rec['kept_live']} in groups whose rewards vary)", flush=True)
        return groups
    return ovf_rollouts


def guard(out: Path | None, grounded: bool) -> None:
    if out is None:
        return
    f = out / "harness_ovf.json"
    rec = {"objective": "GRPO with overflowed rollouts kept in the gradient (reward 0, the conversation they were "
                        "sampled in)" + (" + the grounded reward" if grounded else ""),
           "trainer_ovf_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16], "grounded": grounded}
    if f.exists():
        old = json.loads(f.read_text())
        if old.get("grounded") != grounded:
            raise SystemExit(f"rl/grpo_ovf.py: {out} was trained with grounded={old.get('grounded')} -- refusing to "
                             f"resume it with grounded={grounded}")
        return
    if (out / "history.jsonl").exists():
        raise SystemExit(f"rl/grpo_ovf.py: {out} has training history but no harness_ovf.json -- refusing to resume "
                         "another objective with overflowed rollouts in the gradient")
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1))


def adopt(grpo_mod, dispersion_mod, out: Path | None, grounded: bool) -> None:
    """After rl/grpo_v3.install (and rl/grpo_rg.adopt): wrap what they installed."""
    guard(out, grounded)
    if not getattr(dispersion_mod.make_agent, "ovf_instrumented", False):
        dispersion_mod.make_agent = instrument(dispersion_mod.make_agent)
    grpo_mod.one_rollout = keep_overflow(grpo_mod.one_rollout)
    grpo_mod.rollouts = logged(grpo_mod.rollouts, (out / "ovf_log.jsonl") if out is not None else None)


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    from rl import grpo_v3 as V3
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--grounded", action="store_true")
    oa, rest = ap.parse_known_args(sys.argv[1:])
    v3args, rest = V3.parse_v3_args(rest)
    sys.argv = [sys.argv[0]] + rest
    pr = V3.install(grpo_onpolicy, dispersion, v3args, rest)          # exactly rl/grpo_v3.py's __main__
    if oa.grounded:
        from rl import grpo_rg as RG
        RG.adopt(pr)                                                   # exactly rl/grpo_rg.py's __main__
    po = argparse.ArgumentParser(add_help=False)
    po.add_argument("--out")
    g, _ = po.parse_known_args(rest)
    adopt(grpo_onpolicy, dispersion, Path(g.out) if g.out else None, oa.grounded)
    print(("grpo_rg: GROUNDED reward on; " if oa.grounded else "") +
          "grpo_ovf: overflowed rollouts enter the gradient with their conversation (reward 0)", flush=True)
    raise SystemExit(grpo_onpolicy.main())
