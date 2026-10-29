"""GROUNDED-RL TRAINER (docs/RL_PLAN.md H-GROUNDED-RL): rl/grpo_v3.py UNCHANGED (hash-pinned; installed exactly as it
installs itself) with ONE change to the reward: an answer the H-AUDIT rule finds NOT GROUNDED -- not produced by the
agent's own executed code -- earns 0, whatever the grader said. An answer the rule does not apply to keeps its reward.

THE RULE (the one H-AUDIT registered and measured, 2026-10-01 19:16 UTC; no new matcher):
  numeric answers  harness/process_signals' registered 'grounded' rule -- every submitted numeric value was printed by
                   an executed cell (rounding- and sign-aware, typed literals excluded) -- through
                   experiments/harness_v3/exec_forcing_ext's view of the problem (numeric keys only). MedCalc: every key
                   whose submitted value is a number; QRData: exec_forcing_ext.plan(kind="qrdata"); the synthetic
                   families: grpo_v3's own 'grounded' signal, already computed for the process reward.
  SQL answers      (EHRSQL) the submitted query, lower-cased, whitespace collapsed, trailing ';' dropped, occurs in an
                   assistant turn whose tool output carries no traceback.
  n/a              a date / text answer, an abstention (null query), no numeric key -> reward unchanged.

    venv_vllm/bin/python rl/grpo_rg.py <every rl/grpo_v3.py argument>        (run_v3rg.sh launches it)

LOGS (in --out): grounded.jsonl, one line per rollout: step, inst, passed, r_before, reward, grounded (1/0/null), why;
harness_rg.json records the rule and this file's sha256.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "harness", ROOT / "experiments" / "harness_v3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl import grpo_v3 as V3  # noqa: E402

RULE = "H-AUDIT grounded rule (rl/grpo_rg.py docstring): numeric -> process_signals 'grounded' via exec_forcing_ext view; SQL -> submitted query ran without traceback"
TRACEBACK = "Traceback (most recent call last)"


def norm_sql(s) -> str:
    s = re.sub(r"\s+", " ", str(s).lower()).strip()
    return s[:-1].rstrip() if s.endswith(";") else s


def answer_obj(answer) -> dict:
    obj = answer if isinstance(answer, dict) else {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    return obj


def sql_grounded(messages: list, answer) -> tuple:
    """(1.0 | 0.0 | None, why). messages = the chat as the trainer keeps it: system, task, then assistant turns each
    followed by a user turn carrying the tool output."""
    qs = {k: v for k, v in answer_obj(answer).items()
          if isinstance(v, str) and v.strip() and v.strip().lower() not in ("null", "none")}
    if not qs:
        return None, "abstained / no query"
    ran = []
    for i, m in enumerate(messages):
        if i < 2 or m.get("role") != "assistant":
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else {}
        out = str(nxt.get("content") or "") if nxt.get("role") in ("user", "tool") else ""
        if out and TRACEBACK not in out:
            ran.append(norm_sql(m.get("content") or ""))
    bad = sorted(k for k, q in qs.items() if not any(norm_sql(q) in c for c in ran))
    return (0.0, f"query not run: {bad}") if bad else (1.0, "every submitted query ran in a cell without error")


def numeric_grounded(scorer, messages, cfg, ddir, vals: dict, n_prefix: int = 0) -> tuple:
    import exec_forcing_ext as EX
    sig = scorer.signals(messages, EX._view(cfg, list(vals)), ddir, answer={"answer": dict(vals)}, n_prefix=n_prefix)
    return sig.fired.get("grounded"), sig.detail.get("grounded")


def grounded(scorer, o: dict, cfg: dict, ddir: Path, fired: dict) -> tuple:
    import exec_forcing_ext as EX
    gtype = ((cfg.get("grader") or {}).get("type")) if isinstance(cfg.get("grader"), dict) else None
    msgs, ans, npre = o.get("messages") or [], o.get("answer"), int(o.get("n_prefix") or 0)
    if not msgs:
        return None, "no messages"
    if gtype == "ehrsql2024_official":
        return sql_grounded(msgs, ans)
    if gtype == "medcalc_official":
        num = {k: EX.number(v) for k, v in answer_obj(ans).items()}
        num = {k: v for k, v in num.items() if v is not None}
        if not num:
            return None, "no numeric answer (date / text / missing)"
        if not int(o.get("n_cells") or 0):
            return 0.0, "no executed cell"
        return numeric_grounded(scorer, msgs, cfg, ddir, num, npre)
    if gtype == "qrdata_official":
        pl = EX.plan(cfg, ans, kind="qrdata")
        if not pl["numeric"]:
            return None, "no numeric key"
        for k, cands in pl["numeric"].items():
            if not any(c is not None and numeric_grounded(scorer, msgs, cfg, ddir, {k: c}, npre)[0] == 1.0 for c in cands):
                return 0.0, f"not in any computed output: {k}"
        return 1.0, "every numeric answer value was printed by an executed cell"
    if gtype in ("medagentbench_refsol", "bdsgen_tests", "biocoder_stdout"):   # 10-17 (H-GEN27): grounding is not defined for FHIR actions or for
        return None, "grounding not defined (FHIR actions / a program graded by tests)"   # a program -- the grader's verdict stands
    g = (fired or {}).get("grounded")
    return g, "grpo_v3 grounded signal"


class GroundedReward(V3.ProcessReward):
    """grpo_v3's ProcessReward, re-classed after V3.install (as rl/grpo_ds.py does): the composed reward of every
    rollout the rule finds NOT GROUNDED is set to 0."""

    def score_group(self, inst: Path, outs: list, step: int) -> list:
        recs = super().score_group(inst, outs, step)
        cfg = json.loads((inst / "eval_config.json").read_text())
        lines = []
        for o, r in zip(outs, recs):
            try:
                g, why = grounded(self.scorer, o, cfg, inst / "data_files", r.get("fired"))
            except Exception as e:  # noqa: BLE001 -- a rule failure must not kill the run; it is logged, reward kept
                g, why = None, f"rule error: {type(e).__name__}: {e}"[:200]
            before = r["reward"]
            if g == 0.0:
                r["reward"] = 0.0
                r["bonus"] = round(0.0 - r["r_answer"], 6)
                o["reward"] = 0.0
            lines.append({"step": step, "inst": inst.name, "passed": r["passed"], "r_before": before,
                          "reward": r["reward"], "grounded": g, "why": str(why)[:200]})
        if self.out is not None:
            with self._g_lock:
                with open(self.out / "grounded.jsonl", "a") as fh:
                    for x in lines:
                        fh.write(json.dumps(x) + "\n")
        return recs


def adopt(pr) -> GroundedReward:
    pr.__class__ = GroundedReward
    pr._g_lock = threading.Lock()
    if pr.out is not None:
        rec = {"rule": RULE, "trainer_rg_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
        f = pr.out / "harness_rg.json"
        if not f.exists():
            f.write_text(json.dumps(rec, indent=1))
    return pr


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    v3args, rest = V3.parse_v3_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    adopt(V3.install(grpo_onpolicy, dispersion, v3args, rest))
    print("grpo_rg: GROUNDED reward on (an answer not produced by the agent's own executed code earns 0)", flush=True)
    raise SystemExit(grpo_onpolicy.main())
