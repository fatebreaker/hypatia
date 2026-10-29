"""LABEL-FREE RL TRAINER (docs/RL_PLAN.md H-LABEL-FREE, 2026-10-07): rl/grpo_rg.py's trainer (rl/grpo_v3.py unchanged +
the grounded rule) with the ANSWER KEY REMOVED from the reward. A rollout earns 1 when its answer is REPLICATED by the
other rollouts of its group -- it agrees with the group's consensus answer, the eligible answer that the most other
eligible rollouts agree with -- and 0 otherwise. The grader's verdict is computed (the trainer always grades) and LOGGED
for monitoring only; nothing derived from it reaches the reward or the advantages.

ELIGIBLE  the rollout executed at least one cell (--require-code), submitted an answer, and the H-AUDIT grounded rule
          (rl/grpo_rg.grounded, unchanged) does not find it NOT GROUNDED (a rule that does not apply -- a date / text
          answer -- leaves it eligible, as in the grounded reward).
AGREE     every answer key agrees: both numbers -> exact when both are integers, else within 5% relative (the
          MedCalc tolerance; H-AUDIT-REPLICATION's agreement); otherwise normalised strings (lower case, whitespace
          collapsed) are equal. EHRSQL (H-LABEL-FREE-EHR, 2026-10-07 11:1x UTC): an answer is the EXECUTION RESULT of the
          rollout's own query, as the official grader's predict_result processes it (read-only database; the gold result
          in the instance's spec is never read); an abstention ('null') is a value; a query that errors is no answer.
CONSENSUS the eligible answer with the most agreeing eligible others (support). No consensus -- and every rollout of
          the group earns 0 -- when the top support is 0 (no two eligible rollouts agree) or when two answers that do not
          agree with each other share the top support (a tie between camps).
REWARD    1.0 iff eligible and agrees with the consensus answer; else 0.0. No process term, no partial credit.
SHAM      --lf-mode eligible (run name -lfe): 1.0 iff ELIGIBLE, whether or not any other rollout agrees -- the same
          label-free shaping toward grounded, code-produced answers WITHOUT the replication signal (the control arm).
INDEP     --lf-mode indep --lf-pseudo FILE (run name -lfi; H-LABEL-FREE-INDEP, 2026-10-07 10:46 UTC): 1.0 iff ELIGIBLE and
          agreeing with an INDEPENDENT analyst's answer to the same instance (FILE: JSON {instance name: submission}, e.g.
          the untrained Qwen3.6-27B's answers); an instance without one -> all 0. Nothing about that answer's correctness
          is used.

    venv_vllm/bin/python rl/grpo_lf.py [--lf-mode consensus|eligible|indep] [--lf-pseudo FILE] <every rl/grpo_v3.py argument>

LOGS (in --out): lf.jsonl, one line per rollout: step, inst, passed (the grader; monitoring only), eligible, grounded,
answer, support, consensus (the group had one), reward, consensus_passed (the grader's verdict on the consensus
answer's rollout; monitoring only); harness_lf.json records the rule and this file's sha256.
"""
from __future__ import annotations

import argparse
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

RULE = ("H-LABEL-FREE consensus reward (rl/grpo_lf.py docstring): eligible = executed code + answer + not NOT-GROUNDED "
        "(rl/grpo_rg.grounded); reward 1 iff eligible and agreeing (ints exact, else 5% relative; strings normalised) "
        "with the eligible answer of greatest support; no consensus (top support 0, or a tie between disagreeing "
        "answers) -> all 0. The grader is logged, never used.")
TOL = 0.05


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        try:
            x = float(str(v).replace(",", "").strip().rstrip("%"))
        except (TypeError, ValueError):
            return None
    return x if x == x and abs(x) != float("inf") else None


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


def answer_items(answer) -> tuple | None:
    """((key, value), ...) of a submission, sorted by key; None when nothing usable was submitted."""
    obj = answer
    if isinstance(obj, dict) and isinstance(obj.get("answer"), dict):
        obj = obj["answer"]
    if not isinstance(obj, dict):
        return None
    items = tuple(sorted((str(k), v) for k, v in obj.items()
                         if v is not None and _norm(v) not in ("", "null", "none")))
    return items or None


def _ehr_grader():
    from rl.reward import _bench_grader
    return _bench_grader("ehrsql2024_official")


def sql_items(answer, cfg: dict) -> tuple | None:
    """((key, processed execution result), ...) of an EHRSQL submission, by the official grader's predict_result (which
    reads only spec['postprocess']); None when a key is missing or a query errors."""
    obj = answer
    if isinstance(obj, dict) and isinstance(obj.get("answer"), dict):
        obj = obj["answer"]
    if not isinstance(obj, dict):
        return None
    keys = ((cfg.get("grader") or {}).get("config") or {}).get("keys") or {}
    G, out = _ehr_grader(), []
    for k, spec in sorted(keys.items()):
        if k not in obj:
            return None
        res = G.predict_result(obj.get(k), {"postprocess": spec.get("postprocess", "2024-03")})
        if str(res).startswith("error_"):
            return None
        out.append((str(k), res))
    return tuple(out) or None


def agree_value(a, b, tol: float = TOL) -> bool:
    x, y = _num(a), _num(b)
    if x is not None and y is not None:
        if float(x).is_integer() and float(y).is_integer():
            return x == y
        return abs(x - y) <= tol * max(abs(y), abs(x), 1e-9)
    return _norm(a) == _norm(b)


def agree(a: tuple, b: tuple, tol: float = TOL) -> bool:
    if a is None or b is None or [k for k, _ in a] != [k for k, _ in b]:
        return False
    return all(agree_value(va, vb, tol) for (_, va), (_, vb) in zip(a, b))


def consensus(answers: list, eligible: list, tol: float = TOL) -> tuple:
    """(rewards, supports, index of the consensus answer or None). Pure; unit-tested in experiments/harness_v3/tests/test_lf.py."""
    n = len(answers)
    el = [bool(eligible[i]) and answers[i] is not None for i in range(n)]
    sup = [0] * n
    for i in range(n):
        if el[i]:
            sup[i] = sum(1 for j in range(n) if j != i and el[j] and agree(answers[i], answers[j], tol))
    top = max((sup[i] for i in range(n) if el[i]), default=0)
    if top == 0:
        return [0.0] * n, sup, None
    best = [i for i in range(n) if el[i] and sup[i] == top]
    m = best[0]
    if any(not agree(answers[m], answers[j], tol) for j in best[1:]):
        return [0.0] * n, sup, None
    return [1.0 if el[i] and agree(answers[i], answers[m], tol) else 0.0 for i in range(n)], sup, m


def _install_class():
    from rl import grpo_rg as RG
    from rl import grpo_v3 as V3

    class LabelFreeReward(V3.ProcessReward):
        """grpo_v3's ProcessReward (lambda 0: r_answer = the grader's reward, kept only in the logs), re-classed after
        V3.install; every rollout's reward is replaced by the consensus reward."""

        def score_group(self, inst: Path, outs: list, step: int) -> list:
            recs = super().score_group(inst, outs, step)
            cfg = json.loads((inst / "eval_config.json").read_text())
            answers, elig, gs = [], [], []
            is_sql = ((cfg.get("grader") or {}).get("type")) == "ehrsql2024_official"
            for o, r in zip(outs, recs):
                if not o.get("messages"):
                    a = None
                elif is_sql:
                    try:
                        a = sql_items(o.get("answer"), cfg)
                    except Exception:  # noqa: BLE001 -- an execution failure is no answer
                        a = None
                else:
                    a = answer_items(o.get("answer"))
                g, why = None, "not checked"
                if a is not None and int(o.get("n_cells") or 0) > 0:
                    try:
                        g, why = RG.grounded(self.scorer, o, cfg, inst / "data_files", r.get("fired"))
                    except Exception as e:  # noqa: BLE001 -- a rule failure leaves the rollout ineligible; logged
                        g, why = 0.0, f"rule error: {type(e).__name__}: {e}"[:200]
                answers.append(a)
                elig.append(a is not None and int(o.get("n_cells") or 0) > 0 and g != 0.0)
                gs.append((g, why))
            rewards, sup, m = consensus(answers, elig)
            if self._lf_mode == "eligible":   # the sham: shaping toward grounded answers, no replication signal
                rewards = [1.0 if e else 0.0 for e in elig]
            elif self._lf_mode == "indep":    # agreement with an independent analyst's answer to this instance
                pseudo = self._lf_pseudo.get(inst.name)
                if is_sql:                    # SQL: the independent query's own execution result (H-LABEL-FREE-INDEP-EHR)
                    try:
                        ref = sql_items(pseudo, cfg) if pseudo is not None else None
                    except Exception:  # noqa: BLE001 -- an independent query that cannot run is no pseudo-label
                        ref = None
                else:
                    ref = answer_items(pseudo)
                rewards = [1.0 if e and ref is not None and agree(a, ref) else 0.0 for a, e in zip(answers, elig)]
            cpassed = bool(recs[m]["passed"]) if m is not None else None
            lines = []
            for o, r, a, e, (g, why), s, rw in zip(outs, recs, answers, elig, gs, sup, rewards):
                r["reward"] = rw
                r["bonus"] = round(rw - r["r_answer"], 6)
                o["reward"] = rw
                lines.append({"step": step, "inst": inst.name, "passed": r["passed"], "eligible": e, "grounded": g,
                              "why": str(why)[:120], "answer": json.dumps(a, default=str)[:160] if a else None,
                              "support": s, "consensus": m is not None, "reward": rw, "consensus_passed": cpassed})
            if self.out is not None:
                with self._lf_lock:
                    with open(self.out / "lf.jsonl", "a") as fh:
                        for x in lines:
                            fh.write(json.dumps(x) + "\n")
            return recs

    return LabelFreeReward


MODES = ("consensus", "eligible", "indep")


def adopt(pr, mode: str = "consensus", pseudo: str | None = None):
    if mode not in MODES:
        raise SystemExit(f"rl/grpo_lf.py: --lf-mode must be one of {MODES}")
    if (mode == "indep") != bool(pseudo):
        raise SystemExit("rl/grpo_lf.py: --lf-pseudo FILE goes with --lf-mode indep, and only with it")
    pr.__class__ = _install_class()
    pr._lf_lock = threading.Lock()
    pr._lf_mode = mode
    pr._lf_pseudo = json.loads(Path(pseudo).read_text()) if pseudo else {}
    if pr.lam != 0.0:
        raise SystemExit("rl/grpo_lf.py: the label-free reward has no process term; run with --process-lambda 0")
    if pr.out is not None:
        rec = {"rule": RULE, "mode": mode, "trainer_lf_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
        if pseudo:
            rec.update(pseudo=str(pseudo), pseudo_sha256=hashlib.sha256(Path(pseudo).read_bytes()).hexdigest()[:16],
                       pseudo_n=len(pr._lf_pseudo))
        f = pr.out / "harness_lf.json"
        if f.exists():
            old = json.loads(f.read_text())
            if old.get("mode", "consensus") != mode:
                raise SystemExit(f"rl/grpo_lf.py: {pr.out} was trained with --lf-mode {old.get('mode')} -- refusing {mode}")
        else:
            f.write_text(json.dumps(rec, indent=1))
    return pr


if __name__ == "__main__":
    from rl import dispersion, grpo_onpolicy
    from rl import grpo_v3 as V3
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--lf-mode", choices=MODES, default="consensus")
    ap.add_argument("--lf-pseudo", default=None)
    lfa, rest = ap.parse_known_args(sys.argv[1:])
    v3args, rest = V3.parse_v3_args(rest)
    sys.argv = [sys.argv[0]] + rest
    adopt(V3.install(grpo_onpolicy, dispersion, v3args, rest), lfa.lf_mode, lfa.lf_pseudo)
    print(f"grpo_lf: LABEL-FREE reward on, mode {lfa.lf_mode} (the grader is logged, never used)", flush=True)
    raise SystemExit(grpo_onpolicy.main())
