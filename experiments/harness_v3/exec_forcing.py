"""EF -- THE EXECUTION-FORCING HARNESS, NO TRAINING (docs/RL_PLAN.md 2026-09-28 14:40 UTC, registered before any build or
run; motivated by results/pods/mechanism/MECHANISM.md s.5-6). The untrained Qwen3-8B in the v3 evaluation harness
(experiments/harness_v3/screen_v3.sh: KERNEL + science prompt + notebook echo, NO library) with ONE change: a final answer
is accepted only if its numbers were printed by the agent's own executed code; otherwise the harness replies with one
fixed message and the episode continues under the SAME turn and wall limits.

HOW IT REACHES THE HARNESS (no pinned or shared file is edited). harness/agents.py, harness/v2.py, rl/dispersion.py and
experiments/harness_v2/screen_any.sh are running / pinned files (experiments/acquire/acquire.py PINNED_SHA). This module
is loaded ONLY by experiments/harness_v3/ef_site/sitecustomize.py, which experiments/harness_v3/screen_v3ef.sh puts first on
PYTHONPATH together with GBP_EF=1 (the v3.1 precedent: experiments/harness_v3/v31_site/). Both are needed: GBP_EF=1 without
the hook directory does nothing (nothing else reads it), and the hook directory without GBP_EF=1 installs nothing. When
the flag is on, the hook calls
  install_agents(agents)  right after harness/agents.py's body runs: LLMAgent.solve := the EF loop below. The loop is
                          LLMAgent.solve VERBATIM (every statement, message and limit) with the gate at the three places
                          the original accepts an answer; every added line is marked "# EF". SelfVerifyAgent has its own
                          solve and is untouched (the eval agent is llm:Qwen3-8B:rl = a plain LLMAgent).
  install_v2(v2)          right after harness/v2.py's body runs: HarnessConfig.as_dict gains an "exec_forcing" record
                          (so config.json and every family JSON's "harness_v2" name the EF harness, the feedback text and
                          this file's sha; screen_any.sh refuses to mix a V3EF directory with rows written under another
                          configuration), V2Sandbox.stats() gains the per-rollout EF log (-> rollout["v2"]["ef"]), and the
                          banner says EXEC-FORCING.
Why a copied loop and not a narrower patch: solve() returns on the first parseable answer (a local decision inside the
loop), so no wrapper around _chat / extract_answer can refuse an answer AND reply AND keep counting turns without
changing other behaviour (a patched extract_answer would, e.g., run the code of a refused answer message, which the
harness never does, and answer it with the generic nudge). agents.py is sha-pinned, so the copy cannot drift from it;
experiments/harness_v3/tests/test_ef.py proves the copy with an always-accepting gate reproduces LLMAgent.solve's
transcripts exactly across every branch (code, repeated-error hint, nudges, cut-off, stall abort, trimming, the forcing
tail, the wall budget, the final forced call, an in-context demo).

THE GATE (the registered rule, reused, not reimplemented): harness/process_signals.ProcessScorer().signals(messages, cfg,
data_dir, answer=submission).fired["grounded"] -- the same call results/pods/mechanism/mechanism.py, read_v3.py and
rl/grpo_v3.py make -- on the episode so far rendered exactly as rl/dispersion.conversation renders it (system, task,
then assistant turns and tool feedback as user turns), i.e. every numeric answer value appears (rounding-aware, sign
included) in the STDOUT of an EXECUTED cell and was not typed into that cell as a literal. Executed = the harness's own
rule (process_signals.executed_cells: the first ```python block of a reply with no answer, outside a forced turn).
  1.0   accept, as usual
  0.0   REJECT: the reply is kept, its code (if any) is NOT run (the harness never runs code beside an answer), and the
        harness replies FEEDBACK (one fixed message). The turn is consumed; the stall counter resets (an answer is not a
        stall); the episode continues.
  None  the rule does not apply (the instance has no numeric answer keys): accept (logged "na"; no TEST/DESIGN family
        of this read has such instances -- StepReward._fields is non-empty for all 18).
  error the scorer raised: accept (fail-open, logged "gate_errors": the rollout then behaves as in V3, so an error can
        only shrink the EF effect; read_ef.py prints the count).
Gate points: an answer on an ordinary turn, on a forcing-tail turn (the "STOP ... answer now" turns: the harness's own
answer reserve, unchanged -- no code runs there, so only an answer whose numbers were ALREADY printed passes), and the
final forced call after the loop.
LIMITS: unchanged (GBP_MAX_TURNS, default 25; force_answer_turns 3; max_stalls 5; GBP_WALL_BUDGET, unset = none in the
v3 eval). When the episode ends without an accepted answer (turn limit incl. the final forced call, the wall budget, or
the stall abort) the submission is the LAST answer the model offered that is grounded against ALL cells executed in the
episode (an answer refused before the cell that printed its number was run); if none is, NO answer (submission None).

LOG (per rollout, rollout["v2"]["ef"]): rejected (answers refused), offered (answers the model gave, accepted or not),
na, gate_errors, accepted ("gate" | "na" | "gate_error" | "fallback" | None = no answer), events (one per refusal:
turn, phase turn/forcing/final, the scorer's reason). The FEEDBACK turns are also in the stored messages (user role).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLAG = "GBP_EF"
FEEDBACK = ("Your answer is not backed by printed output: compute it with code, print it, then answer again.")
RULE = "harness/process_signals.py ProcessScorer.signals(...).fired['grounded'] (the registered 'grounded' signal)"
SITE = HERE / "ef_site" / "sitecustomize.py"

_SCORER = None
_SCORER_LOCK = threading.Lock()


def _sha16(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def record() -> dict:
    """What config.json / each family JSON's "harness_v2" records about EF."""
    return {"on": True, "rule": RULE, "feedback": FEEDBACK,
            "sha256": {"experiments/harness_v3/exec_forcing.py": _sha16(Path(__file__).resolve()),
                       "experiments/harness_v3/ef_site/sitecustomize.py": _sha16(SITE)}}


def scorer():
    """One ProcessScorer per process (thread-safe; it caches each instance's data vocabulary)."""
    global _SCORER
    with _SCORER_LOCK:
        if _SCORER is None:
            from process_signals import ProcessScorer
            _SCORER = ProcessScorer()
        return _SCORER


def as_messages(transcript: list) -> list:
    """The episode as rl/dispersion.conversation renders it (placeholders for the system prompt and the task, which
    executed_cells skips): assistant turns as assistant, tool / system entries as user with the text the model read."""
    out = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
    for m in transcript:
        role = m.get("role")
        if role == "assistant":
            out.append({"role": "assistant", "content": m.get("content") or ""})
        elif role in ("tool", "system"):
            out.append({"role": "user", "content": m.get("sent") or m.get("content") or ""})
    return out


def grounded(transcript: list, cfg: dict, data_dir, submission: dict):
    """(fired, reason) of the registered 'grounded' signal for `submission` given the cells executed in `transcript`:
    1.0, 0.0 or None (not applicable). The full submission is passed; _grounded unwraps its "answer" object."""
    sig = scorer().signals(as_messages(transcript), cfg, data_dir, answer=submission)
    return sig.fired.get("grounded"), sig.detail.get("grounded")


class Gate:
    """The per-episode gate and its log."""

    def __init__(self, cfg: dict, sandbox):
        self.cfg, self.data_dir = cfg, getattr(sandbox, "data_dir", None)
        self.refused: list = []                 # submissions refused, in order
        self.log = {"rejected": 0, "offered": 0, "na": 0, "gate_errors": 0, "accepted": None, "events": []}
        try:
            sandbox._ef_log = self.log          # read by the patched V2Sandbox.stats -> rollout["v2"]["ef"]
        except Exception:                        # noqa: BLE001
            pass

    def accept(self, submission: dict, transcript: list, turn: int, phase: str) -> bool:
        self.log["offered"] += 1
        try:
            g, why = grounded(transcript, self.cfg, self.data_dir, submission)
        except Exception as exc:                 # noqa: BLE001  fail-open: the rollout behaves as in V3
            self.log["gate_errors"] += 1
            self.log["accepted"] = "gate_error"
            self.log["events"].append({"turn": turn, "phase": phase, "error": f"{type(exc).__name__}: {exc}"[:300]})
            return True
        if g is None:
            self.log["na"] += 1
            self.log["accepted"] = "na"
            return True
        if g >= 1.0:
            self.log["accepted"] = "gate"
            return True
        self.log["rejected"] += 1
        self.refused.append(submission)
        self.log["events"].append({"turn": turn, "phase": phase, "why": str(why)[:300]})
        return False

    def fallback(self, transcript: list):
        """At the end of an episode with no accepted answer: the LAST refused answer that is grounded against every
        cell the episode executed, else None."""
        for sub in reversed(self.refused):
            try:
                g, _ = grounded(transcript, self.cfg, self.data_dir, sub)
            except Exception as exc:             # noqa: BLE001
                self.log["gate_errors"] += 1
                self.log["events"].append({"phase": "fallback", "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            if g is not None and g >= 1.0:
                self.log["accepted"] = "fallback"
                return sub
        self.log["accepted"] = None
        return None


def make_solve(A):
    """LLMAgent.solve (harness/agents.py, sha-pinned) with the EF gate. A = the loaded agents module: every name the
    original reads from its module globals is read from A at call time, as the original does."""

    def solve(self, cfg: dict, sandbox) -> tuple:
        task = A.build_prompt(cfg["task"], sandbox.listing(), self.prompt_style)
        if self.diagnose:
            task = A.STAGE_DIAGNOSE + task
        messages = [
            {"role": "system", "content": A.SYSTEMS[os.environ.get("GBP_SYSTEM", "default")]},
            *self.icl_demo,
            {"role": "user", "content": (A.ICL_BRIDGE + task) if self.icl_demo else task},
        ]
        transcript: list = []
        stalled = 0
        last_err, repeats = "", 0
        gate = Gate(cfg, sandbox)                                                   # EF

        t0, wall_turn = time.monotonic(), None
        for turn in range(self.max_turns):
            if wall_turn is None and self.wall_budget and time.monotonic() - t0 > self.wall_budget:
                wall_turn = turn
            left = self.max_turns - turn
            if wall_turn is not None:
                left = min(left, wall_turn + self.force_answer_turns - turn)
                if left <= 0:
                    break
            forcing = left <= self.force_answer_turns
            if forcing:
                fm = self._force_msg(left)
                messages.append({"role": "user", "content": fm})
                transcript.append({"turn": turn, "role": "tool",
                                   "content": "[forcing final answer]", "sent": fm,
                                   **({"wall_budget_spent": True} if wall_turn is not None else {})})

            messages = self._trim(messages)
            reply, finish = self._chat(messages)
            messages.append({"role": "assistant", "content": reply})
            transcript.append({"turn": turn, "role": "assistant", "content": reply})

            ans = A.extract_answer(reply)
            if ans is not None:
                if gate.accept(ans, transcript, turn, "forcing" if forcing else "turn"):   # EF
                    return ans, transcript
                messages.append({"role": "user", "content": FEEDBACK})                   # EF
                transcript.append({"turn": turn, "role": "tool", "content": FEEDBACK,    # EF
                                   "ef": "rejected"})                                    # EF
                stalled = 0                                                              # EF
                continue                                                                 # EF
            if forcing:
                continue

            code = A.extract_code(reply)
            if code:
                stalled = 0
                res = sandbox.run(code)
                fb = res.as_feedback()
                sig = (res.stderr or "").strip()[-160:]
                if sig and sig == last_err:
                    repeats += 1
                    if repeats >= 2:
                        fb += ("\n\nHINT: this is the same error as your previous "
                               "attempt. Stop guessing and inspect first -- e.g. "
                               "`np.load(path).files` for .npz keys, `df.columns` "
                               "for table columns, `df.head()` for values.")
                        repeats = 0
                else:
                    repeats = 0
                last_err = sig
                messages.append({"role": "user", "content": fb})
                transcript.append({"turn": turn, "role": "tool", "content": fb})
                continue

            if finish == "length":
                nudge = (
                    "Your reply was cut off before producing any output. Stop "
                    "reasoning and immediately emit a short ```python block "
                    "that loads and inspects the data files. Keep it brief."
                )
            else:
                nudge = (
                    "No ```python block and no parseable JSON answer outside your "
                    "reasoning. Emit an actual ```python block to inspect the data, "
                    "or the final JSON object exactly as specified. Do not answer "
                    "from reasoning alone -- compute the numbers."
                )
            stalled += 1
            if stalled >= self.max_stalls:
                transcript.append({"turn": turn, "role": "system",
                                   "content": f"aborted: {stalled} turns with no code and no answer"})
                return gate.fallback(transcript), transcript                            # EF (was: None)

            messages.append({"role": "user", "content": nudge})
            transcript.append({"turn": turn, "role": "tool", "content": nudge})

        try:
            messages = self._trim(messages + [
                {"role": "user", "content": self._force_msg(0)}])
            reply, _ = self._chat(messages)
            transcript.append({"turn": self.max_turns, "role": "assistant",
                               "content": reply})
            ans = A.extract_answer(reply)
            if ans is not None and gate.accept(ans, transcript, self.max_turns, "final"):  # EF
                return ans, transcript
        except Exception as exc:                                  # noqa: BLE001
            transcript.append({"turn": self.max_turns, "role": "system",
                               "content": f"final forced call failed: {exc}"})
        return gate.fallback(transcript), transcript                                    # EF (was: None)

    solve._ef = True
    return solve


def install_agents(A) -> None:
    """LLMAgent.solve := the EF loop (idempotent). The original stays reachable as LLMAgent._solve_pre_ef."""
    cls = A.LLMAgent
    if getattr(cls.solve, "_ef", False):
        return
    cls._solve_pre_ef = cls.solve
    cls.solve = make_solve(A)


def install_v2(V) -> None:
    """Provenance and logging in harness/v2.py's namespace (idempotent)."""
    HC, SB = V.HarnessConfig, V.V2Sandbox
    if getattr(HC.as_dict, "_ef", False):
        return
    orig_as_dict, orig_stats, orig_banner = HC.as_dict, SB.stats, V.banner

    def as_dict(self) -> dict:
        d = orig_as_dict(self)
        d["exec_forcing"] = record()
        return d

    def stats(self) -> dict:
        s = orig_stats(self)
        log = getattr(self, "_ef_log", None)
        if log is not None:
            s["ef"] = dict(log)
        return s

    def banner(hc) -> str:
        return orig_banner(hc) + " + EXEC-FORCING (answers accepted only if grounded; experiments/harness_v3/exec_forcing.py)"

    as_dict._ef = stats._ef = banner._ef = True
    HC.as_dict, SB.stats, V.banner = as_dict, stats, banner
