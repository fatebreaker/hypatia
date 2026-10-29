"""The self-verifying agent's control flow, driven by a scripted model and a stub sandbox.

No server and no GPU: this checks the machine, not the science -- that the stages run in order, that a
SELFCHECK line the harness can READ is what advances it (a model that only claims success does not pass),
that a failing check triggers repair up to the cap, and that the answer still comes back when the check
never passes.

  venv/bin/python harness/test_selfver.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("GBP_BASE_URL", "http://127.0.0.1:1/v1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agents                                                     # noqa: E402


class StubResult:
    def __init__(self, out="ok", err=""):
        self.stdout, self.stderr = out, err

    def as_feedback(self):
        return self.stdout + (f"\nSTDERR: {self.stderr}" if self.stderr else "")


class StubSandbox:
    def __init__(self):
        self.n = 0
        self.echo = ""

    def listing(self):
        return "data_files/x.csv"

    def run(self, code):
        self.n += 1
        return StubResult(self.echo or f"ran cell {self.n}")


def scripted(replies):
    it = iter(replies)

    def _chat(self, messages):
        try:
            return next(it), "stop"
        except StopIteration:
            return "no more", "stop"
    return _chat


CODE = "```python\nprint(1)\n```"
ANSWER = '```json\n{"answer": {"k": 1.0}}\n```'


def run_case(replies, sandbox_echo="", **kw):
    sb = StubSandbox()
    sb.echo = sandbox_echo
    agents.SelfVerifyAgent._chat = scripted(replies)
    a = agents.SelfVerifyAgent(model="stub", max_turns=30, **kw)
    return a.solve({"task": "estimate k"}, sb)


def main() -> int:
    # 1. the happy path: read, replicate, a passing check, ablate, answer
    ans, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE,
                        f"{CODE}\nSELFCHECK worst_rel_err=0.02", CODE, ANSWER])
    stages = [t.get("stage") for t in tr if t.get("stage")]
    assert ans == {"answer": {"k": 1.0}}, ans
    assert "ablate" in stages and "apply" in stages, stages
    assert any("selfcheck passed" in t["content"] for t in tr if t["role"] == "system"), "no verdict recorded"

    # 2. a model that CLAIMS success without a readable check does not pass verification
    ans, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE,
                        f"{CODE}\nMy analysis is perfect, error is essentially zero.", CODE, CODE, CODE, ANSWER])
    assert ans == {"answer": {"k": 1.0}}
    assert any("selfcheck not passed" in t["content"] for t in tr if t["role"] == "system"), \
        "an unverifiable claim was accepted"
    assert "ablate" not in [t.get("stage") for t in tr], "ablation ran without a passing check"

    # 3. a failing check repairs, up to the cap, then still answers
    bad = f"{CODE}\nSELFCHECK worst_rel_err=0.90"
    ans, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE, bad, bad, bad, bad, ANSWER],
                       max_repair=2)
    assert ans == {"answer": {"k": 1.0}}
    reps = [t for t in tr if t["role"] == "tool" and t["content"].startswith("[stage repair")]
    assert len(reps) == 2, f"expected 2 repairs, got {len(reps)}"

    # 4. the check may arrive in the tool output rather than the reply
    ans, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE, CODE, CODE, ANSWER],
                       sandbox_echo="SELFCHECK worst_rel_err=0.01")
    assert any("selfcheck passed" in t["content"] for t in tr if t["role"] == "system"), "printed check missed"
    # 5. v5's floor stage: a noiseless check that misses repairs, up to the cap, before the realistic check runs
    floor_bad = f"{CODE}\nFLOOR rel_err=0.50"
    ans, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE,
                        floor_bad, floor_bad, floor_bad,
                        f"{CODE}\nSEPARATION planted=1 right=1 wrong=2",
                        f"{CODE}\nSELFCHECK worst_rel_err=0.01", CODE, ANSWER],
                       require_floor=True, require_trap=True, max_repair=2)
    stages = [t.get("stage") for t in tr if t.get("stage")]
    assert "floor" in stages, stages
    assert stages.index("floor") < stages.index("calibrate"), "the floor must be checked before the realistic draw"
    frep = [t for t in tr if t["role"] == "tool" and t["content"].startswith("[stage floor repair")]
    assert len(frep) == 2, f"expected 2 floor repairs, got {len(frep)}"
    assert any("floor 0.500" in t["content"] for t in tr if t["role"] == "system"), "floor result not recorded"

    # 6. a floor claimed in prose, with no readable FLOOR line, is not a floor -- and does not block the loop
    ans, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE,
                        f"{CODE}\nRecovered it exactly on noiseless data.", CODE, CODE, CODE,
                        f"{CODE}\nSELFCHECK worst_rel_err=0.01", CODE, ANSWER],
                       require_floor=True)
    assert any("floor none" in t["content"] for t in tr if t["role"] == "system"), "an unreadable floor was credited"
    assert any("selfcheck passed" in t["content"] for t in tr if t["role"] == "system"), "the loop stalled at the floor"

    # v1-v4 must be unchanged by v5: no floor stage unless it is asked for
    _, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE,
                      f"{CODE}\nSELFCHECK worst_rel_err=0.02", CODE, ANSWER])
    assert "floor" not in [t.get("stage") for t in tr], "the floor stage leaked into the earlier arms"

    # THE ANSWERING TAX (2026-09-22). A model that keeps emitting CODE in the apply stage and never an answer
    # used to run the budget out in silence: the force-an-answer message was sent only when a reply contained no
    # code. That cost ~9% of problems on both external benchmarks, and on QRData it was the entire deficit.
    # force_tail sends the message anyway once that many apply turns remain. Default 0 must stay a no-op.
    applies = [f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE, f"{CODE}\nSELFCHECK worst_rel_err=0.02", CODE] + [CODE] * 40

    _, tr = run_case(applies)
    assert not any("force_tail" in t["content"] for t in tr if t["role"] == "system"), \
        "force_tail fired with the default of 0 -- every previously measured arm would change"
    assert all("Answer now" not in t.get("content", "") for t in tr if t["role"] == "tool"), "unexpected forcing"

    _, tr = run_case(applies, force_tail=3)
    forced = [t for t in tr if t["role"] == "system" and "force_tail" in t["content"]]
    assert forced, "force_tail=3 never demanded an answer on a run that only ever emitted code"
    assert len(forced) <= 3, f"forced on more turns than the tail allows: {len(forced)}"

    # and it must not fire when the model answers promptly -- the fix is for the tail, not for every run
    _, tr = run_case([f"ESTIMAND: k\nPLAN: 1. fit\n{CODE}", CODE,
                      f"{CODE}\nSELFCHECK worst_rel_err=0.02", CODE, ANSWER], force_tail=3)
    assert not any("force_tail" in t["content"] for t in tr if t["role"] == "system"), \
        "force_tail fired on a run that answered in time"

    print("all self-verifying-agent control-flow tests pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
