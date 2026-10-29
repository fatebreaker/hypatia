"""EXTRA SYSTEM PROMPTS for harness/agents.py, registered from OUTSIDE that file (docs/RL_PLAN.md s.18, v3.1
ELICIT-THEN-REWARD). harness/agents.py is a RUNNING file (live trainers and evaluations import it; its sha256 is pinned in
experiments/harness_v3/tests/running_files.sha256), so it is never edited for a new prompt. Its SYSTEMS is a plain
module-level dict that LLMAgent.solve and rl/dispersion.conversation read at CALL time
(SYSTEMS[os.environ.get("GBP_SYSTEM", "default")]), so a key added to that same dict object before the first agent runs
is used by every rollout of the process -- and by the stored episode, which is what the trainer's logprobs are computed on.

    science_check   the "science" prompt VERBATIM + ONE sentence (CHECK_SENTENCE below), appended after its last line
                    with a single space. Nothing else changes. Purpose: make the robustness re-estimate that the v3
                    process reward's `check` (M3 functional form) and `spec` detectors look for OCCUR at all on causal_rdd,
                    where the 8B base rate of both is 0 (RL_PLAN 2026-09-26 16:20), so the process term has something to
                    select. Science-neutral: it names no science, family, trap or column.

WHO REGISTERS IT
    training     rl/grpo_v31.py calls register() before rl.dispersion / rl.grpo_onpolicy make any agent (the rollouts run
                 in threads of that one process: grpo_onpolicy and dispersion use ThreadPoolExecutor).
    evaluation   experiments/harness_v3/screen_v31.sh puts experiments/harness_v3/v31_site/ on PYTHONPATH; its
                 sitecustomize.py calls register(module) the moment any python process of that screen imports `agents`
                 (every harness entry point imports it as the top-level module `agents`, with harness/ on sys.path).
    anything else (screen_v3.sh, eval_ext.sh, experiments/acquire, ...) does NOT see the key. An episode under
    GBP_SYSTEM=science_check there is never run under a wrong prompt, but it is NOT loud either: rl/dispersion.one_rollout
    catches the KeyError and records the rollout as attribution error:KeyError, error_msg "'science_check'", 0 cells,
    pass 0 (measured, experiments/harness_v3/tests/test_v31.py) -- a silent 0% that read_v31.py flags. Score a v3.1
    checkpoint only through experiments/harness_v3/screen_v31.sh.

    venv/bin/python rl/systems_extra.py --print science_check      the exact text (for the registration log)
    venv/bin/python rl/systems_extra.py --check science_check      exit 0 iff the key is defined here
    venv/bin/python rl/systems_extra.py --sha science_check        sha256 of the text (first 16 hex)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_KEY = "science"
CHECK_SENTENCE = ("After your main estimate, re-estimate the same quantity once in a later code block under an "
                  "alternative specification or model form, print both estimates, and comment on the difference before "
                  "you answer.")
# science_strict (2026-10-05, H-PROMPT-GROUND): the "science" prompt VERBATIM + STRICT_SENTENCE -- the grounded reward's
# criterion (rl/grpo_rg.py; the v2 grounding label) stated as an instruction, to ask whether a prompt alone makes a model
# compute its answers. Evaluation only (v31_site + V31_SYSTEM=science_strict); no training run uses it.
STRICT_SENTENCE = ("Every number in your final answer must be printed by code you ran in this session and computed by "
                   "that code, not typed into it; an answer your code did not compute is graded as wrong.")
KEYS = ("science_check", "science_strict")


def _agents():
    h = str(ROOT / "harness")
    if h not in sys.path:
        sys.path.insert(0, h)
    import agents  # noqa: E402  (the SAME module object rl/dispersion.py and harness/runner.py import)
    return agents


def texts(agents_mod=None) -> dict:
    """{key: full system prompt}, built from the agents module's own "science" text (never a copy of it)."""
    a = agents_mod if agents_mod is not None else _agents()
    return {"science_check": a.SYSTEMS[BASE_KEY] + " " + CHECK_SENTENCE,
            "science_strict": a.SYSTEMS[BASE_KEY] + " " + STRICT_SENTENCE}


def register(agents_mod=None):
    """Add KEYS to agents.SYSTEMS (in place). Idempotent; refuses a key that is already there with OTHER text."""
    a = agents_mod if agents_mod is not None else _agents()
    for k, v in texts(a).items():
        cur = a.SYSTEMS.get(k)
        if cur is not None and cur != v:
            raise RuntimeError(f"rl/systems_extra.py: agents.SYSTEMS[{k!r}] already exists with different text")
        a.SYSTEMS[k] = v
    return a


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main(argv: list) -> int:
    if len(argv) != 2 or argv[0] not in ("--print", "--check", "--sha"):
        print(__doc__, file=sys.stderr)
        return 2
    op, key = argv
    if key not in KEYS:
        print(f"rl/systems_extra.py: unknown key {key!r} (defined: {', '.join(KEYS)})", file=sys.stderr)
        return 1
    if op == "--check":
        return 0
    t = texts()[key]
    print(t if op == "--print" else sha(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
