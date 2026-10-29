"""HARNESS v5 -- "WRITE THE METHOD, TEST IT ON NEW DATA" (docs/RL_PLAN.md 2026-09-29 14:15 UTC, "HARNESS v5 ... REGISTERED before
any build or run"; this file is the registered protocol's build, its open details are listed under DECISIONS below).

PROTOCOL (the registration's, verbatim in substance). v3 (KERNEL + science prompt + notebook echo, NO library) exploration;
then the agent must SUBMIT a function `solve(data_dir, task)` returning the task's answer keys (AMENDED 2026-09-29 15:55 UTC,
before any v5 run: the harness passes each dataset's OWN task text as `task`, so a method is a function of (data, question) --
in 7 of the 17 TEST families the question carries a parameter that is only in the task text). The harness runs solve on the
target and on K=3 layout-matched UNLABELLED train siblings (the target family's TRAIN split -- sealed TEST families included, as the
census did -- drawn among instances whose layout equals the target's: rl/grpo_prog.layout_key = data-file names + header
lines + eval_config roles), in the sandbox with the usual limits, and replies with a FIXED-FORMAT REPORT (per sibling: value
or exception; flags: exception, missing / non-numeric keys, constant output across siblings, output identical to the
target's). The agent may revise and resubmit (at most MAX_REVISIONS = 2 revisions, i.e. 3 submissions; turn / wall limits
unchanged). FINAL ANSWER = solve(target data, target task) of the LAST VALID submission (grounded by construction; there is NO gate). No valid
submission by the end -> the model's LAST OFFERED JSON answer (v4's soft fallback); none offered -> no answer. Externals
(QRData / DABench / BixBench): no siblings -> the same protocol without the sibling run.

SUBMISSION CONVENTION (the text NOTE appends to the task, v5 only; its sha is in the config record). A reply whose executable
code block (harness/agents.extract_code: the FIRST ```python block, reasoning stripped) defines a top-level `def solve(` is a
SUBMISSION. It is NOT run as a notebook cell: the harness runs it, unchanged, in fresh sandboxes (see RUN) and replies with the
report. Every other reply is handled as in v3 (a code block runs as a notebook cell; nudges; the stall abort).

RUN (one submission). The submission's source is exec'd in a fresh namespace inside one cell of a FRESH sandbox of the
episode's own class (type(sandbox): the v3 kernel sandbox, its per-cell wall timeout, CPU / memory / process limits and
environment), and solve("data_files", task) is called there (task = THAT dataset's own task text, see TASK TEXT below; a
solve that cannot take two positional arguments is an exception "solve must take two arguments"); the cell writes {status, value | error} to a result file in the
sandbox's own working directory (never parsed from STDOUT, so solve's own printing cannot hide or fake it). One sandbox per
dataset, all 1 + K in parallel threads:
  target     data_files/ = a fresh copy of the target's ORIGINAL data files (as every sandbox gets them); for data over
             LINK_OVER_BYTES (256 MB: only a few BixBench problems) a symlink to the episode's own working copy instead, so a
             submission never copies a GB-sized problem again (the episode copy is private to the episode; nothing else sees it)
  sibling_i  data_files/ = a fresh copy of the sibling's data_files ONLY (rl/grpo_prog.sibling_data_dir: symlinks to the
             sibling's data files under the design's names, copied by the sandbox).
TASK TEXT (the amendment). target: cfg["task"] with the v5 HOW-TO-ANSWER note removed (the question exactly as the benchmark /
generator states it; on the externals: the benchmark's question text). sibling_i: ONLY the "task" field of that sibling's
eval_config.json (sibling_task) -- the question the generator wrote for that instance, the same kind of text the agent reads
for its own instance: it states the answer SCHEMA (key names) but never the answers (audited 2026-09-29 over all 7,423 TRAIN
instances of the 17 TEST families + causal_rdd: no label statement, no ground-truth / provenance text; the only 3-significant-
digit coincidences are 5 pk_oral assay parameters such as "10.3% proportional error"). Every other field of a sibling's
eval_config (answers, grader, provenance) is read only by layout_key (its `_provenance.roles`, to pick WHICH siblings) and
never reaches the model, the submission or the report.
VALID = the TARGET run returned a dict holding every answer key (the task's JSON schema = cfg["ground_truth"]'s KEYS), each
numeric key (rl.reward.StepReward._fields: every key of the 17 TEST families and causal_rdd; none on the externals, whose
labels are strings) a finite number. Sibling results never decide validity (they are information for the agent).
FLAGS (per submission, in the report and the log): exception (a run that raised / timed out / died / returned no dict),
missing (answer keys absent), non-numeric (numeric keys that are not finite numbers), constant (answer keys whose value is the
same on every sibling that returned it, >= 2), identical (siblings whose whole answer equals the target's).
PORTABILITY (logged here; the Phase-2 reward reads it): the fraction of the K siblings where solve ran without exception,
returned every answer key (numeric keys finite) and an output that differs from the target's AND from every other clean
sibling's ("non-constant": a program whose output does not change with the data scores 0 on every sibling). 0 when the
submission is not valid. The final answer's submission's portability is the rollout's.

DECISIONS (details the registration leaves open; the simplest faithful option each, fixed here before any run):
  1. JSON answer BEFORE any valid submission, outside the forcing tail, with submissions left: refused with the one fixed
     message NEED_SUBMIT (recorded as offered; the turn is consumed, the stall counter resets -- EF's refusal mechanics); the
     episode continues. In the forcing tail, or with no submissions left: the episode ends on it (the soft fallback: it IS
     the model's last offered answer).
  2. JSON answer AFTER a valid submission: the episode ends; the answer is solve(target) of the last valid submission (the
     JSON's values are not used -- "grounded by construction").
  3. The forcing tail (turn limit / wall budget, v3's) starts with a valid submission in hand: the episode ends at once with
     it (the forcing turns only ask for an answer, and the answer is already fixed). Without one: v3's forcing turns
     verbatim; a submission there is still run (a valid one ends the episode), a JSON answer ends it (decision 1).
  4. The 3rd submission (no revisions left): a valid submission in hand -> the episode ends; none -> the report says so and
     the next JSON answer ends the episode (decision 1); a 4th submission is not run (NO_SUBMISSIONS_LEFT, logged).
  5. The final forced call after the loop (v3's) is made only without a valid submission; a submission there is run.
  6. A submission counts as one executed code block of the episode (sandbox.attempts += 1), so --require-code (RL) and the
     runner's "answered without code" test treat an episode whose only code was its submission as one that ran code.
  7. The report shows values to 6 significant digits, errors as "Type: message (solve line N)", no STDOUT of solve.
  8. Siblings are drawn once per target with the fixed key SIB_KEY (the same 3 for every arm, checkpoint and rollout of that
     target: paired arms see the same siblings, and every rollout of a GRPO group is judged on the same data).
  9. The target / sibling runs are made in worker threads (never the rollout thread), so harness/v2's per-thread "last
     sandbox built" (-> the rollout's stats) stays the episode's sandbox.
LOG (per rollout -> rollout["v2"]["v5"] through V2Sandbox.stats; externals -> rec["v5"] through runner.run_one): harness,
k, siblings, submissions, valid, revisions, offered, refused, refused_submissions, final ("solve" | "soft_fallback" | None),
final_submission, end (how the episode ended), portability, subs (per submission: turn, phase, valid, why, code, code_sha,
target run, per-sibling runs with their flags, the flag table, portability, seconds).

HOW IT REACHES THE HARNESS (no pinned or running file is edited; the exec_forcing.py pattern). make_solve(A) is
LLMAgent.solve VERBATIM with the v5 deviation points marked "# V5" (experiments/harness_v3/tests/test_v5.py proves that with
enabled=False the copy reproduces LLMAgent.solve's transcripts, prompts and cells on every branch test_ef.py covers). Loaded
ONLY by experiments/harness_v3/v5_site/sitecustomize.py (GBP_V5=1; screen_v3v5.sh) or in-process by rl/grpo_v5.py (the
Phase-2 trainer):
  install_agents(A)  LLMAgent.solve := the v5 loop (SelfVerifyAgent keeps its own solve).
  install_v2(V)      HarnessConfig.as_dict gains "program_first_v5" (record(): harness name, K, caps, the note's sha, this
                     file's and the site's sha) -- so a V3V5 directory can never mix with v3 / v4 rows (screen_any.sh
                     refuses a config.json written under another harness); V2Sandbox.stats gains the per-rollout log;
                     v2.wrap_rollout's rollouts get the note appended to cfg["task"] BEFORE the episode, so the stored
                     episode (rl/dispersion.conversation) carries the prompt the model actually read (the loop appends the
                     same note only when it is not there yet); the banner says PROGRAM-FIRST v5.
  install_runner(R)  externals: runner.run_one adds rec["v5"] and rewrites that problem's JSON atomically.

HARNESS v5.1 (docs/RL_PLAN.md 2026-09-29 21:10 UTC, "HARNESS v5.1 REGISTERED"): v5 + (a) LOOK FIRST, FLAG-GATED. With
LOOK_FIRST False (the default; configure() sets it) every line below is inert and the loop, the log, the record, the banner
and the as_dict key are v5's exactly (tests/test_v51.py runs this file with the flag off against the frozen v5 build,
tests/fixtures/program_first_v5_prebuild.py, on every protocol branch). With it on (v5_site + GBP_V51=1, i.e.
screen_v3v5.sh V51=1; or rl/grpo_v51.py in the trainer):
  RULE     a solve() submission is REFUSED -- not run, not counted as a submission or as executed code; one fixed harness
           message (LOOK_REFUSAL; in the forcing tail LOOK_REFUSAL_FORCING, since the forcing turns run no cell); the
           refusal uses the turn -- until at least one EXECUTED notebook cell of the episode has READ a data file of the
           target. Everything else is v5's (a JSON answer beside a refused submission is recorded as offered; NEED_SUBMIT,
           the cap, the soft fallback, the forcing tail and the final forced call are unchanged; a refused submission
           resets the stall counter as v5's report does). TERMINATION: the refusal consumes a turn of the unchanged
           max_turns / wall-budget loop, so an episode that never looks ends by the forcing tail (its submissions refused
           with LOOK_REFUSAL_FORCING, a JSON answer there ends it as the soft fallback), the final forced call (a submission
           there is refused and logged; no further turn) or the stall abort -> v5's soft fallback / no answer.
  DETECTION (LookWatch; decided here, justified in the 21:40 build entry). "Read a data file of the target" = while the
           cell ran, some process read >= 1 byte from an ORIGINAL file under the episode's own copy of the data
           (<sandbox workdir>/data_files: the files present when the episode started; files the agent writes there later
           do not count) -- observed at the SANDBOX level through Linux inotify (IN_ACCESS on that file; the kernel emits it
           for every successful read by any process, so pandas / numpy / h5py / pyarrow / a `head` subprocess all count,
           while os.listdir, stat, glob and printing the string 'data_files' do not). ONE inotify instance per process
           (Linux allows 128 per user per node; 48 concurrent RL episodes each holding one could exhaust it), a watch per
           episode data directory, events dispatched to the owning episode under a lock; the watch is removed as soon as
           the episode has looked (look is monotone), so the event volume is bounded by the first reading cell.
           FALLBACK (logged per episode, look_detector 'static' + look_error; per cell on an inotify queue overflow): the
           STATIC rule -- the cell ran without error (returncode 0, no timeout) and its code names data_files or a data
           file's name. VACUOUS: a target with no data files is looked at from the start (look_how 'vacuous'), so a
           problem without data can never block submission. NOT counted by inotify: a read done ONLY by mmap (e.g.
           np.load(mmap_mode=...)) -- the refusal message then asks for an ordinary read; at most one turn.
  LOG      rollout["v2"]["v5"] / rec["v5"] gains (look-first episodes only): look_first, looked, look_turn, look_how
           ('inotify' | 'static' | 'vacuous'), look_file, look_detector, look_error, look_refusals, look_overflow; harness
           'program_first_v51'. The record (as_dict key 'program_first_v51', harness 'program_first_v51') adds version 5.1
           and look_first {rule, refusal sha}.
  (b) LABELLED PORTABILITY is TRAINING-ONLY and lives in rl/grpo_v51.py (the trainer's reward reads the siblings' labels
      after the episode, behind the sealed-family guard): this harness file never reads a label, in training or
      evaluation.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "harness")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HARNESS = "program_first_v5"
FLAG = "GBP_V5"
SITE = HERE / "v5_site" / "sitecustomize.py"
K_SIBLINGS = 3
MAX_REVISIONS = 2
MAX_SUBMISSIONS = 1 + MAX_REVISIONS
SIB_KEY = "program_first_v5"
CALL_ARG = "data_files"
RESULT_FILE = "_v5_solve_result.json"
LINK_OVER_BYTES = 256 * 1024 ** 2
SUB_RE = re.compile(r"^def[ \t]+solve[ \t]*\(", re.M)
_TLS = threading.local()                   # the current episode's log (read by install_runner's run_one)
# v5.1 (docs/RL_PLAN.md 2026-09-29 21:10 UTC): (a) LOOK FIRST, flag-gated -- LOOK_FIRST False = v5 exactly
HARNESS_V51 = "program_first_v51"
FLAG_V51 = "GBP_V51"
LOOK_FIRST = False                         # set by configure(): v5_site with GBP_V51=1, rl/grpo_v51.py
INOTIFY_OFF = False                        # tests only: force the static detector

NOTE_SIB = """

HOW TO ANSWER IN THIS HARNESS. Explore the data with ```python cells as usual. When your analysis is ready, submit it AS A \
PROGRAM: reply with ONE ```python block that defines

def solve(data_dir, task):
    ...
    return {...}   # a dict with every key of the answer object above

data_dir is a folder with the same files as your data_files/; task is the text of that dataset's question -- for your \
dataset, the task above (without this section); for another dataset, ITS OWN question of the same kind, whose parameters \
(names, codes, thresholds, doses, ...) may differ from yours, so read such parameters from task instead of typing yours in. \
solve must be self-contained -- its own imports, its input read only from data_dir and task, nothing from your earlier \
cells -- because the harness runs it in fresh Python processes: on YOUR dataset and on @K@ more datasets of the same study \
design (the same files and layout, different data, no answers). It replies with what solve returned on each dataset, or the error it raised, and flags errors, missing keys or non-numeric \
values where a number is expected, outputs that do not change with the data, and outputs identical to yours. You may fix \
solve and submit it again, up to @R@ more times. Your final answer is what your LAST WORKING solve returns on YOUR dataset: \
once a working solve exists, numbers you type into a JSON answer are not used. When you are satisfied, reply with your \
final JSON answer."""
NOTE_TARGET = """

HOW TO ANSWER IN THIS HARNESS. Explore the data with ```python cells as usual. When your analysis is ready, submit it AS A \
PROGRAM: reply with ONE ```python block that defines

def solve(data_dir, task):
    ...
    return {...}   # a dict with every key of the answer object above

data_dir is a folder with the same files as your data_files/; task is the text of the question above (without this \
section). solve must be self-contained -- its own imports, its input read only from data_dir and task, nothing from your \
earlier cells -- because the harness runs it in a fresh Python process on your dataset. It replies with what solve returned, or the error it raised, and flags missing keys or non-numeric values \
where a number is expected. You may fix solve and submit it again, up to @R@ more times. Your final answer is what your \
LAST WORKING solve returns: once a working solve exists, numbers you type into a JSON answer are not used. When you are \
satisfied, reply with your final JSON answer."""
NEED_SUBMIT = ("Your JSON answer was not used: in this harness the answer comes from a program. Submit your method as ONE "
               "```python block that defines solve(data_dir, task) (see HOW TO ANSWER IN THIS HARNESS in the task); what it "
               "returns on your dataset becomes your answer.")
NO_SUBMISSIONS_LEFT = "No submissions left: this solve was not run. Reply with your final JSON answer."
FALLBACK = ("no valid submission by the end of the episode (turn limit incl. the final forced call, wall budget, stall "
            "abort, or a JSON answer in the forcing tail / after the last submission) -> the LAST JSON answer the model "
            "offered (v4's soft fallback, logged 'soft_fallback'); none offered -> no answer")
LOOK_REFUSAL = ("Your solve() was NOT run: in this harness a method is run only after you have looked at your data. First "
                "run at least one ```python cell that reads a file in data_files/ (for example, load it and print its first "
                "rows, its columns and summary statistics); then submit solve(data_dir, task) again. This refusal did not "
                "use up one of your submissions.")
LOOK_REFUSAL_FORCING = ("Your solve() was NOT run: no cell of yours has read a file in data_files/, and a method is run only "
                        "after you have looked at your data. The turn limit is reached -- reply with your final JSON answer.")
LOOK_RULE = ("a solve() submission is refused (not run, not counted; LOOK_REFUSAL, in the forcing tail LOOK_REFUSAL_FORCING; "
             "the refusal uses the turn) until an EXECUTED notebook cell of the episode has READ an original data file of "
             "the target (the episode's data_files/ copy as it was at the start): inotify IN_ACCESS on that file while the "
             "cell ran (any process); fallback when inotify is unavailable or overflowed: the cell ran without error and "
             "its code names data_files or a data file; a target without data files is looked at vacuously")


def configure(look_first: bool = False) -> None:
    """v5.1 (a): LOOK FIRST on / off for every episode started after the call (off = v5 exactly)."""
    global LOOK_FIRST
    LOOK_FIRST = bool(look_first)


def harness_name() -> str:
    return HARNESS_V51 if LOOK_FIRST else HARNESS


def note(k: int) -> str:
    return (NOTE_SIB.replace("@K@", str(k)) if k > 0 else NOTE_TARGET).replace("@R@", str(MAX_REVISIONS))


def with_note(task: str, k: int) -> str:
    """The task as v5 shows it (idempotent: a task that already ends with the note is returned unchanged)."""
    n = note(k)
    return task if task.endswith(n) else task + n


def base_task(task: str) -> str:
    """The task text without the v5 note (what solve receives for the target: the question as stated)."""
    for k in range(0, K_SIBLINGS + 1):
        n = note(k)
        if task.endswith(n):
            return task[: -len(n)]
    return task


def sibling_task(sib: Path) -> str:
    """A sibling's OWN question: ONLY the "task" field of its eval_config.json (never its answers, grader or provenance)."""
    return str(json.loads((Path(sib) / "eval_config.json").read_text()).get("task") or "")


def _sha16(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def record() -> dict:
    """What config.json / each family JSON's "harness_v2" / each ext harness_v2.json records (key "program_first_v5")."""
    notes = NOTE_SIB + NOTE_TARGET + NEED_SUBMIT + NO_SUBMISSIONS_LEFT
    rec = {"harness": HARNESS, "on": True, "k_siblings": K_SIBLINGS, "max_revisions": MAX_REVISIONS,
            "max_submissions": MAX_SUBMISSIONS, "sibling_key": SIB_KEY,
            "siblings": "layout-matched (rl/grpo_prog.layout_key) instances of the target family's TRAIN split, target "
                        "excluded, a seeded draw keyed (train dir, target, sibling_key); none for non-generated problems",
            "signature": "solve(data_dir, task)", "amendment": "2026-09-29 15:55 UTC: task = each dataset's own task text",
            "call": f"solve({CALL_ARG!r}, task) in a fresh sandbox of the episode's class per dataset (per-cell timeout = the "
                    f"episode's); result via {RESULT_FILE}",
            "final": "solve(target) of the last valid submission", "fallback": FALLBACK,
            "note_sha16": hashlib.sha256(notes.encode()).hexdigest()[:16],
            "sha256": {"experiments/harness_v3/program_first.py": _sha16(Path(__file__).resolve()),
                       "experiments/harness_v3/v5_site/sitecustomize.py": _sha16(SITE)}}
    if LOOK_FIRST:                                                        # v5.1 (a); off: v5's record exactly
        rec.update(harness=HARNESS_V51, version="5.1", registered="docs/RL_PLAN.md 2026-09-29 21:10 UTC",
                   look_first={"on": True, "rule": LOOK_RULE,
                               "refusal_sha16": hashlib.sha256((LOOK_REFUSAL + LOOK_REFUSAL_FORCING).encode()).hexdigest()[:16]})
    return rec


# ============================================================================================================ helpers
def is_submission(code: str | None) -> bool:
    """A code block that defines a top-level `def solve(` (after dedent)."""
    if not code:
        return False
    return SUB_RE.search(textwrap.dedent(code)) is not None


def answer_keys(cfg: dict) -> list:
    return list((cfg.get("ground_truth") or {}).keys())


def numeric_keys(cfg: dict) -> list:
    """rl.reward.StepReward._fields: the keys graded as numbers (never the externals' string labels)."""
    try:
        from rl.reward import StepReward
        return list(StepReward._fields(cfg))
    except Exception:                                                     # noqa: BLE001
        return []


def _num(v):
    """A finite float for a number (not a bool) or a string that is exactly one; else None."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, str):
        try:
            x = float(v.strip())
        except ValueError:
            return None
        return x if math.isfinite(x) else None
    return None


def _same(a, b) -> bool:
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-12 * max(1.0, abs(na), abs(nb))
    return a == b


def instance_dir(data_dir) -> Path | None:
    """generated/<family>/<split>/<instance>/data_files -> the instance dir (resolved), else None (externals, fixtures
    without an eval_config)."""
    if not data_dir:
        return None
    try:
        d = Path(data_dir).resolve()
    except OSError:
        return None
    inst = d.parent if d.name == "data_files" else d
    if not (inst / "eval_config.json").is_file() or len(inst.parents) < 3:
        return None
    if inst.parent.parent.parent.name != "generated":
        return None
    return inst


def siblings_for(data_dir, k: int = K_SIBLINGS, key: str = SIB_KEY) -> list:
    """K layout-matched instances of the target family's TRAIN split (target excluded), a seeded draw keyed (train dir,
    target, key); all of them if fewer; [] for a data dir that is not a generated instance."""
    inst = instance_dir(data_dir)
    if inst is None or k <= 0:
        return []
    train = inst.parent.parent / "train"
    if not train.is_dir():
        return []
    from rl import grpo_prog as GP
    from rl import siblings as SIB
    mine = GP.layout_key(inst)
    pool = [p for p in SIB._instances(train)
            if p.resolve() != inst and p.name != inst.name and GP.layout_key(p) == mine]
    if len(pool) <= k:
        return sorted(pool)
    seed = int(hashlib.sha256(f"{train.resolve()}|{inst.name}|{key}".encode()).hexdigest()[:16], 16)
    return random.Random(seed).sample(pool, k)


def _dir_bytes(d: Path) -> int:
    n = 0
    for root, _, files in os.walk(d, followlinks=True):
        for f in files:
            try:
                n += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return n


RUNNER = '''import json as _v5_json, time as _v5_time, traceback as _v5_tb
_v5_t0 = _v5_time.time()
_v5_out = {"status": "exception", "error": "the runner did not finish"}
try:
    _v5_ns = {"__name__": "__v5_submission__", "__builtins__": __builtins__}
    exec(compile(_V5_SRC, "<solve>", "exec"), _v5_ns)
    if not callable(_v5_ns.get("solve")):
        raise NameError("the submission defines no function solve")
    try:
        import inspect as _v5_inspect
        _v5_inspect.signature(_v5_ns["solve"]).bind(_V5_ARG, _V5_TASK)
    except (TypeError, ValueError):
        raise TypeError("solve must take two arguments: solve(data_dir, task)") from None
    _v5_res = _v5_ns["solve"](_V5_ARG, _V5_TASK)
    if isinstance(_v5_res, dict):
        _v5_out = {"status": "ok", "value": {str(_k): _v5_v for _k, _v5_v in _v5_res.items()}}
    else:
        _v5_out = {"status": "not_a_dict", "error": "solve returned " + type(_v5_res).__name__ + " " + repr(_v5_res)[:120] + ", not a dict"}
except BaseException as _v5_e:
    _v5_ln = [f.lineno for f in _v5_tb.extract_tb(_v5_e.__traceback__) if f.filename == "<solve>"]
    if isinstance(_v5_e, SyntaxError) and _v5_e.filename == "<solve>":
        _v5_ln = [_v5_e.lineno]
    _v5_out = {"status": "exception", "error": (type(_v5_e).__name__ + ": " + str(_v5_e))[:300],
               "line": _v5_ln[-1] if _v5_ln else None}
def _v5_conv(o):
    try:
        if hasattr(o, "item") and getattr(o, "size", 1) == 1:
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:
        pass
    return str(o)[:200]
_v5_out["seconds"] = round(_v5_time.time() - _v5_t0, 3)
try:
    _v5_txt = _v5_json.dumps(_v5_out, default=_v5_conv, allow_nan=True)
except BaseException as _v5_e:
    _v5_txt = _v5_json.dumps({"status": "exception", "error": "the returned value could not be recorded: " + str(_v5_e)[:200]})
with open(_V5_RESULT, "w") as _v5_f:
    _v5_f.write(_v5_txt)
'''


def runner_cell(code: str, task: str = "") -> str:
    """The one cell a fresh sandbox runs: the submission exec'd in its own namespace, solve(CALL_ARG, task) called, the
    outcome written to RESULT_FILE in the sandbox's working directory (the cell's cwd). task = that dataset's own task text."""
    return (f"_V5_SRC = {textwrap.dedent(code)!r}\n_V5_ARG = {CALL_ARG!r}\n_V5_TASK = {str(task)!r}\n"
            f"_V5_RESULT = {RESULT_FILE!r}\n" + RUNNER)


FRESH = None          # tests: a callable(episode_sandbox, data_dir, workdir, timeout) -> sandbox; None = type(sandbox)(...)


def _fresh(sandbox, data_dir, workdir, timeout):
    if FRESH is not None:
        return FRESH(sandbox, data_dir, workdir, timeout)
    return type(sandbox)(data_dir=data_dir, workdir=workdir, timeout=timeout)


def run_once(cell: str, sandbox, sib: Path | None, timeout: int) -> dict:
    """One dataset: a fresh sandbox, the cell, the result file. Never raises."""
    t0 = time.time()
    tmpd, wd, fresh = None, None, None
    try:
        wd = Path(tempfile.mkdtemp(prefix="v5run_"))
        if sib is None:                                                   # the target
            src = Path(sandbox.data_dir) if getattr(sandbox, "data_dir", None) else None
            own = Path(getattr(sandbox, "workdir", "") or "") / "data_files"
            if src is not None and src.is_dir() and own.is_dir() and _dir_bytes(src) > LINK_OVER_BYTES:
                (wd / "data_files").symlink_to(own.resolve(), target_is_directory=True)
            data = src
        else:
            from rl import grpo_prog as GP
            tmpd = GP.sibling_data_dir(sib)
            data = tmpd
        fresh = _fresh(sandbox, data, wd, timeout)
        res = fresh.run(cell)
        rf = Path(getattr(fresh, "workdir", wd)) / RESULT_FILE
        if getattr(res, "timed_out", False):
            out = {"status": "timeout", "error": f"timed out after {timeout} s"}
        elif not rf.is_file():
            tail = (getattr(res, "stderr", "") or "").strip().splitlines()[-2:]
            out = {"status": "crashed", "error": ("the process died before solve returned"
                                                   + (": " + " | ".join(tail) if tail else ""))[:300]}
        else:
            try:
                out = json.loads(rf.read_text())
            except ValueError as exc:
                out = {"status": "exception", "error": f"unreadable result: {exc}"[:300]}
    except Exception as exc:                                              # noqa: BLE001
        out = {"status": "harness_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        if fresh is not None:
            try:
                fresh.cleanup()
            except Exception:                                             # noqa: BLE001
                pass
        for d in (wd, tmpd):
            if d is not None:
                shutil.rmtree(d, ignore_errors=True)
    out["wall"] = round(time.time() - t0, 2)
    return out


def check(run: dict, keys: list, numeric: list) -> dict:
    """missing / non-numeric keys and the clean test of one run."""
    ok = run.get("status") == "ok" and isinstance(run.get("value"), dict)
    val = run.get("value") if ok else {}
    missing = [k for k in keys if k not in val or val.get(k) is None] if ok else list(keys)
    nonnum = [k for k in numeric if ok and k in val and val.get(k) is not None and _num(val[k]) is None]
    return {"ran": ok, "missing": missing, "nonnumeric": nonnum, "clean": ok and not missing and not nonnum}


def _answer(val: dict, keys: list) -> dict:
    return {k: val[k] for k in keys if k in val} if keys else dict(val)


def _outputs_same(a: dict, b: dict, keys: list) -> bool:
    ks = keys or sorted(set(a) | set(b))
    return all(k in a and k in b and _same(a[k], b[k]) for k in ks)


def evaluate(target: dict, sibs: list, keys: list, numeric: list) -> dict:
    """VALID, the flags and the portability of one submission from its runs (target + [(label, name, run)])."""
    tc = check(target, keys, numeric)
    tval = _answer(target.get("value") or {}, keys) if tc["ran"] else None
    rows = []
    for label, name, run in sibs:
        c = check(run, keys, numeric)
        val = _answer(run.get("value") or {}, keys) if c["ran"] else None
        rows.append({"label": label, "name": name, **c, "value": val, "status": run.get("status"),
                     "error": run.get("error"), "line": run.get("line"), "seconds": run.get("seconds"),
                     "wall": run.get("wall"),
                     "same_as_target": bool(tval is not None and val is not None and not c["missing"]
                                            and _outputs_same(val, tval, keys))})
    constant = []
    for k in keys:
        vs = [r["value"][k] for r in rows if r["ran"] and k in (r["value"] or {}) and r["value"][k] is not None]
        if len(vs) >= 2 and all(_same(v, vs[0]) for v in vs[1:]):
            constant.append(k)
    for r in rows:
        others = [o for o in rows if o is not r and o["clean"]]
        r["portable"] = bool(tc["clean"] and r["clean"] and not r["same_as_target"]
                             and not any(_outputs_same(r["value"], o["value"], keys) for o in others))
    flags = {"exception": (["target"] if not tc["ran"] else []) + [r["label"] for r in rows if not r["ran"]],
             "missing": (["target"] if (tc["ran"] and tc["missing"]) else []) + [r["label"] for r in rows
                                                                                 if r["ran"] and r["missing"]],
             "nonnumeric": (["target"] if tc["nonnumeric"] else []) + [r["label"] for r in rows if r["nonnumeric"]],
             "constant": constant, "identical": [r["label"] for r in rows if r["same_as_target"]]}
    port = (sum(r["portable"] for r in rows) / len(rows)) if (rows and tc["clean"]) else 0.0
    why = None
    if not tc["ran"]:
        why = f"{target.get('status')}: {target.get('error')}"
    elif tc["missing"] or tc["nonnumeric"]:
        why = "missing " + ", ".join(tc["missing"]) if tc["missing"] else "non-numeric " + ", ".join(tc["nonnumeric"])
    return {"valid": tc["clean"], "why": why, "target": {**tc, "value": tval, "status": target.get("status"),
                                                          "error": target.get("error"), "line": target.get("line"),
                                                          "seconds": target.get("seconds"), "wall": target.get("wall")},
            "siblings": rows, "flags": flags, "portability": round(port, 6)}


def _fmt(v) -> str:
    x = _num(v) if not isinstance(v, str) else None
    if x is not None:
        return f"{x:.6g}"
    if isinstance(v, str):
        return json.dumps(v[:60])
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return f"<list of {len(v)}>"
    return json.dumps(str(v)[:60])


def _show(r: dict) -> str:
    if r.get("ran"):
        val = r.get("value") or {}
        body = ", ".join(f'"{k}": {_fmt(v)}' for k, v in val.items())
        extra = ""
        if r.get("missing"):
            extra += "   [missing: " + ", ".join(r["missing"]) + "]"
        if r.get("nonnumeric"):
            extra += "   [not a number: " + ", ".join(r["nonnumeric"]) + "]"
        return "{" + body + "}" + extra
    err = r.get("error") or r.get("status") or "error"
    line = f" (solve line {r['line']})" if r.get("line") else ""
    return f"ERROR {err}{line}"


def report(ev: dict, n: int, left: int, has_sibs: bool, ended_valid: bool = False) -> str:
    """The fixed-format report for submission n (left = revisions still allowed after it)."""
    L = [f"SOLVE REPORT -- submission {n} of at most {MAX_SUBMISSIONS} ({left} revision(s) left); each dataset's solve got "
         f"its own data_dir and task",
         f"YOUR dataset: {_show(ev['target'])}"]
    for r in ev["siblings"]:
        L.append(f"{r['label']}: {_show(r)}")
    f = ev["flags"]

    def who(xs):
        return ", ".join("YOUR dataset" if x == "target" else x for x in xs) or "none"
    parts = [f"exception: {who(f['exception'])}", f"missing keys: {who(f['missing'])}",
             f"non-numeric: {who(f['nonnumeric'])}"]
    if has_sibs:
        parts += [f"constant across siblings: {', '.join(f['constant']) or 'none'}",
                  f"identical to yours: {', '.join(f['identical']) or 'none'}"]
    L.append("FLAGS: " + " | ".join(parts))
    if ev["valid"]:
        L.append("STATUS: VALID -- your answer is now solve's output on YOUR dataset (above).")
        L.append("NEXT: " + ("if a flag shows a problem with your method, fix solve and submit it again; otherwise reply "
                             "with your final JSON answer." if left > 0 else "no revisions left -- reply with your final "
                                                                             "JSON answer."))
    else:
        L.append(f"STATUS: NOT VALID on YOUR dataset ({ev['why']}).")
        L.append("NEXT: " + ("fix solve and submit it again." if left > 0 else
                             "no revisions left -- reply with your final JSON answer."))
    return "\n".join(L)


# ============================================================================================================ v5.1 (a) look first
_IN_ACCESS, _IN_Q_OVERFLOW, _IN_IGNORED, _IN_ISDIR = 0x00000001, 0x00004000, 0x00008000, 0x40000000
_INO_LOCK = threading.Lock()
_INO = None                                   # the process's one inotify instance (created on first use)


class _Inotify:
    """ONE inotify instance per process, a watch per episode data directory, each event dispatched to the episode that owns
    the watch (every method holds the lock). Only IN_ACCESS (a successful read of >= 1 byte) is watched."""

    def __init__(self):
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            e = ctypes.get_errno()
            raise OSError(e, f"inotify_init1: {os.strerror(e)}")
        self.libc, self.ctypes, self.fd, self.lock, self.owners = libc, ctypes, fd, threading.Lock(), {}

    def add(self, path: Path, owner, rel: str) -> int:
        with self.lock:
            wd = self.libc.inotify_add_watch(self.fd, os.fsencode(str(path)), _IN_ACCESS)
            if wd < 0:
                e = self.ctypes.get_errno()
                raise OSError(e, f"inotify_add_watch {path}: {os.strerror(e)}")
            if wd in self.owners:
                raise RuntimeError(f"inotify watch {wd} ({path}) is already owned by another episode")
            self.owners[wd] = (owner, rel)
            return wd

    def rm(self, wds) -> None:
        with self.lock:
            for wd in wds:
                if self.owners.pop(wd, None) is not None:
                    self.libc.inotify_rm_watch(self.fd, wd)

    def pump(self) -> None:
        """Read every queued event (non-blocking) and dispatch it: a read of an ORIGINAL file of an ARMED episode marks
        that episode's hit; an overflow marks every armed episode (its cell is then judged by the static rule)."""
        with self.lock:
            while True:
                try:
                    buf = os.read(self.fd, 1 << 16)
                except BlockingIOError:
                    break
                except InterruptedError:
                    continue
                if not buf:
                    break
                off = 0
                while off + 16 <= len(buf):
                    wd, mask, _cookie, n = struct.unpack_from("iIII", buf, off)
                    name = os.fsdecode(buf[off + 16: off + 16 + n].split(b"\0", 1)[0])
                    off += 16 + n
                    if mask & _IN_Q_OVERFLOW:
                        for w, _ in self.owners.values():
                            if w.armed:
                                w.overflow = True
                        continue
                    if mask & _IN_IGNORED:                       # the watched directory is gone (sandbox cleanup)
                        self.owners.pop(wd, None)
                        continue
                    own = self.owners.get(wd)
                    if own is None or mask & _IN_ISDIR or not mask & _IN_ACCESS:
                        continue
                    w, rel = own
                    f = f"{rel}/{name}" if rel else name
                    if w.armed and w.hit is None and f in w.originals:
                        w.hit = f


def _inotify() -> _Inotify:
    global _INO
    with _INO_LOCK:
        if _INO is None:
            _INO = _Inotify()
        return _INO


def _data_files(root: Path) -> set:
    """Relative (posix) paths of the regular files under root."""
    out = set()
    for d, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                out.add(os.path.relpath(p, root).replace(os.sep, "/"))
    return out


def static_look(code: str | None, res, names) -> bool:
    """The fallback rule: the cell ran without error and its code names data_files or one of the data files."""
    if res is None or getattr(res, "timed_out", False) or getattr(res, "returncode", 1) != 0:
        return False
    c = textwrap.dedent(code or "")
    return "data_files" in c or any(n and n in c for n in names)


class LookWatch:
    """Did an executed cell READ a data file of the target? One per look-first episode (module docstring, DETECTION).
    arm() before the cell, judge(code, result) after it, close() once looked / at the episode's end. Never raises."""

    def __init__(self, sandbox):
        self.mode, self.error, self.vacuous = "static", None, False
        self.armed, self.hit, self.overflow, self.overflows = False, None, False, 0
        self.wds, self.ino, self.originals, self.names = [], None, set(), []
        try:
            wd = getattr(sandbox, "workdir", None)
            root = (Path(wd) / "data_files") if wd else None
            if root is not None and root.is_dir():
                self.originals = _data_files(root)
            else:
                root = None
                dd = getattr(sandbox, "data_dir", None)
                self.originals = _data_files(Path(dd)) if dd and Path(dd).is_dir() else set()
            self.names = sorted({os.path.basename(f) for f in self.originals})
            if not self.originals:
                self.vacuous = True
                return
            if root is None:
                self.error = "the sandbox has no data_files/ copy of its own"
                return
            if INOTIFY_OFF:
                self.error = "inotify disabled (program_first.INOTIFY_OFF)"
                return
            ino = _inotify()
            try:
                for rel in sorted({os.path.dirname(f) for f in self.originals}):
                    self.wds.append(ino.add(root / rel if rel else root, self, rel))
            except Exception:
                ino.rm(self.wds)
                self.wds = []
                raise
            self.ino, self.mode = ino, "inotify"
        except Exception as exc:                                          # noqa: BLE001
            self.mode, self.ino, self.error = "static", None, f"{type(exc).__name__}: {exc}"[:300]

    def arm(self) -> None:
        if self.ino is None:
            return
        self.ino.pump()                          # events from before the cell are dispatched while NOT armed: dropped
        with self.ino.lock:
            self.hit, self.overflow, self.armed = None, False, True

    def judge(self, code: str | None, res) -> tuple:
        """(looked, how, file) for the cell that just ran."""
        if self.ino is not None:
            self.ino.pump()
            with self.ino.lock:
                self.armed = False
                hit, ovf = self.hit, self.overflow
            if hit is not None:
                return True, "inotify", hit
            if not ovf:
                return False, None, None
            self.overflows += 1                  # the queue overflowed during the cell: the static rule decides it
        ok = static_look(code, res, self.names)
        return ok, ("static" if ok else None), None

    def close(self) -> None:
        if self.ino is not None and self.wds:
            self.ino.rm(self.wds)
        self.wds, self.ino, self.armed = [], None, False


# ============================================================================================================ the episode
class Episode:
    """One rollout's v5 state and log. enabled=False: inert (the loop is then LLMAgent.solve exactly). look_first: v5.1 (a)
    for this episode (None = the module's LOOK_FIRST at creation); off = v5 exactly."""

    def __init__(self, cfg: dict, sandbox, enabled: bool = True, look_first: bool | None = None):
        self.enabled, self.cfg, self.sb = enabled, cfg, sandbox
        self.look = bool(enabled) and bool(LOOK_FIRST if look_first is None else look_first)      # v5.1 (a)
        self.watch = None                                                                          # v5.1 (a)
        self.keys, self.numeric = answer_keys(cfg), numeric_keys(cfg)
        self.sibs, self.sib_err, self.sib_tasks, self.target_task = [], None, [], ""
        self.valid_ev, self.valid_n = None, None
        self.last_offer = None
        self.subs: list = []
        self.log = {"harness": HARNESS_V51 if self.look else HARNESS, "k": 0, "siblings": [], "sibling_error": None,
                    "submissions": 0, "valid": 0,
                    "revisions": 0, "offered": 0, "refused": 0, "refused_submissions": 0, "final": None,
                    "final_submission": None, "end": None, "portability": 0.0, "subs": self.subs}
        if not enabled:
            return
        try:
            self.sibs = siblings_for(getattr(sandbox, "data_dir", None))
        except Exception as exc:                                          # noqa: BLE001
            self.sibs, self.sib_err = [], f"{type(exc).__name__}: {exc}"[:300]
        try:
            self.sib_tasks = [sibling_task(p) for p in self.sibs]
        except Exception as exc:                                          # noqa: BLE001
            self.sibs, self.sib_tasks, self.sib_err = [], [], f"sibling task: {type(exc).__name__}: {exc}"[:300]
        self.target_task = base_task(str(cfg.get("task") or ""))
        self.log.update(k=len(self.sibs), siblings=[p.name for p in self.sibs], sibling_error=self.sib_err)
        if self.look:                                                     # v5.1 (a): the episode's data watch
            self.watch = LookWatch(sandbox)
            self.log.update(look_first=True, looked=self.watch.vacuous, look_turn=None,
                            look_how="vacuous" if self.watch.vacuous else None, look_file=None,
                            look_detector=self.watch.mode, look_error=self.watch.error, look_refusals=0, look_overflow=0)
        try:
            sandbox._v5_log = self.log               # -> V2Sandbox.stats -> rollout["v2"]["v5"]
        except Exception:                            # noqa: BLE001
            pass
        _TLS.log = self.log                          # -> runner.run_one (externals)

    def task(self, task: str) -> str:
        return with_note(task, len(self.sibs)) if self.enabled else task

    def has_valid(self) -> bool:
        return self.valid_ev is not None

    def cap_reached(self) -> bool:
        return len(self.subs) >= MAX_SUBMISSIONS

    def offer(self, ans) -> None:
        self.last_offer = ans
        self.log["offered"] += 1

    def refuse(self) -> None:
        self.log["refused"] += 1

    # ---- v5.1 (a) look first: every method below is inert unless self.look
    def look_blocked(self) -> bool:
        """A submission now would be refused: look-first is on and no executed cell has read the data yet."""
        return self.look and not self.log.get("looked")

    def refuse_look(self, forcing: bool) -> str:
        self.log["look_refusals"] += 1
        return LOOK_REFUSAL_FORCING if forcing else LOOK_REFUSAL

    def before_cell(self) -> None:
        if self.look_blocked() and self.watch is not None:
            try:
                self.watch.arm()
            except Exception as exc:                                      # noqa: BLE001
                self.log["look_error"] = f"arm: {type(exc).__name__}: {exc}"[:300]

    def after_cell(self, code: str, res, turn: int) -> None:
        if not self.look_blocked() or self.watch is None:
            return
        try:
            looked, how, f = self.watch.judge(code, res)
        except Exception as exc:                                          # noqa: BLE001
            self.log["look_error"] = f"judge: {type(exc).__name__}: {exc}"[:300]
            looked, how, f = static_look(code, res, self.watch.names), "static", None
        self.log["look_overflow"] = self.watch.overflows
        if looked:
            self.log.update(looked=True, look_turn=turn, look_how=how, look_file=f)
            self.watch.close()

    def _close_watch(self) -> None:
        if self.watch is not None:
            self.watch.close()

    def submit(self, code: str, turn: int, phase: str) -> tuple[str | None, bool]:
        """Run one submission (or refuse it past the cap). (reply text, episode done)."""
        if self.cap_reached():
            self.log["refused_submissions"] += 1
            return NO_SUBMISSIONS_LEFT, False
        t0 = time.time()
        timeout = int(getattr(self.sb, "timeout", 180) or 180)
        jobs = [("target", None, None, runner_cell(code, self.target_task))] + [
            (f"sibling_{i}", s.name, s, runner_cell(code, t)) for i, (s, t) in enumerate(zip(self.sibs, self.sib_tasks), 1)]
        with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:        # worker threads: decision 9
            runs = list(ex.map(lambda j: run_once(j[3], self.sb, j[2], timeout), jobs))
        try:
            self.sb.attempts += 1                                         # decision 6
        except Exception:                                                 # noqa: BLE001
            pass
        ev = evaluate(runs[0], [(j[0], j[1], r) for j, r in zip(jobs[1:], runs[1:])], self.keys, self.numeric)
        n = len(self.subs) + 1
        rec = {"n": n, "turn": turn, "phase": phase, "valid": ev["valid"], "why": ev["why"],
               "code": textwrap.dedent(code), "code_sha": hashlib.sha256(textwrap.dedent(code).encode()).hexdigest()[:16],
               "target": ev["target"], "siblings": ev["siblings"], "flags": ev["flags"],
               "portability": ev["portability"], "seconds": round(time.time() - t0, 2)}
        self.subs.append(rec)
        self.log["submissions"] = n
        self.log["revisions"] = n - 1
        if ev["valid"]:
            self.valid_ev, self.valid_n = ev, n
            self.log["valid"] += 1
        left = MAX_SUBMISSIONS - n
        done = self.has_valid() and (left == 0 or (ev["valid"] and phase in ("forcing", "final")))
        return report(ev, n, left, bool(self.sibs)), done

    def finish(self, end: str) -> dict:
        """The final answer from the last valid submission."""
        ev = self.valid_ev
        self._close_watch()                                               # v5.1 (a)
        self.log.update(final="solve", final_submission=self.valid_n, end=end, portability=ev["portability"])
        return {"answer": dict(ev["target"]["value"]),
                "reasoning": f"{self.log['harness']}: solve(data_dir, task) of submission {self.valid_n} on the target"}

    def end(self, end: str):
        """End of an episode by the limits: the last valid submission, else the last offered answer, else None."""
        if not self.enabled:
            return None
        if self.has_valid():
            return self.finish(end)
        self._close_watch()                                               # v5.1 (a)
        self.log.update(end=end, portability=0.0)
        if self.last_offer is not None:
            self.log["final"] = "soft_fallback"
            return self.last_offer
        self.log["final"] = None
        return None

    def fallback_now(self, ans, end: str):
        """A JSON answer that ends the episode without a valid submission (decision 1): it is the last offered one."""
        self._close_watch()                                               # v5.1 (a)
        self.log.update(final="soft_fallback", end=end, portability=0.0)
        return ans


# ============================================================================================================ the loop
def make_solve(A, enabled: bool = True):
    """LLMAgent.solve (harness/agents.py, sha-pinned) with the v5 protocol. A = the loaded agents module: every name the
    original reads from its module globals is read from A at call time, as the original does. enabled=False: tests only
    (every deviation point is inert -- the loop is then LLMAgent.solve)."""

    def solve(self, cfg: dict, sandbox) -> tuple:
        v5 = Episode(cfg, sandbox, enabled)                                             # V5
        task = A.build_prompt(v5.task(cfg["task"]), sandbox.listing(), self.prompt_style)   # V5 (was: cfg["task"])
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
            if forcing and v5.enabled and v5.has_valid():                              # V5 decision 3
                return v5.finish("forcing_start"), transcript                           # V5
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

            if v5.enabled:                                                              # V5: a submission first
                sub_code = A.extract_code(reply)                                        # V5
                if is_submission(sub_code):                                             # V5
                    beside = A.extract_answer(reply)                                    # V5
                    if beside is not None:                                              # V5
                        v5.offer(beside)                                                # V5
                    if v5.look_blocked():                                               # V51 (a): not run before a look
                        fb = v5.refuse_look(forcing)                                    # V51
                        messages.append({"role": "user", "content": fb})                # V51
                        transcript.append({"turn": turn, "role": "tool", "content": fb, "v5": "look_first"})   # V51
                        stalled = 0                                                     # V51
                        continue                                                        # V51 (the turn is used)
                    fb, done = v5.submit(sub_code, turn, "forcing" if forcing else "turn")   # V5
                    if done:                                                            # V5 decisions 3 / 4
                        return v5.finish("cap" if v5.cap_reached() else "forcing_submission"), transcript   # V5
                    messages.append({"role": "user", "content": fb})                    # V5
                    transcript.append({"turn": turn, "role": "tool", "content": fb, "v5": "report"})   # V5
                    stalled = 0                                                         # V5
                    continue                                                            # V5

            ans = A.extract_answer(reply)
            if ans is not None:
                if not v5.enabled:                                                      # V5 (tests: the original)
                    return ans, transcript
                v5.offer(ans)                                                           # V5
                if v5.has_valid():                                                      # V5 decision 2
                    return v5.finish("answer"), transcript                              # V5
                if forcing or v5.cap_reached():                                         # V5 decision 1 / 4
                    return v5.fallback_now(ans, "answer_forcing" if forcing else "answer_after_cap"), transcript  # V5
                v5.refuse()                                                             # V5 decision 1
                messages.append({"role": "user", "content": NEED_SUBMIT})               # V5
                transcript.append({"turn": turn, "role": "tool", "content": NEED_SUBMIT, "v5": "need_submit"})  # V5
                stalled = 0                                                             # V5
                continue                                                                # V5
            if forcing:
                continue

            code = A.extract_code(reply)
            if code:
                stalled = 0
                v5.before_cell()                                                        # V51 (a) (inert unless look-first)
                res = sandbox.run(code)
                v5.after_cell(code, res, turn)                                          # V51 (a)
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
                return v5.end("stall"), transcript                                      # V5 (was: None)

            messages.append({"role": "user", "content": nudge})
            transcript.append({"turn": turn, "role": "tool", "content": nudge})

        if v5.enabled and v5.has_valid():                                               # V5 decision 5
            return v5.finish("limit"), transcript                                       # V5
        try:
            messages = self._trim(messages + [
                {"role": "user", "content": self._force_msg(0)}])
            reply, _ = self._chat(messages)
            transcript.append({"turn": self.max_turns, "role": "assistant",
                               "content": reply})
            if v5.enabled:                                                              # V5
                sub_code = A.extract_code(reply)                                        # V5
                if is_submission(sub_code):                                             # V5
                    beside = A.extract_answer(reply)                                    # V5
                    if beside is not None:                                              # V5
                        v5.offer(beside)                                                # V5
                    if v5.look_blocked():                                               # V51 (a): refused, logged; no
                        v5.refuse_look(True)                                            # V51      further turn
                    else:                                                               # V51
                        fb, _done = v5.submit(sub_code, self.max_turns, "final")        # V5 (logged; no further turn)
                        if v5.has_valid():                                              # V5
                            return v5.finish("final_call"), transcript                  # V5
                else:                                                                   # V5
                    ans = A.extract_answer(reply)                                       # V5
                    if ans is not None:                                                 # V5
                        v5.offer(ans)                                                   # V5
                        return v5.fallback_now(ans, "final_call"), transcript           # V5
            else:
                ans = A.extract_answer(reply)
                if ans is not None:
                    return ans, transcript
        except Exception as exc:                                  # noqa: BLE001
            transcript.append({"turn": self.max_turns, "role": "system",
                               "content": f"final forced call failed: {exc}"})
        return v5.end("limit"), transcript                                              # V5 (was: None)

    solve._v5 = True
    return solve


# ============================================================================================================ installers
def install_agents(A) -> None:
    """LLMAgent.solve := the v5 loop (idempotent; the original stays reachable as LLMAgent._solve_pre_v5). Refuses a
    LLMAgent.solve that is already another harness's loop (EF / EFS): v5 is v3 + program-first, never on top of a gate."""
    cls = A.LLMAgent
    if getattr(cls.solve, "_v5", False):
        return
    if getattr(cls.solve, "_ef", False) or getattr(cls.solve, "_efs", False):
        raise RuntimeError("LLMAgent.solve is an execution-gate loop (EF / EFS) -- v5 is installed on the v3 harness only")
    cls._solve_pre_v5 = cls.solve
    cls.solve = make_solve(A)


def note_wrap(rollout):
    """A rollout function whose cfg["task"] carries the v5 note (so the stored episode shows what the model read)."""
    if getattr(rollout, "_v5", False):
        return rollout

    def v5_rollout(cfg: dict, data_dir, spec: str, *a, **kw):
        try:
            k = len(siblings_for(data_dir))
        except Exception:                                                 # noqa: BLE001
            k = 0
        return rollout({**cfg, "task": with_note(cfg["task"], k)}, data_dir, spec, *a, **kw)

    for attr in ("harness_v2", "siblings_k"):
        if hasattr(rollout, attr):
            setattr(v5_rollout, attr, getattr(rollout, attr))
    v5_rollout._v5 = True
    return v5_rollout


def install_v2(V) -> None:
    """Provenance, logging and the stored task text in harness/v2.py's namespace (idempotent). Refuses a v2 already
    patched by EF / EFS (their as_dict carries _ef)."""
    HC, SB = V.HarnessConfig, V.V2Sandbox
    if getattr(HC.as_dict, "_v5", False):
        return
    if getattr(HC.as_dict, "_ef", False) or getattr(HC.as_dict, "_efs", False):
        raise RuntimeError("harness/v2.py already carries an execution-gate record (EF / EFS) -- v5 refused")
    orig_as_dict, orig_stats, orig_banner, orig_wrap = HC.as_dict, SB.stats, V.banner, V.wrap_rollout

    def as_dict(self) -> dict:
        d = orig_as_dict(self)
        d[harness_name()] = record()                                      # v5.1: 'program_first_v51' when LOOK_FIRST
        return d

    def stats(self) -> dict:
        s = orig_stats(self)
        log = getattr(self, "_v5_log", None)
        if log is not None:
            s["v5"] = json.loads(json.dumps(log, default=str))
        return s

    def banner(hc) -> str:
        return orig_banner(hc) + (f" + PROGRAM-FIRST v5 (submit solve(data_dir, task); run on the target + {K_SIBLINGS} "
                                  f"layout-matched train siblings; <= {MAX_REVISIONS} revisions; "
                                  "experiments/harness_v3/program_first.py)") + (
            " + v5.1 LOOK FIRST (a solve() is run only after an executed cell has read the data)" if LOOK_FIRST else "")

    def wrap_rollout(one_rollout, hc, mode="eval", tmp_root=None):
        return orig_wrap(note_wrap(one_rollout), hc, mode, tmp_root)

    as_dict._v5 = stats._v5 = banner._v5 = wrap_rollout._v5 = True
    HC.as_dict, SB.stats, V.banner, V.wrap_rollout = as_dict, stats, banner, wrap_rollout


def install_runner(R) -> None:
    """harness/runner.py (externals): run_one := the original, then rec["v5"] = the episode's log, and the problem's JSON
    is rewritten atomically with it (idempotent)."""
    if getattr(R.run_one, "_v5", False):
        return
    orig = R.run_one

    def run_one(pdir, agent_spec, outdir, timeout):
        _TLS.log = None
        rec = orig(pdir, agent_spec, outdir, timeout)
        log = getattr(_TLS, "log", None)
        _TLS.log = None
        rec["v5"] = json.loads(json.dumps(log, default=str)) if log is not None else None
        p = Path(outdir) / f"{rec['id']}.json"
        tmp = p.with_name(f".{p.name}.v5.tmp")
        tmp.write_text(json.dumps(rec, indent=2, default=str))
        os.replace(tmp, p)
        return rec

    run_one._v5 = True
    run_one._orig = orig
    R.run_one = run_one
