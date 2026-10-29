"""BioDSBench grading (copied into benchmarks/biodsbench/reference_grader.py by scripts/fetch_biodsbench.py).

The benchmark's OWN protocol: "The generated code solutions must pass all the testing cases to be considered right"
(Wang et al., arXiv 2410.21591). A task = prefix code (`code_histories`, may be empty) + the model's solution + the
task's `test_cases` (plain `assert` statements over the variables the solution must define). This grader runs exactly
that, in a FRESH Python process, in a scratch directory whose data_files/ holds the study's tables:

    exec(prefix) ; exec(solution) ; exec(test_cases)      -- one shared namespace, in that order
    passed  <=>  all three finish without an exception AND the driver reaches its end (a random token printed after
                 the tests; a solution that calls sys.exit()/os._exit() before the tests ran therefore fails)

WHERE THE SOLUTION COMES FROM (the design choice; see fetch_biodsbench.py):
  * the agent writes its final program to solution.py in its working directory and answers {"<key>": "solution.py"};
    harness/runner.py attaches that file to the submission because this module declares SUBMISSION_FILES
    (an opt-in runner path added for this benchmark; no other grader declares it);
  * or the answer value is the solution source itself (inline code) -- how the oracle submits the reference, and
    how a harness without the runner opt-in can still use this grader.
Nothing from the agent's notebook state is reused: exploration cells that happen to leave the right variables behind
do not count, exactly as a single-shot generation would not get them.

Deviations from the upstream setup, all path/environment only (the tests are never edited except for paths):
  * `/workdir` and `./workdir` are rewritten to `data_files` in prefix, tests and task text (the sandbox layout);
  * the interpreter is the harness sandbox's ($GBP_SANDBOX_PYTHON, else this one), MPLBACKEND=Agg, one BLAS thread,
    RLIMIT_AS $BIODS_MEM_GB (default 8), wall clock $BIODS_TEST_TIMEOUT seconds (default 600).
"""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# runner.py attaches these from the agent's final working directory (opt-in; see harness/runner.py).
SUBMISSION_FILES = ("solution.py",)
MAX_TAIL = 1500

DRIVER = r'''
import os, sys, traceback
{limits}
_ns = {{"__name__": "__main__"}}
for _label in ("prefix", "solution", "tests"):
    _src = open(_label + ".py").read()
    try:
        exec(compile(_src, "<" + _label + ">", "exec"), _ns)
    except SystemExit as _e:
        print("__BIODS_FAIL__ %s called sys.exit(%r)" % (_label, _e.code), flush=True)
        os._exit(3)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        print("__BIODS_FAIL__ %s raised" % _label, flush=True)
        os._exit(2)
sys.stdout.flush()
print({token!r}, flush=True)
'''


def _limits() -> str:
    mem = int(float(os.environ.get("BIODS_MEM_GB", "8")) * 1024 ** 3)
    return ("import resource as _r\n"
            f"_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))\n"
            "del _r")


def _python() -> str:
    return os.environ.get("GBP_SANDBOX_PYTHON") or sys.executable


def _stage_data(study: str, dest: Path) -> None:
    """data_files/ = hard links to the study's read-only tables (copy if linking fails)."""
    src = DATA / study
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            try:
                os.link(f, dest / f.name)
            except OSError:
                shutil.copy2(f, dest / f.name)


def run_tests(prefix: str, solution: str, tests: str, study: str, timeout: float | None = None) -> dict:
    timeout = float(timeout if timeout is not None else os.environ.get("BIODS_TEST_TIMEOUT", "600"))
    work = Path(tempfile.mkdtemp(prefix="biods_"))
    token = "__BIODS_TESTS_COMPLETED__" + secrets.token_hex(16)
    try:
        _stage_data(study, work / "data_files")
        (work / "prefix.py").write_text(prefix or "")
        (work / "solution.py").write_text(solution or "")
        (work / "tests.py").write_text(tests or "")
        (work / "_driver.py").write_text(DRIVER.format(limits=_limits(), token=token))
        env = dict(os.environ)
        env.update({"MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore", "PYTHONNOUSERSITE": "1",
                    "PIP_NO_INDEX": "1", "UV_NO_INDEX": "1"})
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[v] = "1"
        try:
            p = subprocess.run([_python(), "_driver.py"], cwd=work, capture_output=True, text=True,
                               timeout=timeout, env=env)
            out, err, rc, timed_out = p.stdout, p.stderr, p.returncode, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            rc, timed_out = -9, True
        completed = token in out
        stage = "timeout" if timed_out else next(
            (ln.split()[1] for ln in out.splitlines() if ln.startswith("__BIODS_FAIL__")), None)
        return {"passed": rc == 0 and completed and not timed_out, "returncode": rc, "failed_stage": stage,
                "stderr_tail": err[-MAX_TAIL:], "stdout_tail": out.replace(token, "<token>")[-MAX_TAIL:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def solution_source(value, files: dict) -> tuple[str | None, str]:
    """(source, how) for one answer value. A filename that was attached wins; any other string is inline code."""
    if value is None:
        return None, "no_answer"
    if not isinstance(value, str):
        return None, "answer_not_a_string"
    v = value.strip()
    if v in files:
        return files[v], f"file:{v}"
    if v.endswith(".py") and "\n" not in v:
        return None, f"file_not_attached:{v}"
    return value, "inline"


def evaluate(config: dict, submission: dict) -> dict:
    """Harness entry point -- same contract as the other vendored graders."""
    grader = config.get("grader") or {}
    if grader.get("type") != "biodsbench_tests":
        raise ValueError(f"unsupported grader type: {grader.get('type')}")
    gcfg = grader.get("config") or {}
    obj = submission or {}
    files = obj.get("_files") or {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    details, correct = [], 0
    for k, spec in (gcfg.get("keys") or {}).items():
        src, how = solution_source(obj.get(k), files)
        if src is None:
            res = {"passed": False, "failed_stage": how}
        else:
            res = run_tests(spec["prefix"], src, spec["tests"], spec["study"])
        correct += bool(res["passed"])
        details.append({"key": k, "correct": bool(res["passed"]), "solution_from": how, **res})
    n = max(len(details), 1)
    return {"eval_id": config.get("id"), "grader_type": "biodsbench_tests",
            "score": correct / n, "passed": correct == n, "correct": correct, "total": n, "details": details}
