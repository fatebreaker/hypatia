"""biocoder_reference_grader.py (10-17, H-GEN27) -- grader for the BioCoder TRAINING tasks (MedAgentGym's BioCoder training
split, converted by scripts/biocoder_build.py). MedAgentGym's own check, made offline and file-based:

    the agent's function (solution.py, attached by the harness, or inline code) is inserted into the task's context program
    in place of the line <<insert solution here>>; the program runs in a FRESH Python process; the task is passed iff the
    program exits 0 and its stdout equals the reference program's stdout (trailing whitespace ignored). MedAgentGym compares
    the same printed output by string equality.

Environment: the harness sandbox's interpreter ($GBP_SANDBOX_PYTHON, else this one), a fresh scratch directory as the working
directory (and HOME / TMPDIR), MPLBACKEND=Agg, one BLAS thread, HTTP(S) proxies pointed at a closed port (no web services), RLIMIT_AS
$BIOCODER_MEM_GB (default 8), wall clock $BIOCODER_TIMEOUT seconds (default 120).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PLACEHOLDER = "<<insert solution here>>"
_SLOT = re.compile(r"^" + re.escape(PLACEHOLDER) + r"[ \t]*$", re.M)
SUBMISSION_FILES = ("solution.py",)
MAX_TAIL = 1500


def _python() -> str:
    return os.environ.get("GBP_SANDBOX_PYTHON") or sys.executable


def program(context: str, source: str) -> str:
    assert len(_SLOT.findall(context)) == 1, "context must hold exactly one top-level insertion line"
    return _SLOT.sub(lambda _m: source.rstrip("\n"), context, count=1)


def run_program(context: str, source: str, timeout: float | None = None, pad: int = 0, workroot: str | None = None) -> dict:
    """pad / workroot are for the BUILD's determinism check only (scripts/biocoder_build.py): pad allocates objects of many
    sizes and enlarges the environment before the program runs (shifts the heap, so output that depends on object addresses
    -- e.g. iterating a set of objects -- changes), workroot puts the scratch directory on another filesystem (so output that
    depends on directory-listing order changes). Grading uses neither."""
    timeout = float(timeout if timeout is not None else os.environ.get("BIOCODER_TIMEOUT", "120"))
    mem = int(float(os.environ.get("BIOCODER_MEM_GB", "8")) * 1024 ** 3)
    work = Path(tempfile.mkdtemp(prefix="biocoder_", dir=workroot))
    try:
        (work / "program.py").write_text(program(context, source))
        (work / "_driver.py").write_text(
            "import resource, runpy\n"
            f"resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))\n"
            + ((f"import random as _r\n_r.seed({int(pad)})\nclass _P:\n    pass\n_pad = []\n"
                f"for _i in range({int(pad) % 3000 + 1}):\n"
                "    _pad += [_P(), 'x' * _r.randint(1, 300), {_i: _i}, [0] * _r.randint(0, 40), (_i, _i + 1)]\n"
                "del _r, _i\n") if pad else "")
            + "runpy.run_path('program.py', run_name='__main__')\n")
        env = dict(os.environ)
        env.update({"MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
                    "PIP_NO_INDEX": "1", "UV_NO_INDEX": "1", "http_proxy": "http://127.0.0.1:9",
                    "https_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"})
        env.update({"HOME": str(work), "TMPDIR": str(work), "TEMP": str(work), "TMP": str(work)})   # stray writes stay in the scratch dir
        if pad:
            env["BIOCODER_PAD"] = "x" * (int(pad) % 4096)       # a different environment size shifts the interpreter's allocations
        env.pop("no_proxy", None)
        env.pop("NO_PROXY", None)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[v] = "1"
        try:
            p = subprocess.run([_python(), "_driver.py"], cwd=work, capture_output=True, text=True, timeout=timeout,
                               env=env, errors="replace")
            return {"returncode": p.returncode, "stdout": p.stdout, "stderr_tail": p.stderr[-MAX_TAIL:], "timeout": False}
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return {"returncode": -9, "stdout": out, "stderr_tail": "", "timeout": True}
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
    if grader.get("type") != "biocoder_stdout":
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
            res, ok = {"failed_stage": how}, False
        else:
            r = run_program(spec["context"], src)
            ok = r["returncode"] == 0 and not r["timeout"] and r["stdout"].rstrip() == spec["expected_stdout"].rstrip()
            res = {"returncode": r["returncode"], "timeout": r["timeout"], "stdout_tail": r["stdout"][-MAX_TAIL:],
                   "stderr_tail": r["stderr_tail"]}
        correct += bool(ok)
        details.append({"key": k, "correct": bool(ok), "solution_from": how, **res})
    n = max(len(details), 1)
    return {"eval_id": config.get("id"), "grader_type": "biocoder_stdout",
            "score": correct / n, "passed": correct == n, "correct": correct, "total": n, "details": details}
