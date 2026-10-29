"""Sandboxed Python execution for GeneBench-Pro style analysis tasks.

Each problem ships ~130 KB of gzipped TSVs, so we do not need containers or a
cluster job per task. What we do need is isolation between attempts: the agent
writes files, mutates state, and occasionally writes an infinite loop.

Guarantees:
  * fresh working dir per episode, data_files/ copied in (never the original)
  * wall-clock timeout and RLIMIT_AS / RLIMIT_CPU per exec
  * stdout+stderr captured and truncated so one runaway print cannot blow up
    the LLM context
  * state persists across cells within an episode (like a notebook), because
    multi-stage analysis is the thing being measured

Not a security boundary. It runs untrusted-ish model code as your user; use a
real container if you ever run adversarial code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

MAX_OUTPUT_CHARS = 8000
SENTINEL = "__GBP_CELL_START__"


def _after_sentinel(text: str) -> str:
    """Keep only output produced after the last cell's sentinel."""
    idx = text.rfind(SENTINEL)
    return text if idx < 0 else text[idx + len(SENTINEL) :].lstrip("\n")


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    def as_feedback(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(f"STDOUT:\n{self.stdout.rstrip()}")
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr.rstrip()}")
        if self.timed_out:
            parts.append("ERROR: execution timed out")
        if not parts:
            parts.append("(no output)")
        return "\n".join(parts)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    return f"{head}\n... [{len(text) - limit} chars truncated] ...\n{tail}"


def _user_threads() -> int:
    """Threads already owned by this uid.

    RLIMIT_NPROC is a PER-USER cap, not a per-process one, so a fixed value is
    the wrong tool for per-episode isolation: it is compared against everything
    the account is already running. A flat 256 works on an idle login node and
    fails closed on a busy one -- the child is born over the limit, every
    pthread_create returns EAGAIN, and `import numpy` dies inside OpenBLAS
    before a single cell runs. Costing an episode nothing but a /proc walk to
    express the limit as HEADROOM instead is worth it: the limit is meant to
    stop the agent's code fork-bombing, which headroom still does.
    """
    uid, n = os.getuid(), 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != uid:
                continue
            with open(f"/proc/{entry}/status") as fh:
                for line in fh:
                    if line.startswith("Threads:"):
                        n += int(line.split()[1])
                        break
        except (OSError, ValueError):
            continue          # process exited mid-walk; it does not count
    return n


# The sandbox runs cells with sys.executable, so the interpreter it uses is
# whichever environment imported it. That is venv/ under the eval harness and
# venv_verl/ under Ray -- and statsmodels was present in the first and absent
# in the second, so every scRNA training rollout died on the import while the
# identical policy scored 0.298 under evaluation. Zero solves in 336 rollouts,
# and it looked exactly like RL failing to learn.
#
# The environments must agree on what a cell can import. This check names the
# problem at construction instead of leaving it to surface as a flat reward.
# Everything a generated instance's reference pipeline or a model's code has
# been observed to import. The list exists because the two interpreters drifted
# apart silently: statsmodels alone cost 336 rollouts scoring zero, and
# sklearn/matplotlib/openpyxl/seaborn/joblib were missing on the same side.
PREFLIGHT = ("numpy", "pandas", "scipy", "statsmodels", "sklearn",
             "matplotlib", "openpyxl", "seaborn", "joblib")


def sandbox_python() -> str:
    """The interpreter cells run under: $GBP_SANDBOX_PYTHON, else this one.

    A trainer that lives in a different environment from the evaluation harness
    (vLLM/torch in venv_vllm, analysis stack in venv) must point cells at the
    harness's interpreter, or training rollouts run a different -- here, a
    scipy-less -- Python from the one every reported number is scored under.
    """
    return os.environ.get("GBP_SANDBOX_PYTHON") or sys.executable


def preflight(strict: bool = False) -> list[str]:
    """Return the PREFLIGHT modules missing from the SANDBOX's interpreter."""
    py = sandbox_python()
    code = ("import importlib.util as u; print(' '.join(m for m in %r "
            "if u.find_spec(m) is None))" % (PREFLIGHT,))
    res = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
    missing = res.stdout.split() if res.returncode == 0 else list(PREFLIGHT)
    if missing and strict:
        raise RuntimeError(
            f"sandbox interpreter {py} is missing {missing}. "
            f"Cells will fail on import and every rollout will score ~0.")
    return missing


@dataclass
class Sandbox:
    """A per-episode scratch dir with notebook-like persistent state.

    Cells accumulate in `self.cells`; each exec replays prior cells before the
    new one. Replay is O(n^2) but these analyses are seconds-long and it keeps
    the executor stateless, which matters when episodes get retried or resumed.
    """

    data_dir: Path
    workdir: Path = field(default=None)  # type: ignore[assignment]
    timeout: int = 300
    # Hard-link large read-only data instead of copying it. Off by default so
    # every existing caller keeps byte-for-byte copy semantics.
    link_data: bool = False
    mem_limit_gb: int = 8
    # Headroom above the user's CURRENT thread count. 512 was sized for the
    # eval harness, which runs a handful of sandboxes at once. RL runs 48
    # concurrent episodes, the user sits at ~10.7k threads, and the base count
    # is a snapshot taken at construction -- so by the time a cell's BLAS pool
    # spawns, other episodes have eaten the headroom. Cells then died with
    # "OpenBLAS blas_thread_init: RLIMIT_NPROC" and "libgomp: Thread creation
    # failed", NON-DETERMINISTICALLY and for reasons unrelated to the model's
    # code, which depressed the training solve rate below the eval's.
    nproc_headroom: int = 2048   # processes/threads this episode may ADD
    # BLAS threads per cell; None -> $GBP_SANDBOX_THREADS -> 1. See run().
    blas_threads: int | None = None
    cells: list[str] = field(default_factory=list)
    attempts: int = 0     # code blocks submitted (the real activity signal)
    failures: int = 0     # of those, how many raised
    python: str = field(default_factory=sandbox_python)

    def __post_init__(self) -> None:
        if self.workdir is None:
            self.workdir = Path(tempfile.mkdtemp(prefix="gbp_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        # Counted once per episode, not per cell: the /proc walk is cheap but
        # not free, and the account's load does not change meaningfully inside
        # one episode.
        self._base_threads = _user_threads()
        dest = self.workdir / "data_files"
        if self.data_dir and Path(self.data_dir).is_dir() and not dest.exists():
            # BixBench instances carry up to 1.1 GB of data files each, against
            # 12 KB for a generated GeneBench-Pro instance -- roughly a
            # 90,000x difference. A synchronous copytree per rollout blocks the
            # asyncio event loop for minutes, which is why three RL runs hung
            # with `running: 1, finished: 7` and why an outer asyncio.wait_for
            # could not rescue them: wait_for cannot interrupt a thread that is
            # not yielding. It would also copy terabytes over a training run.
            #
            # Hard links are instantaneous and share storage. They are only
            # safe because the source tree is made read-only (see
            # scripts/protect_benchmark_data.sh), so a cell that tries to write
            # through one fails loudly instead of silently corrupting the
            # benchmark for every future rollout.
            if self.link_data:
                dest.mkdir(parents=True, exist_ok=True)
                for src in Path(self.data_dir).rglob("*"):
                    rel = src.relative_to(self.data_dir)
                    tgt = dest / rel
                    if src.is_dir():
                        tgt.mkdir(parents=True, exist_ok=True)
                        continue
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(src, tgt)
                    except OSError:
                        shutil.copy2(src, tgt)
            else:
                shutil.copytree(self.data_dir, dest)

    def _limit_preamble(self) -> str:
        """Resource limits, applied by the child to ITSELF.

        These used to be a `preexec_fn`, which subprocess documents as unsafe
        with threads -- and every caller here is threaded (`runner.py
        --workers N`, `rl/sft_traces.py --workers N`). It deadlocks exactly as
        documented: a forked child hangs before exec, and because
        `ThreadPoolExecutor.map` yields IN ORDER, one stuck early job stalls
        the whole run while later jobs keep completing invisibly. Observed as
        800 traces producing zero output at 230% CPU.

        Running them as the first statements of the episode script is
        thread-safe by construction and costs nothing. It does mean the agent's
        own code could raise its limits back -- which changes nothing, because
        this was never a security boundary (see module docstring); the limits
        exist to stop a runaway loop, not an adversary.
        """
        mem = self.mem_limit_gb * 1024**3
        nproc = self._base_threads + self.nproc_headroom
        return (
            "import resource as _r\n"
            f"_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))\n"
            f"_r.setrlimit(_r.RLIMIT_CPU, ({self.timeout}, {self.timeout + 10}))\n"
            "try:\n"
            "    _h = _r.getrlimit(_r.RLIMIT_NPROC)[1]\n"
            f"    _r.setrlimit(_r.RLIMIT_NPROC, ({nproc} if _h == _r.RLIM_INFINITY "
            f"else min({nproc}, _h), _h))\n"
            "except (ValueError, OSError):\n"
            "    pass\n"
            "del _r\n"
        )

    def run(self, code: str) -> ExecResult:
        self.attempts += 1

        # A cell that does not COMPILE cannot be quarantined by the try/except
        # wrapper below: SyntaxError is raised at parse time, so the whole
        # replay file fails and EVERY later cell returns the same error. One
        # `!pip install x` -- which models emit constantly, because this looks
        # like a notebook -- permanently bricks the episode. Observed: 21 cells
        # submitted, 21 failed, all with the same SyntaxError from cell 3.
        #
        # So reject it here, report it like any other failure, and keep it OUT
        # of self.cells so the replay stays clean and the agent can recover.
        # This is C4's lesson extended to compile-time errors.
        try:
            compile(textwrap.dedent(code), "<cell>", "exec")
        except SyntaxError as exc:
            self.failures += 1
            hint = ""
            if any(code.lstrip().startswith(p) for p in ("!", "%", "%%")):
                hint = ("\nNOTE: this is a plain Python process, not IPython -- "
                        "shell escapes (!cmd) and magics (%cmd) are not available. "
                        "Use subprocess or the standard library instead.")
            return ExecResult(
                "", f"SyntaxError: {exc.msg} (line {exc.lineno}){hint}", 1)

        self.cells.append(code)
        # Prior cells are wrapped in try/except so a cell that raised does not
        # abort the replay of everything after it. This matters more than it
        # looks: dropping failed cells (the previous approach) meant NO state
        # survived a failure -- not even the successful lines before the raise
        # -- so a model that hit one error would reload everything from scratch
        # every turn and loop. Wrapping keeps partial state, which is what a
        # real notebook does.
        parts = [
            "try:\n"
            + textwrap.indent(textwrap.dedent(c), "    ")
            + "\nexcept BaseException:\n    pass"
            for c in self.cells[:-1]
        ]
        last = textwrap.dedent(self.cells[-1])
        if os.environ.get("GBP_ECHO_LAST") == "1":
            last = _echo_last_expression(last)
        parts.append(
            f"import sys as _s; print({SENTINEL!r}); print({SENTINEL!r}, file=_s.stderr)\n"
            f"# --- cell {len(self.cells) - 1} ---\n"
            f"{last}"
        )
        script = self._limit_preamble() + "\n" + "\n\n".join(parts)
        path = self.workdir / "_episode.py"
        path.write_text(script)

        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        # BLAS threads per cell. ONE by default, and that default is load
        # bearing: 48 concurrent episodes each opening a full thread pool
        # oversubscribes the box and is what exhausts RLIMIT_NPROC, which is
        # also why generators/emit.py pins to 1.
        #
        # But measured on this box (ROADMAP T3b), a burden cell's SVD +
        # 1500x1500 kinship runs 1.76 s at one thread and 1.33 s at four, and
        # the RL loop holds only 8 episodes in flight against 128 cores at load
        # ~15 -- nothing like the minting case the pin was written for. So it is
        # a knob, defaulting to the old behaviour, opt-in via GBP_SANDBOX_THREADS
        # so that changing this file cannot alter a run already in flight.
        _thr = self.blas_threads or os.environ.get("GBP_SANDBOX_THREADS") or "1"
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[_v] = str(_thr)
        env["PYTHONWARNINGS"] = "ignore"
        # A cell must never be able to modify the environment every other job shares. On 2026-09-16 a rollout that hit
        # an array-shape error tried `pip install numpy`; the index returned an S3 "NoSuchKey" XML page, pip wrote it to
        # site-packages/numpy.py, uninstalled the real numpy and rewrote venv/bin -- taking down every job on 32 GPUs.
        # These make an install inside a cell fail immediately and locally instead: no index to fetch from, nowhere to
        # write, and no user site directory to fall back on.
        env["PIP_NO_INDEX"] = "1"
        env["PIP_NO_BUILD_ISOLATION"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_TARGET"] = str(Path(self.workdir) / ".pip-refused")
        env["PIP_CACHE_DIR"] = str(Path(self.workdir) / ".pip-cache")
        env["PYTHONNOUSERSITE"] = "1"
        env["UV_NO_INDEX"] = "1"

        try:
            proc = subprocess.run(
                [self.python, str(path)],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            if proc.returncode != 0:
                self.failures += 1
            return ExecResult(
                _truncate(_after_sentinel(proc.stdout)),
                _truncate(_after_sentinel(proc.stderr)),
                proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            # A hanging cell must not be replayed forever -- drop this one.
            self.cells.pop()
            self.failures += 1
            return ExecResult(
                _truncate(_after_sentinel(exc.stdout.decode() if exc.stdout else "")),
                _truncate(_after_sentinel(exc.stderr.decode() if exc.stderr else "")),
                -1,
                timed_out=True,
            )

    def listing(self) -> str:
        d = self.workdir / "data_files"
        if not d.is_dir():
            return "(no data files)"
        return "\n".join(
            f"  data_files/{p.name}  ({p.stat().st_size:,} bytes)"
            for p in sorted(d.iterdir())
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


def _echo_last_expression(code: str) -> str:
    """NOTEBOOK SEMANTICS, opt-in (GBP_ECHO_LAST=1): if the cell ends in a bare expression, print its repr.

    Why (2026-09-22). This sandbox runs cells as a SCRIPT, so a final bare expression -- `df.head()`,
    `mean, sd` -- prints nothing and the agent sees "(no output)". Models write notebook-style code constantly.
    On DABench 54.9% of the plain agent's problems had EVERY cell silent; the agent re-ran the same silent code
    and then invented numbers (12345.67 for both a mean and an SD). Any scaffold that forces printing (the
    self-verify loop, a multi-cell worked example that shows the '(no output) -> print()' recovery) removes that
    failure, which confounds every DABench comparison against the plain agent. This flag gives the plain agent
    the behaviour a notebook would, so the baseline can be measured fairly. OFF by default: nothing already
    measured changes. Only the CURRENT cell echoes -- replayed cells are silent, as in Jupyter.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    expr = tree.body[-1].value
    tree.body[-1] = ast.Assign(targets=[ast.Name(id="_gbp_echo", ctx=ast.Store())], value=expr)
    tree.body.append(ast.parse("if _gbp_echo is not None:\n    print(repr(_gbp_echo))").body[0])
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return code
