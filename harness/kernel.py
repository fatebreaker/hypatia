"""KERNEL (harness v2, switch GBP_KERNEL=1): a persistent-process sandbox. docs/HARNESS_V2.md.

harness/sandbox.py REPLAYS every earlier cell on each exec (a fresh interpreter per turn). That keeps the executor
stateless, but one slow cell is re-run on every later turn: the 30-60 min RL stragglers (docs/RL_PLAN.md s.9) were
episodes whose early fit took minutes and was then replayed 10-20 times. KernelSandbox keeps state in ONE Python
process per episode (harness/sandbox_lib/_gbp_kernel_worker.py, run by the SANDBOX interpreter) and runs each cell
once. Everything else is the replay sandbox's contract, kept on purpose:

  * the same data copy, working dir, env (MPLBACKEND, BLAS threads, pip guards), RLIMIT_AS / RLIMIT_NPROC headroom;
  * the same SyntaxError pre-check (a non-compiling cell is rejected and never reaches the kernel), the same
    notebook echo of a final bare expression under GBP_ECHO_LAST=1 (sandbox._echo_last_expression, reused);
  * the same wall-clock `timeout` and CPU budget PER CELL (in the replay sandbox they covered the whole replay);
  * the same output truncation (sandbox._truncate on each stream), ExecResult, attempts/failures counters, cells list;
  * cleanup() kills the kernel's whole process group, then removes the working dir.

Failure handling (the model is TOLD, never left to guess):
  * wall timeout -> SIGINT (KeyboardInterrupt in the cell); state is KEPT, as in a notebook; "execution timed out";
  * the cell ignores SIGINT for KILL_GRACE s, or the process dies (segfault, os._exit, OOM kill) -> the process group
    is killed, the result carries KERNEL_LOST_NOTE, and the next cell starts a fresh kernel (restarts counted);
  * the kernel cannot start at all (MAX_START_FAILURES times) -> this episode falls back to the replay sandbox, and
    the model is told once that state now persists by replay.
"""
from __future__ import annotations

import json
import os
import secrets
import select
import signal
import struct
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

from sandbox import ExecResult, Sandbox, _echo_last_expression, _truncate

LIB_DIR = Path(__file__).resolve().parent / "sandbox_lib"
KILL_GRACE = 10.0            # seconds a SIGINT-ed cell gets to return before the kernel is killed
MARKER_GRACE = 5.0           # seconds to wait for the end-of-cell markers after the worker replied
START_TIMEOUT = 120.0        # interpreter start under a loaded node
MAX_START_FAILURES = 3
KEEP = 1 << 17               # bytes kept from the head and from the tail of each stream (bounded collection)

KERNEL_LOST_NOTE = ("[kernel] The Python process running your code {why}. It has been restarted: every variable, "
                    "import and loaded dataset from earlier cells is GONE. Re-run what you need before continuing.")
FALLBACK_NOTE = ("[kernel] The persistent Python process could not be started; from now on each cell re-runs your "
                 "earlier cells before running, so state still persists.")


def _readable(fds: list, timeout: float) -> list:
    """The fds of `fds` that are readable (or hung up) within `timeout` seconds. poll(), not select(): a trainer process
    holds hundreds of fds, and select() refuses any fd >= 1024 (ValueError), which would fail every cell."""
    p = select.poll()
    for fd in fds:
        p.register(fd, select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR)
    return [fd for fd, _ in p.poll(max(0, int(timeout * 1000)))]


class _Stream:
    """Bounded collector for one output fd: the first and last KEEP bytes, the total count, and the end marker."""

    def __init__(self, marker: bytes, carry: bytes = b""):
        self.marker, self.head, self.tail, self.total = marker, bytearray(), bytearray(), 0
        self.pending, self.done, self.carry = bytearray(), False, b""
        if carry:
            self.data(carry)

    def _feed(self, b: bytes) -> None:
        self.total += len(b)
        room = KEEP - len(self.head)
        if room > 0:
            self.head += b[:room]
            b = b[room:]
        if b:
            self.tail += b
            if len(self.tail) > KEEP:
                del self.tail[: len(self.tail) - KEEP]

    def data(self, chunk: bytes) -> None:
        if self.done:                        # output after the marker belongs to the next cell
            self.carry += chunk
            return
        self.pending += chunk
        i = self.pending.find(self.marker)
        if i >= 0:
            self._feed(bytes(self.pending[:i]))
            self.carry += bytes(self.pending[i + len(self.marker):])
            self.pending = bytearray()
            self.done = True
            return
        keep = len(self.marker) - 1          # a marker may be split across reads: hold back its possible prefix
        if len(self.pending) > keep:
            self._feed(bytes(self.pending[:-keep]))
            del self.pending[:-keep]

    def text(self) -> str:
        if self.pending:
            self._feed(bytes(self.pending))
            self.pending = bytearray()
        if self.total <= 2 * KEEP:
            raw = bytes(self.head) + bytes(self.tail)
            return _truncate(raw.decode("utf-8", errors="replace"))
        h = bytes(self.head).decode("utf-8", errors="replace")
        t = bytes(self.tail).decode("utf-8", errors="replace")
        # the same shape sandbox._truncate produces, with the true size of what was dropped
        from sandbox import MAX_OUTPUT_CHARS as M
        return f"{h[: M // 2]}\n... [{self.total - M} chars truncated] ...\n{t[-M // 2:]}"


@dataclass
class KernelSandbox(Sandbox):
    """Sandbox whose cells run in one persistent kernel process. Constructor-compatible with Sandbox."""

    kernel_restarts: int = 0          # kernels started after the first (timeouts that needed a kill, crashes)
    kernel_timeouts: int = 0
    kernel_fallback: bool = False     # this episode fell back to the replay sandbox
    cell_seconds: list = field(default_factory=list)
    # extra statements run once in each new kernel (APIDOCS: import help_api), and whether cells may import
    # harness/sandbox_lib (CHECKS / APIDOCS); set by harness/v2.py
    kernel_preamble: list = field(default_factory=list)
    expose_lib: bool = False
    apidocs: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self._proc = None
        self._ctl_w = self._ctl_r = None
        self._starts = 0
        self._start_failures = 0
        self._lost_pending = False
        self._told_lost = False
        self._carry = {1: b"", 2: b""}
        self._rbuf = b""

    # ------------------------------------------------------------------ process management
    def _env(self) -> dict:
        """The replay sandbox's per-cell environment, verbatim (harness/sandbox.py Sandbox.run)."""
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        _thr = self.blas_threads or os.environ.get("GBP_SANDBOX_THREADS") or "1"
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[_v] = str(_thr)
        env["PYTHONWARNINGS"] = "ignore"
        env["PIP_NO_INDEX"] = "1"
        env["PIP_NO_BUILD_ISOLATION"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_TARGET"] = str(Path(self.workdir) / ".pip-refused")
        env["PIP_CACHE_DIR"] = str(Path(self.workdir) / ".pip-cache")
        env["PYTHONNOUSERSITE"] = "1"
        env["UV_NO_INDEX"] = "1"
        return env

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self) -> bool:
        """Start a kernel; True when it reported ready."""
        p2c_r, p2c_w = os.pipe()
        c2p_r, c2p_w = os.pipe()
        cfg = {"lib_dir": str(LIB_DIR), "expose_lib": bool(self.expose_lib), "apidocs": bool(self.apidocs),
               "preamble": list(self.kernel_preamble), "mem_bytes": self.mem_limit_gb * 1024 ** 3,
               "nproc": self._base_threads + self.nproc_headroom}
        boot = (f"import sys; sys.path.insert(0, {str(LIB_DIR)!r}); import _gbp_kernel_worker as _k; "
                f"_k.serve({p2c_r}, {c2p_w}, {json.dumps(cfg)!r})")
        try:
            self._proc = subprocess.Popen(
                [self.python, "-c", boot], cwd=self.workdir, env=self._env(), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(p2c_r, c2p_w), start_new_session=True)
        except OSError:
            for fd in (p2c_r, p2c_w, c2p_r, c2p_w):
                os.close(fd)
            self._proc = None
            return False
        os.close(p2c_r)
        os.close(c2p_w)
        self._ctl_w, self._ctl_r, self._rbuf = p2c_w, c2p_r, b""
        self._carry = {1: b"", 2: b""}
        self._starts += 1
        if self._starts > 1:
            self.kernel_restarts += 1
        msg = self._recv(time.monotonic() + START_TIMEOUT)
        if not msg or not msg.get("ready"):
            self._kill()
            return False
        return True

    def _kill(self) -> None:
        p = self._proc
        if p is not None:
            try:
                os.killpg(p.pid, signal.SIGKILL)      # the kernel and anything it spawned
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                p.wait(timeout=10)
            except Exception:                           # noqa: BLE001
                pass
            for s in (p.stdout, p.stderr):
                try:
                    s.close()
                except Exception:                       # noqa: BLE001
                    pass
        for fd in (self._ctl_w, self._ctl_r):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._proc, self._ctl_w, self._ctl_r = None, None, None

    def _send(self, obj: dict) -> bool:
        data = json.dumps(obj).encode("utf-8")
        try:
            os.write(self._ctl_w, struct.pack(">I", len(data)) + data)
            return True
        except (OSError, TypeError):
            return False

    def _try_parse(self) -> dict | None:
        if len(self._rbuf) < 4:
            return None
        (n,) = struct.unpack(">I", self._rbuf[:4])
        if len(self._rbuf) < 4 + n:
            return None
        msg = json.loads(self._rbuf[4:4 + n].decode("utf-8"))
        self._rbuf = self._rbuf[4 + n:]
        return msg

    def _recv(self, deadline: float) -> dict | None:
        """One control message, or None on EOF/timeout (only used at start-up, when no output is expected)."""
        while True:
            msg = self._try_parse()
            if msg is not None:
                return msg
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            r = _readable([self._ctl_r], left)
            if not r:
                return None
            chunk = os.read(self._ctl_r, 65536)
            if not chunk:
                return None
            self._rbuf += chunk

    # ------------------------------------------------------------------ one cell
    def run(self, code: str) -> ExecResult:
        if self.kernel_fallback:
            return super().run(code)
        # the replay sandbox's compile gate, verbatim in effect: a non-compiling cell never reaches the kernel
        try:
            compile(textwrap.dedent(code), "<cell>", "exec")
        except SyntaxError:
            return super().run(code)            # Sandbox.run returns before executing anything for a SyntaxError
        if not self._alive():
            # a kernel that died BETWEEN cells was never reported (a death during a cell is, by _lost)
            note_restart = self._starts > 0 and not self._told_lost
            self._told_lost = False
            while not self._start():
                self._start_failures += 1
                if self._start_failures >= MAX_START_FAILURES:
                    self.kernel_fallback = True
                    res = super().run(code)
                    return ExecResult(res.stdout, (res.stderr.rstrip() + "\n" if res.stderr.strip() else "")
                                      + FALLBACK_NOTE, res.returncode, res.timed_out)
            if note_restart:                    # died between cells without a result that said so
                self._lost_pending = True
        self.attempts += 1
        self.cells.append(code)
        cell = textwrap.dedent(code)
        if os.environ.get("GBP_ECHO_LAST") == "1":
            cell = _echo_last_expression(cell)
        n = len(self.cells) - 1
        marker = f"\n__GBP_KERNEL_END_{secrets.token_hex(8)}_{n}__\n"
        mk = marker.encode()
        out, err = _Stream(mk, self._carry[1]), _Stream(mk, self._carry[2])
        t0 = time.monotonic()
        if not self._send({"op": "exec", "code": cell, "cell": n, "marker": marker, "cpu": self.timeout}):
            return self._lost(out, err, "exited before it could run this cell", t0)
        fds = {self._proc.stdout.fileno(): out, self._proc.stderr.fileno(): err}
        deadline, phase, reply, grace_end = t0 + self.timeout, "run", None, None
        while True:
            now = time.monotonic()
            if reply is not None:
                if out.done and err.done:
                    break
                if grace_end is None:
                    grace_end = now + MARKER_GRACE
                if now >= grace_end:
                    break
                wait = grace_end - now
            elif phase == "run":
                if now >= deadline:
                    self.kernel_timeouts += 1
                    try:
                        os.kill(self._proc.pid, signal.SIGINT)
                    except OSError:
                        pass
                    phase, deadline = "grace", now + KILL_GRACE
                    continue
                wait = deadline - now
            else:                               # "grace": interrupted, not answering
                if now >= deadline:
                    return self._lost(out, err, f"did not stop within {KILL_GRACE:.0f} s of the time limit "
                                                "and was killed", t0, timed_out=True)
                wait = deadline - now
            watch = [fd for fd, s in fds.items() if not s.done]
            if reply is None:
                watch.append(self._ctl_r)
            try:
                ready = _readable(watch, max(0.0, min(wait, 1.0)))
            except (OSError, ValueError):
                return self._lost(out, err, "stopped responding", t0)
            for fd in ready:
                chunk = os.read(fd, 65536)
                if fd == self._ctl_r:
                    if not chunk:
                        return self._lost(out, err, self._death_reason(), t0)
                    self._rbuf += chunk
                    msg = self._try_parse()
                    if msg is not None and msg.get("done"):
                        reply = msg
                    continue
                if not chunk:                   # the kernel closed its stdout/stderr: it is exiting
                    fds[fd].done = True
                    continue
                fds[fd].data(chunk)
            if reply is None and not self._alive():
                return self._lost(out, err, self._death_reason(), t0)
        self._carry = {1: out.carry, 2: err.carry}
        self.cell_seconds.append(round(time.monotonic() - t0, 3))
        rc = int(reply.get("rc") or 0)
        timed_out = bool(reply.get("interrupted")) and phase == "grace"
        if timed_out:
            self.cells.pop()                    # as the replay sandbox drops a timed-out cell
        stderr = err.text()
        if self._lost_pending:                  # a kernel lost between cells: say so on the next result
            stderr = KERNEL_LOST_NOTE.format(why="had stopped") + ("\n" + stderr if stderr else "")
            self._lost_pending = False
        if rc != 0 or timed_out:
            self.failures += 1
        return ExecResult(out.text(), stderr, -1 if timed_out else rc, timed_out=timed_out)

    def _death_reason(self) -> str:
        p = self._proc
        try:
            rc = p.wait(timeout=5) if p is not None else None
        except Exception:                               # noqa: BLE001
            rc = None
        if rc is None:
            return "stopped responding"
        if rc < 0:
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = f"signal {-rc}"
            hint = " (out of memory?)" if -rc == signal.SIGKILL else ""
            return f"was killed by {name}{hint}"
        return f"exited with status {rc}"

    def _lost(self, out: _Stream, err: _Stream, why: str, t0: float, timed_out: bool = False) -> ExecResult:
        """The kernel is gone (or is being killed): collect what it printed, kill the group, tell the model."""
        # drain what is already buffered (non-blocking), then kill
        for s, f in ((out, self._proc.stdout if self._proc else None), (err, self._proc.stderr if self._proc else None)):
            if f is None:
                continue
            try:
                while True:
                    r = _readable([f.fileno()], 0.2)
                    if not r:
                        break
                    chunk = os.read(f.fileno(), 65536)
                    if not chunk:
                        break
                    s.data(chunk)
            except (OSError, ValueError):
                pass
        self._kill()
        self._carry = {1: b"", 2: b""}
        self._lost_pending = False
        self._told_lost = True
        if timed_out:
            self.cells.pop()
        self.failures += 1
        self.cell_seconds.append(round(time.monotonic() - t0, 3))
        stderr = err.text()
        note = KERNEL_LOST_NOTE.format(why=why)
        stderr = (stderr.rstrip() + "\n" if stderr.strip() else "") + note
        return ExecResult(out.text(), stderr, -1, timed_out=timed_out)

    def stats(self) -> dict:
        cs = self.cell_seconds
        return {"kernel_restarts": self.kernel_restarts, "kernel_timeouts": self.kernel_timeouts,
                "kernel_fallback": self.kernel_fallback, "cells_run": len(cs),
                "cell_seconds_sum": round(sum(cs), 2), "cell_seconds_max": max(cs) if cs else 0.0}

    def cleanup(self) -> None:
        if self._proc is not None and self._alive():
            self._send({"op": "shutdown"})
        self._kill()
        super().cleanup()
