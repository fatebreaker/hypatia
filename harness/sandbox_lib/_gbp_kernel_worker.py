"""Persistent Python kernel for harness v2 (harness/kernel.py, switch GBP_KERNEL=1). docs/HARNESS_V2.md.

Runs UNDER THE SANDBOX INTERPRETER (venv/bin/python), one process per episode, started by KernelSandbox:

    python -c "import sys; sys.path.insert(0, LIB); import _gbp_kernel_worker as k; k.serve(CTL_IN, CTL_OUT, CFG)"

Why it exists: harness/sandbox.py REPLAYS every earlier cell on each exec, so one slow cell is re-run on every later
turn (the 30-60 min RL stragglers, docs/RL_PLAN.md s.9). Here state lives in this process and each cell runs once.

Protocol (control pipes, not stdout): 4-byte big-endian length + UTF-8 JSON, both directions.
  parent -> worker  {"op": "exec", "code": str, "cell": int, "marker": str, "cpu": seconds}
  worker -> parent  {"ready": true, "pid": int}                                  once, after setup
                    {"done": true, "rc": int, "error": str|null, "interrupted": bool}   per exec
Cell output goes to this process's real fd 1 / fd 2 (pipes the parent drains, bounded); after each cell the worker
flushes Python and C buffers and writes `marker` to BOTH fds, so the parent knows where the cell's output ends.

The namespace is the real __main__ module's dict (so `if __name__ == "__main__":` runs and functions defined in cells
pickle, as they do in the replay script). This module's own globals are NOT that namespace, so a cell cannot clobber
the loop. Resource limits are applied by the process to itself, exactly as the replay preamble does; RLIMIT_CPU is
cumulative per process, so only its SOFT limit is moved, per cell, to (CPU used so far + the cell budget), and
SIGXCPU becomes an exception in the cell instead of killing the kernel. SIGINT (the parent's wall-clock interrupt)
becomes KeyboardInterrupt; a cell that ignores it is killed by the parent and the kernel is restarted there.
"""
from __future__ import annotations

import builtins
import ctypes
import json
import linecache
import os
import resource
import signal
import struct
import sys
import threading
import time
import traceback


class CPUTimeLimitExceeded(Exception):
    """Raised inside a cell when it has used its CPU-time budget (RLIMIT_CPU soft limit, SIGXCPU)."""


_IN_CELL = False                 # SIGXCPU/SIGINT only turn into exceptions while a cell runs


def _on_xcpu(signum, frame):
    if _IN_CELL:
        raise CPUTimeLimitExceeded("the cell exceeded its CPU-time limit")


def _on_int(signum, frame):
    if _IN_CELL:
        raise KeyboardInterrupt
    # between cells: ignore (a late interrupt must not kill the kernel)


def _read_exact(fd: int, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def _recv(fd: int) -> dict:
    (n,) = struct.unpack(">I", _read_exact(fd, 4))
    return json.loads(_read_exact(fd, n).decode("utf-8"))


def _send(fd: int, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    os.write(fd, struct.pack(">I", len(data)) + data)


def _flush_all() -> None:
    for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            s.flush()
        except Exception:                                   # noqa: BLE001
            pass
    try:                                                    # C-level stdio buffers (printf from extensions)
        ctypes.CDLL(None).fflush(None)
    except Exception:                                       # noqa: BLE001
        pass


def _write_marker(marker: str) -> None:
    m = marker.encode()
    for fd in (1, 2):
        try:
            os.write(fd, m)
        except OSError:
            pass


def _watch_parent(ppid: int) -> None:
    """Exit if the parent (the harness process) dies: an orphaned kernel must not outlive its episode."""
    while True:
        time.sleep(5)
        if os.getppid() != ppid:
            os._exit(0)


def _apply_limits(cfg: dict) -> None:
    mem = int(cfg.get("mem_bytes") or 0)
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    nproc = int(cfg.get("nproc") or 0)
    if nproc:
        try:
            h = resource.getrlimit(resource.RLIMIT_NPROC)[1]
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc if h == resource.RLIM_INFINITY else min(nproc, h), h))
        except (ValueError, OSError):
            pass


def _set_cpu_budget(seconds: float | None) -> None:
    """Soft RLIMIT_CPU = CPU already used + seconds (None: lift the soft limit back to the hard limit)."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if seconds is None:
            new = hard
        else:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            new = int(ru.ru_utime + ru.ru_stime + seconds) + 1
            if hard != resource.RLIM_INFINITY:
                new = min(new, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (new, hard))
    except (ValueError, OSError):
        pass


def serve(ctl_in: int, ctl_out: int, cfg_json: str) -> None:
    global _IN_CELL
    cfg = json.loads(cfg_json)
    lib = cfg.get("lib_dir") or ""
    # the worker's own directory was put on sys.path to import this module; keep it only where cells may import the
    # environment's helpers (CHECKS / APIDOCS), and then at the END so a user's own module of the same name wins.
    while lib in sys.path:
        sys.path.remove(lib)
    if cfg.get("expose_lib"):
        sys.path.append(lib)
    _apply_limits(cfg)
    signal.signal(signal.SIGXCPU, _on_xcpu)
    signal.signal(signal.SIGINT, _on_int)
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()

    main = sys.modules["__main__"]
    ns = main.__dict__
    for k in list(ns):
        if k not in ("__name__", "__doc__", "__builtins__", "__loader__", "__spec__", "__package__"):
            del ns[k]
    ns["__name__"] = "__main__"
    ns["__builtins__"] = builtins
    explain = None
    for stmt in cfg.get("preamble") or []:                 # e.g. `from gbp_help import help_api` (APIDOCS)
        try:
            exec(compile(stmt, "<kernel-preamble>", "exec"), ns)
        except Exception:                                   # noqa: BLE001
            pass
    if cfg.get("apidocs"):
        try:
            import gbp_help                                  # harness/sandbox_lib/gbp_help.py
            explain = gbp_help.explain_exception
        except Exception:                                   # noqa: BLE001
            explain = None
    real_out, real_err = sys.stdout, sys.stderr
    _send(ctl_out, {"ready": True, "pid": os.getpid()})

    while True:
        try:
            msg = _recv(ctl_in)
        except (EOFError, OSError):
            os._exit(0)
        op = msg.get("op")
        if op == "shutdown":
            os._exit(0)
        if op != "exec":
            continue
        code, n, marker = msg["code"], int(msg.get("cell", 0)), msg["marker"]
        fname = f"<cell-{n}>"
        linecache.cache[fname] = (len(code), None, code.splitlines(True), fname)
        rc, err, interrupted = 0, None, False
        _set_cpu_budget(float(msg.get("cpu") or 0) or None)
        try:
            compiled = compile(code, fname, "exec")
            _IN_CELL = True
            try:
                exec(compiled, ns)
            finally:
                _IN_CELL = False
        except KeyboardInterrupt:
            rc, err, interrupted = -1, "KeyboardInterrupt", True
        except SystemExit as exc:                           # sys.exit()/exit(): end the CELL, as the script would
            c = exc.code
            if c is None:
                rc = 0
            elif isinstance(c, int):
                rc = c
            else:
                print(c, file=sys.stderr)
                rc = 1
            err = None if rc == 0 else "SystemExit"
        except BaseException as exc:                        # noqa: BLE001
            _IN_CELL = False
            rc, err = 1, type(exc).__name__
            etype, value, tb = sys.exc_info()
            try:
                # drop this module's exec frame: the traceback starts in the cell, as the script's would
                tb_user = tb.tb_next if tb is not None else None
                sys.stderr.write("".join(traceback.format_exception(etype, value, tb_user)))
                if explain is not None:
                    note = explain(etype, value, tb_user)
                    if note:
                        sys.stderr.write(note if note.endswith("\n") else note + "\n")
            except Exception:                               # noqa: BLE001
                pass
            del tb
        finally:
            _IN_CELL = False
            _set_cpu_budget(None)
        # a cell that swapped sys.stdout/sys.stderr must not swallow every later cell's output
        if sys.stdout is not real_out:
            sys.stdout = real_out
        if sys.stderr is not real_err:
            sys.stderr = real_err
        _flush_all()
        _write_marker(marker)
        try:
            _send(ctl_out, {"done": True, "rc": rc, "error": err, "interrupted": interrupted})
        except OSError:
            os._exit(0)
