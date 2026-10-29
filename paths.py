"""Single source of truth for where things live.

Code lives permanently in /project. Bulk data lives in a Unity scratch
workspace, which EXPIRES and whose mount point is not stable -- an expired
workspace re-allocated later can land on /scratch3 instead of /scratch4. So
nothing may hardcode the scratch path.

Resolution order for the data root:
  1. $DEEP_DATA_ROOT                      (explicit override)
  2. `ws_find <WORKSPACE_NAME>`           (authoritative)
  3. ./data symlink next to this file     (convenience)

Shell scripts get the same value from scripts/env.sh.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

WORKSPACE_NAME = "deep-ai4s"

PROJECT_ROOT = Path(__file__).resolve().parent
HARNESS = PROJECT_ROOT / "harness"
SCRIPTS = PROJECT_ROOT / "scripts"
BENCHMARKS = PROJECT_ROOT / "benchmarks"
RESULTS = PROJECT_ROOT / "results"
GENEBENCH_PUBLIC = BENCHMARKS / "genebench-pro-public"


def _from_ws_find() -> Path | None:
    if not shutil.which("ws_find"):
        return None
    try:
        out = subprocess.run(
            ["ws_find", WORKSPACE_NAME], capture_output=True, text=True, timeout=30
        )
    except Exception:
        return None
    for line in out.stdout.splitlines():
        p = Path(line.strip())
        if line.strip().startswith("/") and p.is_dir():
            return p
    return None


def data_root(required: bool = True) -> Path:
    env = os.environ.get("DEEP_DATA_ROOT")
    if env and Path(env).is_dir():
        return Path(env)

    found = _from_ws_find()
    if found:
        return found

    link = PROJECT_ROOT / "data"
    if link.is_dir():
        return link.resolve()

    if required:
        raise FileNotFoundError(
            f"Data root not found. The '{WORKSPACE_NAME}' scratch workspace may have "
            f"expired.\n"
            f"  check:   ws_list\n"
            f"  extend:  ws_extend {WORKSPACE_NAME} 30\n"
            f"  or set:  export DEEP_DATA_ROOT=/path/to/data"
        )
    return link


# Convenience accessors -- call data_root() lazily, never at import time, so
# importing this module still works after the workspace expires.
def opentargets(release: str | None = None) -> Path:
    p = data_root() / "opentargets"
    return p / release if release else p


def pubmed() -> Path:
    return data_root() / "pubmed"


def pmc_txt() -> Path:
    return data_root() / "pmc" / "txt"


def chembl() -> Path:
    return data_root() / "chembl"


def uniprot() -> Path:
    return data_root() / "uniprot"


def aact() -> Path:
    return data_root() / "aact"


def venv_python() -> Path:
    return PROJECT_ROOT / "venv" / "bin" / "python"


def workspace_days_left() -> float | None:
    """Days until the scratch workspace expires, or None if undeterminable."""
    if not shutil.which("ws_list"):
        return None
    try:
        out = subprocess.run(["ws_list"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    block, seen = [], False
    for line in out.splitlines():
        if line.strip().startswith("id:"):
            seen = line.strip() == f"id: {WORKSPACE_NAME}"
        elif seen and "remaining time" in line:
            block = line.split(":", 1)[1].strip().split()
            break
    if not block:
        return None
    days = hours = 0.0
    for i, tok in enumerate(block):
        if tok.startswith("day"):
            days = float(block[i - 1])
        elif tok.startswith("hour"):
            hours = float(block[i - 1])
    return days + hours / 24.0


if __name__ == "__main__":
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    try:
        print(f"DATA_ROOT    : {data_root()}")
    except FileNotFoundError as e:
        print(f"DATA_ROOT    : MISSING\n{e}")
    d = workspace_days_left()
    if d is not None:
        flag = "  <-- EXTEND SOON" if d < 7 else ""
        print(f"workspace    : {d:.1f} days left{flag}")
