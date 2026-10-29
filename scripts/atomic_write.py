#!/usr/bin/env python
"""Replace a file without breaking whatever is reading it right now.

Writing in place (open(p,"w"), Path.write_text, sed -i) truncates the file first, and a pod that is executing that
script re-reads it as it goes: on 2026-09-16 that killed a running astronomy evaluation with "syntax error near
unexpected token `done'" and a benchmark run with "unexpected EOF while looking for matching quote". Writing a
temporary file in the same directory and renaming it over the original is atomic -- a reader keeps the old inode and
finishes on the old bytes -- and the mode is copied, which an atomic write otherwise drops (that cost a queued run
twice, with exit 126 "Permission denied").

  venv/bin/python scripts/atomic_write.py TARGET < new_contents
  python -c "from scripts.atomic_write import write; write(path, text)"
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


def write(target: str | os.PathLike, text: str) -> Path:
    p = Path(target)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        if p.exists():
            shutil.copymode(p, tmp)          # an atomic write drops the executable bit otherwise
        os.replace(tmp, p)
    except BaseException:
        os.unlink(tmp)
        raise
    return p


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: atomic_write.py TARGET < new_contents")
    print(f"wrote {write(sys.argv[1], sys.stdin.read())}")
