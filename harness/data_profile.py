#!/usr/bin/env python
"""PROFILE (harness v2, switch GBP_PROFILE=1): an automatic, descriptive profile of the task's data files.

docs/HARNESS_V2.md. Evidence only: per file its schema, dtypes, row counts, missingness, low-cardinality columns with
their counts, numeric ranges and date ranges. It never names a trap, a hazard or a method, and it never interprets.

    venv/bin/python harness/data_profile.py DATA_DIR [--skip siblings] [--max-chars 6000]

Run as a SUBPROCESS under the sandbox interpreter (harness/v2.py does, with a timeout), because the trainer's own
interpreter may lack pandas extras and a 1 GB file must not be parsed inside the trainer. Large inputs are capped:
at most MAX_ROWS rows per table are read (the profile says so), files above MAX_BYTES are listed by size only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MAX_ROWS = 200_000
MAX_BYTES = 400 * 1024 ** 2
LOW_CARD = 12            # a column with at most this many distinct values is shown with its counts
MAX_COLS = 30            # columns shown in full per table; the rest are summarised
HEADER = ("Data profile (computed automatically by the environment from the files above; descriptive only; "
          "long listings are cut):")

TABLE_EXT = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".txt": None}


def _fmt(v) -> str:
    try:
        import numpy as np
        if isinstance(v, (float, np.floating)):
            if v != v:
                return "nan"
            a = abs(float(v))
            if a != 0 and (a >= 1e6 or a < 1e-3):
                return f"{float(v):.3g}"
            return f"{float(v):.4g}"
        if isinstance(v, (np.integer,)):
            return str(int(v))
    except Exception:                                        # noqa: BLE001
        pass
    s = str(v)
    return s if len(s) <= 24 else s[:21] + "..."


def _read_table(path: Path):
    import pandas as pd
    name = path.name.lower()
    stem = name[:-3] if name.endswith(".gz") else name
    ext = os.path.splitext(stem)[1]
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return df.head(MAX_ROWS), len(df) > MAX_ROWS, len(df)
    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, nrows=MAX_ROWS)
        return sheets, False, None
    sep = TABLE_EXT.get(ext, None)
    if sep is None:
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", errors="replace") as fh:
            first = fh.readline()
        sep = "\t" if first.count("\t") > first.count(",") else "," if "," in first else r"\s+"
    kw = {} if sep == r"\s+" else {"low_memory": False}
    df = pd.read_csv(path, sep=sep, nrows=MAX_ROWS + 1, **kw)
    capped = len(df) > MAX_ROWS
    return df.head(MAX_ROWS), capped, None


def _date_range(s):
    import pandas as pd
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            d = s.dropna()
        elif s.dtype == object:
            sample = s.dropna().astype(str).head(200)
            if len(sample) == 0 or not sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}").mean() > 0.9:
                return None
            d = pd.to_datetime(s, errors="coerce").dropna()
            if len(d) < 0.9 * s.notna().sum():
                return None
        else:
            return None
        if len(d) == 0:
            return None
        return f"dates {d.min().date()} .. {d.max().date()}"
    except Exception:                                        # noqa: BLE001
        return None


def _profile_df(df, label: str, capped: bool, total_rows) -> list[str]:
    import pandas as pd
    n = total_rows if total_rows is not None else len(df)
    head = f"{label} -- {n:,}{'+' if capped else ''} rows x {df.shape[1]} columns"
    if capped:
        head += f" (first {MAX_ROWS:,} rows profiled)"
    lines = [head]
    cols = list(df.columns)
    shown, rest = cols[:MAX_COLS], cols[MAX_COLS:]
    for c in shown:
        s = df[c]
        miss = int(s.isna().sum())
        nun = int(s.nunique(dropna=True))
        mtxt = f"missing {miss} ({100 * miss / max(len(s), 1):.1f}%)" if miss else "no missing"
        desc = ""
        dr = _date_range(s)
        if dr:
            desc = dr
        elif 0 < nun <= LOW_CARD and (not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                                      or nun <= max(2, s.notna().sum() // 2)):
            vc = s.value_counts(dropna=True).head(LOW_CARD)
            try:
                vc = vc.sort_index()
            except TypeError:
                pass
            desc = "values " + " | ".join(f"{_fmt(k)}: {v}" for k, v in vc.items())
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            a = pd.to_numeric(s, errors="coerce").dropna()
            if len(a):
                desc = f"min {_fmt(a.min())}  median {_fmt(a.median())}  max {_fmt(a.max())}"
        else:
            ex = [str(x) for x in s.dropna().astype(str).head(3)]
            desc = "e.g. " + ", ".join(e if len(e) <= 24 else e[:21] + "..." for e in ex)
        lines.append(f"  {str(c)[:32]:<32} {str(s.dtype):<9} {nun:>7} distinct  {mtxt:<22} {desc}")
    if rest:
        import numpy as np
        sub = df[rest]
        num = sub.select_dtypes(include=[np.number])
        miss = int(sub.isna().sum().sum())
        line = f"  ... {len(rest)} more columns ({len(num.columns)} numeric; {miss} missing cells)"
        if len(num.columns):
            vals = num.to_numpy(dtype=float)
            fin = vals[np.isfinite(vals)]
            if fin.size:
                line += f"; numeric range {_fmt(fin.min())} .. {_fmt(fin.max())}"
        names = ", ".join(str(x) for x in rest[:5])
        lines.append(line + f"; names {names}{', ...' if len(rest) > 5 else ''}")
    return lines


def _profile_npz(path: Path) -> list[str]:
    import numpy as np
    out = [f"{path.name} -- numpy archive"]
    with np.load(path, allow_pickle=False) as z:
        for k in list(z.files)[:20]:
            a = z[k]
            line = f"  {k:<32} shape {a.shape} dtype {a.dtype}"
            if a.dtype.kind in "fiu" and a.size:
                af = a.astype(float)
                fin = af[np.isfinite(af)]
                nn = int(a.size - fin.size)
                if fin.size:
                    line += f"  min {_fmt(fin.min())}  max {_fmt(fin.max())}"
                if nn:
                    line += f"  non-finite {nn}"
            out.append(line)
    return out


def _profile_json(path: Path) -> list[str]:
    with (gzip.open(path, "rt") if path.name.endswith(".gz") else open(path)) as fh:
        obj = json.load(fh)
    out = [f"{path.name} -- JSON {type(obj).__name__}"]
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:25]:
            t = type(v).__name__
            ln = f" (len {len(v)})" if hasattr(v, "__len__") and not isinstance(v, str) else ""
            val = f" = {_fmt(v)}" if isinstance(v, (int, float, str, bool)) else ""
            out.append(f"  {str(k)[:32]:<32} {t}{ln}{val}")
    elif isinstance(obj, list):
        out.append(f"  {len(obj)} elements; first is {type(obj[0]).__name__ if obj else '-'}")
        if obj and isinstance(obj[0], dict):
            import pandas as pd
            out += _profile_df(pd.DataFrame(obj[:MAX_ROWS]), "  as a table", len(obj) > MAX_ROWS, len(obj))[1:]
    return out


def _profile_h5(path: Path) -> list[str]:
    import h5py
    out = [f"{path.name} -- HDF5"]
    with h5py.File(path, "r") as f:
        items = []
        f.visititems(lambda name, o: items.append((name, o)) if len(items) < 30 else None)
        for name, o in items:
            if hasattr(o, "shape"):
                out.append(f"  {name:<40} shape {o.shape} dtype {o.dtype}")
            else:
                out.append(f"  {name}/")
    return out


def profile_file(path: Path, rel: str) -> list[str]:
    size = path.stat().st_size
    name = path.name.lower()
    stem = name[:-3] if name.endswith(".gz") else name
    ext = os.path.splitext(stem)[1]
    if size > MAX_BYTES:
        return [f"{rel} -- {size:,} bytes (too large to profile)"]
    try:
        if ext in TABLE_EXT or ext in (".parquet", ".pq", ".xlsx", ".xls"):
            res, capped, total = _read_table(path)
            if isinstance(res, dict):
                out = []
                for sheet, df in list(res.items())[:6]:
                    out += _profile_df(df, f"{rel} [sheet {sheet}]", len(df) >= MAX_ROWS, None)
                return out
            return _profile_df(res, rel, capped, total)
        if ext == ".npz":
            return [rel + _profile_npz(path)[0][len(path.name):]] + _profile_npz(path)[1:]
        if ext == ".npy":
            import numpy as np
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            return [f"{rel} -- numpy array shape {a.shape} dtype {a.dtype}"]
        if ext == ".json":
            lines = _profile_json(path)
            return [rel + lines[0][len(path.name):]] + lines[1:]
        if ext in (".h5", ".hdf5", ".h5ad"):
            lines = _profile_h5(path)
            return [rel + lines[0][len(path.name):]] + lines[1:]
    except Exception as exc:                                  # noqa: BLE001
        return [f"{rel} -- could not be profiled ({type(exc).__name__})"]
    return [f"{rel} -- {size:,} bytes"]


def profile_dir(data_dir: Path, skip=("siblings",), max_chars: int = 6000) -> str:
    data_dir = Path(data_dir)
    files = []
    for root, dirs, fnames in os.walk(data_dir, followlinks=True):
        rel_root = Path(root).relative_to(data_dir)
        if rel_root.parts and rel_root.parts[0] in skip:
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (not rel_root.parts and d in skip))
        for fn in sorted(fnames):
            if fn.startswith("."):
                continue
            files.append(Path(root) / fn)
    blocks = []
    for p in files:
        rel = "data_files/" + str(p.relative_to(data_dir))
        blocks.append("\n".join(profile_file(p, rel)))
    text = HEADER + "\n" + "\n".join(blocks) if blocks else HEADER + "\n(no files)"
    if len(text) > max_chars:
        cut = text[: max_chars - 60].rsplit("\n", 1)[0]
        text = cut + f"\n... [profile cut here: {len(text) - len(cut)} more characters not shown]"
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--skip", action="append", default=None)
    ap.add_argument("--max-chars", type=int, default=6000)
    a = ap.parse_args()
    sys.stdout.write(profile_dir(Path(a.data_dir), tuple(a.skip or ["siblings"]), a.max_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
