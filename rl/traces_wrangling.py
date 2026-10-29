"""Neutral data-wrangling traces -- the CONTROL corpus for Section 6.

Section 6.1 claims the causal-reasoning loss comes from WHAT our demonstrations
omit (they always name the estimand). The rival explanation is duller: LoRA SFT
on long agentic code traces degrades causal reasoning no matter what the traces
are about. Nothing in the paper currently distinguishes these.

This corpus is the control. Identical in every respect that is not the
hypothesis -- same system prompt, same sandbox, same multi-turn code-and-output
format, same trace length, same order of magnitude of examples -- but the tasks
contain no statistics, no inference, and no estimand of any kind. They are pure
data manipulation: joins, deduplication, group-and-count, reshaping, parsing.

If fine-tuning on THESE also costs causal reasoning, the Section 6.1 story is
wrong and the effect is generic forgetting. If it does not, the effect is
specific to what our scientific demonstrations leave out.

Answers are computed by executing the reference solution, so ground truth is
exact by construction.

  venv/bin/python rl/traces_wrangling.py --n 620
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

import numpy as np

SYSTEM = """You are an expert computational biologist and statistician.

You solve a data-analysis problem by writing Python. Rules:

1. To run code, emit exactly one ```python fenced block. You will receive its
   stdout and stderr. State persists across blocks like a notebook.
2. Inspect the data before modelling it. These datasets contain deliberate
   irregularities -- batch effects, ascertainment, winner's curse, ambient
   contamination, pseudogene mismapping, LD structure. Diagnose before you
   estimate, and let what you find change your estimator.
3. Available: numpy, scipy, pandas, statsmodels, scikit-learn. Files are under
   data_files/ in the working directory.
4. When confident, emit your final answer as ONE JSON object in a ```json
   block, with no code block in that same message.

Do not guess numbers. Compute them."""

WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta",
         "north", "south", "east", "west", "amber", "cobalt", "ivory", "slate"]


def run_cells(workdir: Path, cells: list[str]) -> list[str]:
    delim = "===CELL-BREAK==="
    src = "".join(c + f"\nprint({delim!r})\n" for c in cells)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, cwd=workdir, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return [o.rstrip() for o in r.stdout.split(delim + "\n")[:len(cells)]]


def build(rng, idx, only=None):
    """One of four wrangling shapes, chosen by index.

    `only` pins the shape, which is how the diversity arm is built: the same
    generator, the same trace format and the same corpus size, but drawn from a
    single template instead of four. Corpus diversity is the last standing
    explanation for why the ten-family genomics corpus trains stably (range
    0.013 over four seeds) while this one does not (0.311 over six).
    """
    import pandas as pd
    kind = idx % 4 if only is None else only
    n = int(rng.integers(300, 1200))
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td); (wd / "data_files").mkdir()
        if kind == 0:                              # group-and-count with dupes
            key = rng.choice(WORDS, n)
            rid = rng.integers(1000, 1000 + n // 2, n)   # duplicate ids
            pd.DataFrame({"record_id": rid, "category": key,
                          "value": rng.integers(1, 100, n)}).to_csv(
                wd / "data_files" / "records.csv", index=False)
            question = ("After removing duplicate record_id entries (keeping the first), "
                        "how many DISTINCT categories remain, and what is the total value?")
            keys = {"n_categories": "int", "total_value": "int"}
            c1 = ("import pandas as pd\n"
                  "df = pd.read_csv('data_files/records.csv')\n"
                  "print(df.shape, df['record_id'].duplicated().sum(), 'duplicate ids')")
            c2 = ("d = df.drop_duplicates('record_id', keep='first')\n"
                  "print('n_categories', d['category'].nunique())\n"
                  "print('total_value', int(d['value'].sum()))")
        elif kind == 1:                            # join with unmatched rows
            ids = np.arange(n)
            left = pd.DataFrame({"id": ids, "score": rng.integers(0, 50, n)})
            keep = rng.choice(ids, size=int(n * 0.7), replace=False)
            right = pd.DataFrame({"id": np.sort(keep),
                                  "label": rng.choice(WORDS[:5], len(keep))})
            left.to_csv(wd / "data_files" / "left.csv", index=False)
            right.to_csv(wd / "data_files" / "right.csv", index=False)
            question = ("Inner-join left.csv to right.csv on id. How many rows survive, "
                        "and what is the mean score among survivors (2 decimals)?")
            keys = {"n_rows": "int", "mean_score": "float"}
            c1 = ("import pandas as pd\n"
                  "l = pd.read_csv('data_files/left.csv'); r = pd.read_csv('data_files/right.csv')\n"
                  "print(l.shape, r.shape)")
            c2 = ("m = l.merge(r, on='id', how='inner')\n"
                  "print('n_rows', len(m))\n"
                  "print('mean_score', round(m['score'].mean(), 2))")
        elif kind == 2:                            # wide-to-long reshape
            cols = {f"t{t}": rng.normal(10, 3, n).round(3) for t in range(1, 5)}
            pd.DataFrame({"subject": np.arange(n), **cols}).to_csv(
                wd / "data_files" / "wide.csv", index=False)
            question = ("Reshape wide.csv from wide to long over the t1..t4 columns. "
                        "How many rows does the long table have, and which timepoint has "
                        "the highest mean?")
            keys = {"n_rows": "int", "max_timepoint": "str"}
            c1 = ("import pandas as pd\n"
                  "df = pd.read_csv('data_files/wide.csv')\n"
                  "print(df.shape, df.columns.tolist())")
            c2 = ("lg = df.melt(id_vars='subject', var_name='timepoint', value_name='v')\n"
                  "print('n_rows', len(lg))\n"
                  "print('max_timepoint', lg.groupby('timepoint')['v'].mean().idxmax())")
        else:                                      # parse a messy delimited field
            tags = [";".join(rng.choice(WORDS, rng.integers(1, 4), replace=False))
                    for _ in range(n)]
            pd.DataFrame({"row": np.arange(n), "tags": tags}).to_csv(
                wd / "data_files" / "tagged.csv", index=False)
            question = ("The tags column holds semicolon-separated tags. How many distinct "
                        "tags appear in total, and how many rows carry more than one tag?")
            keys = {"n_tags": "int", "n_multi": "int"}
            c1 = ("import pandas as pd\n"
                  "df = pd.read_csv('data_files/tagged.csv')\n"
                  "print(df.shape); print(df['tags'].head(3).tolist())")
            c2 = ("sp = df['tags'].str.split(';')\n"
                  "print('n_tags', len(set(t for row in sp for t in row)))\n"
                  "print('n_multi', int((sp.str.len() > 1).sum()))")
        files = sorted(q.name for q in (wd / 'data_files').glob('*'))
        outs = run_cells(wd, [c1, c2])
    ans = {}
    for line in outs[1].splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in keys:
            ans[parts[0]] = float(parts[1]) if keys[parts[0]] == "float" else (
                int(parts[1]) if keys[parts[0]] == "int" else parts[1])
    if set(ans) != set(keys):
        return None
    task = (f"You are given a data file to process.\n\n{question}\n\n"
            'Return the JSON object in your final answer:\n{\n"answer": '
            + json.dumps({k: "<value>" for k in keys})
            + ',\n"reasoning": "<what you did>"\n}\n\nFiles in your working directory:\n  '
            + "\n  ".join(f"data_files/{f}" for f in files))
    r1 = "I look at the file first to see its shape and columns before transforming anything."
    r2 = "Now the transformation itself, printing each requested quantity."
    fin = ("```json\n{\n\"answer\": " + json.dumps(ans)
           + ",\n\"reasoning\": \"Loaded the file, inspected its structure, applied the "
             "requested transformation and read off the quantities.\"\n}\n```")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task},
            {"role": "assistant", "content": r1 + "\n\n```python\n" + c1 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[0]},
            {"role": "assistant", "content": r2 + "\n\n```python\n" + c2 + "\n```"},
            {"role": "user", "content": "STDOUT:\n" + outs[1]},
            {"role": "assistant", "content": fin}]
    return {"id": f"wrangle_{idx:04d}", "messages": msgs, "reward": 1.0, "n_cells": 2,
            "plan": "inspect then transform", "_family": "wrangling"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=620)
    ap.add_argument("--only", type=int, default=None,
                    help="pin to a single template (0-3) for the low-diversity arm")
    ap.add_argument("--out", default=str(data_root() / "sft/wrangling.jsonl"))
    a = ap.parse_args()
    rng = np.random.default_rng(20260825)
    rows, tries = [], 0
    while len(rows) < a.n and tries < a.n * 6:
        tries += 1
        try:
            r = build(rng, tries, only=a.only)
        except Exception:
            continue
        if r: rows.append(r)
    out = Path(a.out)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} traces from {tries} attempts -> {out}")


if __name__ == "__main__":
    main()
