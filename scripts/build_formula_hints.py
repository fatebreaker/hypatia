"""Per-calculator FORMULA HINTS for hint-guided RL (docs/RL_PLAN.md H-HINT9, 2026-10-08).

From MedCalc-Bench-Verified's TRAIN split only (the cached train_data.csv that scripts/build_medcalc_train.py draws the RL
pool from), each calculator's hint is the text its reference explanations SHARE: the explanation of every row opens with
the calculator's formula or scoring criteria and then applies it to that row's patient. A sentence (or line) is kept when
it occurs in at least --min-share of the calculator's rows (and in at least 2 rows), in the order of the row that holds
the most of them -- so the formula / criteria survive and no patient's values do. Capped at --max-chars.

    venv/bin/python scripts/build_formula_hints.py --out results/pods/hints/medcalc_formula_hints.json

Writes {"source": ..., "min_share": ..., "hints": {calculator name: hint text}} and prints, per calculator, the rows used
and the hint length; calculators with no shared text get no hint (listed).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN = ("nsk7153/MedCalc-Bench-Verified", "58fc3de568bfc53a68f32327e58f60dc6eaa17f8", "train_data.csv")
_SPLIT = re.compile(r"(?<=[.!?])(?<!\b\d\.)(?<!\b\d\d\.)\s+(?=[A-Z(\[])|\n+")   # "1. Hypertension" stays one unit


def units(expl: str) -> list[str]:
    """The explanation's sentences / lines, whitespace-normalised (a decimal point never splits: no space follows it)."""
    return [u for u in (" ".join(p.split()) for p in _SPLIT.split(expl or "")) if u]


def hint_for(explanations: list[str], min_share: float, max_chars: int) -> str:
    rows = [units(e) for e in explanations if (e or "").strip()]
    if len(rows) < 2:
        return ""
    seen = Counter(u for r in rows for u in set(r))
    keep = {u for u, n in seen.items() if n >= 2 and n >= min_share * len(rows)}
    if not keep:
        return ""
    best = max(rows, key=lambda r: sum(u in keep for u in r))      # the row that holds the most shared units
    out, used = [], set()
    for u in best:
        if u in keep and u not in used:
            out.append(u); used.add(u)
    text = "\n".join(out)
    return text[:max_chars].rstrip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="the TRAIN csv (default: scripts/fetch_medcalc.fetch of the pinned revision)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-share", type=float, default=0.8)
    ap.add_argument("--max-chars", type=int, default=4000)
    a = ap.parse_args()
    if a.csv:
        src = Path(a.csv)
    else:
        from fetch_medcalc import fetch  # noqa: E402  (the same cached file build_medcalc_train.py reads)
        src = Path(fetch(*TRAIN))
    assert "train" in src.name, f"{src}: hints come from the TRAIN split only"
    csv.field_size_limit(10 ** 9)
    by_calc = defaultdict(list)
    with open(src, newline="") as fh:
        for row in csv.DictReader(fh):
            by_calc[row["Calculator Name"].strip()].append(row.get("Ground Truth Explanation") or "")
    hints, none = {}, []
    for calc in sorted(by_calc):
        h = hint_for(by_calc[calc], a.min_share, a.max_chars)
        if h:
            hints[calc] = h
        else:
            none.append(calc)
        print(f"{len(by_calc[calc]):5d} rows  {len(h):5d} chars  {calc}")
    print(f"{len(hints)} of {len(by_calc)} calculators have a hint; none for: {none}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({"source": str(src), "split": "train", "min_share": a.min_share, "max_chars": a.max_chars,
                               "hints": hints}, indent=1))
    os.replace(tmp, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
