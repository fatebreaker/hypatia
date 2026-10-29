#!/usr/bin/env python
"""Re-read every same-draw comparison in the tree BY ABSTRACT MODE instead of by family.

This is the payoff of `trap_modes.py`: no new measurement, a different question. Per-family the programme could
only say "the scaffold causes the trap on causal_confounded and reduces it on four others". Per-mode it can ask
whether that pattern tracks the KIND of error, which is the thing that could transfer.

Reuses `trap_sweep.py`'s output so the exhaustion filter, the same-draw rule and the gated scorer all still apply:
rows with a >=10pt non-termination gap are unreadable and are dropped here too.
"""
import re, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trap_modes import mode_of, MODES

PLACEBOS = {"selfver20", "selfver16", "selfver25", "selfver24"}
CONTENT = {"selfver12", "general", "selfver17", "selfver21", "selfver23", "ident"}

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad/trapsweep.txt"
lines = open(path).read().split("\n")

rows = []
for i, l in enumerate(lines):
    m = re.match(r"^(\S+) / (\S+?)(\*?)\s+([\d.]+)% v\s+([\d.]+)%\s+(\S+)\s+([\d.]+)% v\s+([\d.]+)%\s+(\S+)\s+([\d.]+)/([\d.]+)", l)
    if not m:
        continue
    fam, arm, _, pa, pc, pp, ta, tc, pt, na, nc = m.groups()
    meta = re.search(r"\s(\S+)\s+\[(\S*)\]\s+trap=(\S+)", lines[i+1] if i+1 < len(lines) else "")
    if not meta:
        continue
    gap = abs(float(na) - float(nc))
    if gap >= 10:                        # unreadable, as everywhere else
        continue
    mode = mode_of(meta.group(3))
    if not mode:
        continue
    kind = "placebo" if arm in PLACEBOS else ("content" if arm in CONTENT else None)
    if kind is None:
        continue
    rows.append(dict(mode=mode, fam=fam, arm=arm, kind=kind, model=meta.group(2),
                     trap=meta.group(3), p=float(pt),
                     sign=("CAUSES" if float(ta) > float(tc) else "reduces"),
                     sig=float(pt) < 0.05, gap=gap, draw=meta.group(1)))

print(f"{len(rows)} exhaustion-clean, mode-mapped comparisons\n")
for kind in ("placebo", "content"):
    print(f"=== {kind.upper()} arms, by mode ===")
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["kind"] != kind:
            continue
        by[r["mode"]][r["sign"] if r["sig"] else "ns"].append(r)
    for mode in sorted(by, key=lambda m: -(len(by[m]["CAUSES"]) + len(by[m]["reduces"]))):
        c, red, ns = by[mode]["CAUSES"], by[mode]["reduces"], by[mode]["ns"]
        fams = sorted({r["fam"] for r in c + red})
        print(f"  {mode:<20} CAUSES {len(c):>2}  reduces {len(red):>2}  ns {len(ns):>2}   "
              f"significant in: {', '.join(fams) if fams else '-'}")
    print()
print("MODES:")
for k, v in MODES.items():
    print(f"  {k:<20} {v}")
