#!/usr/bin/env python
"""Build a family's scale table, refusing to put two DIFFERENT specs in one column.

WHY: twice on 2026-09-21 a table was wrong because two things shared a label.
  * `:general` resolves to v23 below 20B and v12 at 32B (harness/agents.py), so a ":general" row across sizes is
    two agents, not one.
  * a surv_cox table put `selfver16` (READ-placebo) in the 8B row and `selfver20` (REPLICATE-placebo) at 14B/32B,
    both under the header "placebo". Different devices, same column.
Both were assembled by hand from directory listings. This reads the SPEC out of each file's recorded `agent`
string and refuses to print a column whose spec is not constant, so the error cannot be made silently.

  scale_table.py FAMILY [arm ...]      default arms: control selfver20 selfver general
"""
import sys, os, glob, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_table import score, resolved_spec

fam = sys.argv[1] if len(sys.argv) > 1 else "surv_cox"
arms = sys.argv[2:] or ["control", "selfver20", "selfver", "general"]
ORDER = ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]

cells = collections.defaultdict(dict)          # (model, arm) -> (rate, spec, draw)
for p in glob.glob(f"results/pods/general_agent_*/*_{fam}.json"):
    arm = os.path.basename(p)[: -len(f"_{fam}.json")]
    if arm not in arms:
        continue
    d = os.path.dirname(p)
    if not os.path.isfile(os.path.join(d, f"control_{fam}.json")):
        continue                                # no same-draw control -> not a table row
    try:
        k, n = score(p)
        agent = json.load(open(p)).get("agent", "")
    except Exception:
        continue
    if not n:
        continue
    model, spec = resolved_spec(agent)          # the routing lives in ga_table.resolved_spec now -- see there
    cells[model].setdefault(arm, []).append((100 * k / n, spec, os.path.basename(d)))

print(f"{fam}\n")
hdr = f"{'model':<11}" + "".join(f"{a:>14}" for a in arms)
print(hdr); print("-" * len(hdr))
specs = collections.defaultdict(set)
for model in [m for m in ORDER if m in cells] + [m for m in cells if m not in ORDER]:
    row = f"{model:<11}"
    for a in arms:
        got = cells[model].get(a)
        if not got:
            row += f"{'--':>14}"; continue
        rate = sum(g[0] for g in got) / len(got)
        for _, s, _ in got:
            specs[a].add(s)
        row += f"{rate:>13.1f}%"
    print(row)
print()
bad = {a: s for a, s in specs.items() if len(s) > 1}
if bad:
    print("!! COLUMN(S) NOT COMPARABLE -- the same header covers different specs:")
    for a, s in bad.items():
        print(f"     {a}: {sorted(s)}")
    print("   Split these into separate columns before citing the table.")
else:
    print("every column holds ONE spec:", {a: sorted(s)[0] for a, s in specs.items() if s})
