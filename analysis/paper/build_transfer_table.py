"""Extended Data Table 5 of the v05 manuscript: transfer between domains (RL_PLAN 10-21 16:37 TRANSFER MATRIX, extended 10-24 to
every Table 1 arm). Each arm of Table 1, three runs @100 (the distilled arms at their registered step: B-GEN @575, B-GEN-X2 @775, B-GEN-X3 @800), on all six benchmarks: mean accuracy and the difference
from the untrained model of the same size; the arm's own benchmark in bold. Cells with runs still being read print 'k/3', and
cells with none yet 'pending'. Strict scoring (an unanswered problem is wrong).
    python3 docs/rl4sci/v05/build_transfer_table.py      (writes ed_table5.md; make_v05.py inserts it at [[ed_table5]])
"""
import contextlib
import io
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
with contextlib.redirect_stdout(io.StringIO()):
    import build_table1 as bt   # noqa: E402  (rebuilds Table 1 on import; reuses its loaders)

Q9, Q4 = bt.Q9, bt.Q4


def fam(q, run, step=100):
    return [q.format(f"{run}-s{s}_s{step}") for s in (1, 2, 3)]


# (group, arm, own benchmark, untrained row, dirs)
ARMS = [
    # 10-27 (RL_PLAN 19:00): QD9-FG, MAB9-T8 and QRREAL4 are Table 1 cells since 10-26/27 -- each listed before the arm it replaced
    ("9B + RL", "QD9-FG-L", "qrdata", "Qwen3.5-9B", fam(Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5", 200)),   # 10-29: the 9B QRData cell
    ("9B + RL", "QD9-FG", "qrdata", "Qwen3.5-9B", fam(Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
    ("9B + RL", "SCI9", "qrdata", "Qwen3.5-9B", fam(Q9, "causal_rdd-think-q9base-echo-sysscience-rc-hv3pr0-lr1e-5")),
    ("9B + RL", "GR9-MC", "medcalc", "Qwen3.5-9B", fam(Q9, "medcalc_ver-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    ("9B + RL", "SCI2CLIN9", "ehrsql", "Qwen3.5-9B", fam(Q9, "clinmix-sci9-q9base-echo-sysscience-rc-hv3pr0-lr1e-5")),
    ("9B + RL", "MAB9-T8", "mab", "Qwen3.5-9B", fam(Q9, "mablab27mix_t8-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    ("9B + RL", "MAB9", "mab", "Qwen3.5-9B", fam(Q9, "mablab27mix-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    ("9B + RL", "BDS9-KL", "bds", "Qwen3.5-9B", fam(Q9, "bdsgen-think-q9base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5")),
    ("9B + RL", "BDS9 (first recipe)", "bds", "Qwen3.5-9B", fam(Q9, "bdsgen-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    ("9B + RL", "ASTRO9", "kepler", "Qwen3.5-9B", fam(Q9, "astro_mix-think-q9base-echo-sysscience-rc-32k-hv3pr0-lr1e-5")),
    # 10-28 (RL_PLAN 22:40): B-GEN-X3 (clinical calculation) and B-GEN-X2 (health-record questions) are the distilled cells; both trained
    # on both clinical splits, so both clinical columns are their own; B-GEN, the arm they replaced, stays below them
    ("9B distilled", "B-GEN-X3", "medcalc ehrsql", "Qwen3.5-9B", fam(Q9, "gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 800)),
    ("9B distilled", "B-GEN-X2", "medcalc ehrsql", "Qwen3.5-9B", fam(Q9, "gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 775)),
    ("9B distilled", "B-GEN", "medcalc ehrsql", "Qwen3.5-9B", fam(Q9, "gen_d27u-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 575)),
    ("9B distilled", "B-ASTRO9 †", "kepler", "Qwen3.5-9B", fam(Q9, "kepreal_mix-dastro-q9base-echo-sysscience-rc-32k-hv3pr0-lr1e-5")),
    ("4B + RL", "QRREAL4", "qrdata", "Qwen3.5-4B", fam(Q4, "qrreal89-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
    ("4B + RL", "QM4-KL", "qrdata", "Qwen3.5-4B", fam(Q4, "qrmix-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
    ("4B + RL", "QD4-KL", "qrdata", "Qwen3.5-4B", fam(Q4, "qrdisc-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
    ("4B + RL", "OPD4-MC ‡", "medcalc", "Qwen3.5-4B", fam(Q4, "medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5")),
    ("4B + RL", "EHR4-KL", "ehrsql", "Qwen3.5-4B", fam(Q4, "ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5")),
    ("4B + RL", "MAB4-T8", "mab", "Qwen3.5-4B", fam(Q4, "mablab27mix_t8-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    ("4B + RL", "BDS4-KL", "bds", "Qwen3.5-4B", fam(Q4, "bdsgen-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5")),
    ("4B + RL", "ASTRO4", "kepler", "Qwen3.5-4B", fam(Q4, "astro_mix-think-q4base-echo-sysscience-rc-32k-hv3pr0-lr1e-5")),
    ("4B + RL", "ASTRO4-KL", "kepler", "Qwen3.5-4B", fam(Q4, "astro_mix-think-q4base-echo-sysscience-rc-32k-hv3pr0-kl0.2-lr1e-5")),
    ("4B + RL", "B-ASTRO4 ‡", "kepler", "Qwen3.5-4B", fam(Q4, "kepreal_mix-dastro-q4base-echo-sysscience-rc-32k-hv3pr0-lr1e-5")),
]
M = "−"
hdr = "| Group | Arm | " + " | ".join(bt.NAME[b] for b in bt.BENCHES) + " |"
rows = [hdr, "|:--|:--|" + ":-:|" * len(bt.BENCHES)]
for name in ("Qwen3.5-9B", "Qwen3.5-4B"):
    rows.append(f"| untrained | {name} | " + " | ".join(bt.base_rows[name][b][0] for b in bt.BENCHES) + " |")
done = total = 0
for grp, arm, own, ref, dirs in ARMS:
    cells = []
    for b in bt.BENCHES:
        runs = [x for x in (bt.load(d, b, bt.OPEN_SUB) for d in dirs) if x is not None]
        total += 3
        done += len(runs)
        if not runs:
            cells.append("pending")
            continue
        m = sum(bt.acc(r, b) for r in runs) / len(runs)
        d = m - bt.acc(bt.base_rows[ref][b][1], b)
        c = f"{m:.1f} ({d:+.1f})".replace("-", M) + ("" if len(runs) == 3 else f" {len(runs)}/3")
        cells.append(f"**{c}**" if b in own.split() else c)
    rows.append(f"| {grp} | {arm} | " + " | ".join(cells) + " |")
open(os.path.join(D, "ed_table5.md"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
print(f"runs read: {done} of {total} arm x benchmark x seed cells")
