"""Fig. 2 of the v05 manuscript: reinforcement learning lifts both model sizes on the six benchmarks, and a teacher trained
the same way brings the 9B level with GPT-6 Sol. Data: figdata_v05.csv (written by build_table1.py; every difference and
hierarchical 95% CI shown in requirements.md). Style: ../v04/figstyle.py (models trained in this work are blue circles).
    python3 docs/rl4sci/v05/make_fig2.py
"""
import csv
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "v04"))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from figstyle import API, BAND, BASE, INK, MUTED, OURS, OURS_LIGHT, W2, MM, letter  # noqa: E402

rows = list(csv.DictReader(open(f"{D}/figdata_v05.csv")))
get = {(r["row"], r["ref"], r["bench"]): r for r in rows}
BENCH = [("qrdata", "QRData"), ("medcalc", "MedCalc"), ("ehrsql", "EHRSQL"), ("mab", "MedAgentBench"),
         ("bds", "BioDSBench"), ("kepler", "Kepler (real)")]
R9, R9T, R4 = "9B + RL (ours)", "9B distilled from an RL-trained 27B (ours)", "4B + RL (ours)"
REPLACING = {("4B + RL (ours)", "qrdata"): "pending: QRREAL4-L"}   # 10-28: QRREAL4 shown until QRREAL4-L (step 200) is read
TEACHER = {"medcalc", "ehrsql", "kepler"}   # build_table1.TAUGHT9: elsewhere the taught row only repeats the RL-alone value


def series(ax, specs, ref, xlim):
    """specs: (row label, legend label, face colour, edge colour, offset). One row per benchmark, one marker per spec."""
    ax.axvspan(-3, 3, color=BAND, zorder=0, linewidth=0)
    ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    for i, (b, _) in enumerate(BENCH):
        live = [sp for sp in specs if sp[0] != R9T or b in TEACHER]
        for row, _, fc, ec, off in live:
            r = get.get((row, ref, b))
            y = i + (off if len(live) > 1 else 0)
            if r is None:
                ax.text(xlim[0] + 1, y, "pending (training)", fontsize=5.2, color=MUTED, va="center", ha="left")
                continue
            d, lo, hi, n = float(r["diff"]), float(r["ci_lo"]), float(r["ci_hi"]), int(r["runs"])
            ax.plot([lo, hi], [y, y], color=ec, linewidth=0.9, zorder=2, solid_capstyle="butt")
            ax.scatter(d, y, marker="o", s=15, facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=3)
            if n < 3:   # still training or reading: say how many of the three runs are in
                ax.text(hi + 1.0, y, f"{n}/3 runs", fontsize=5.0, color=MUTED, va="center", ha="left")
            elif (row, b) in REPLACING:   # 10-24 (mock review): a cell whose registered replacement arm is still training
                ax.text(hi + 1.0, y, REPLACING[(row, b)], fontsize=5.0, color=MUTED, va="center", ha="left")
    ax.set_yticks(range(len(BENCH)))
    ax.set_yticklabels([name for _, name in BENCH], fontsize=6)
    ax.set_ylim(len(BENCH) - 0.45, -0.55)
    ax.set_xlim(*xlim)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="-", linewidth=0.9, color=ec, markerfacecolor=fc,
                              markeredgecolor=ec, markersize=4, label=lab) for _, lab, fc, ec, _ in specs],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=5.8, ncol=len(specs), handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.2)


fig = plt.figure(figsize=(W2, 72 * MM))
gs = GridSpec(1, 2, figure=fig, wspace=0.08, left=0.10, right=0.99, top=0.84, bottom=0.17)

# a: trained minus untrained, same size (requirement: at least +3 points with the interval above zero)
ax = fig.add_subplot(gs[0, 0])
series(ax, [(R4, "4B + RL", "white", OURS, -0.14), (R9, "9B + RL", OURS, OURS, 0.14)], "untrained", (-22, 48))
ax.set_xlabel("Accuracy minus the untrained model of the same size (points)")
letter(fig, 0.005, 0.99, "a")

# b: the 9B minus GPT-6 Sol at high reasoning effort, with and without the teacher
ax = fig.add_subplot(gs[0, 1])
series(ax, [(R9, "9B + RL", OURS, OURS, -0.14), (R9T, "9B distilled from the RL-trained 27B", OURS_LIGHT, OURS, 0.14)],
       "GPT-6 Sol (high)", (-40, 48))
ax.set_yticklabels([])
ax.set_xlabel("Accuracy minus GPT-6 Sol at high reasoning effort (points)")
letter(fig, 0.545, 0.99, "b")

for ext in ("pdf", "png"):
    fig.savefig(f"{D}/fig2.{ext}", bbox_inches="tight", pad_inches=0.02)
print("wrote", f"{D}/fig2.pdf/png")
