"""rigor_v05.py (10-28; the 10-27 mock review's concerns 2 and 3; descriptive -- changes no registered verdict). For every Table 1 cell:
  (a) requirement 1 (lift over the untrained model of the same size) with the multiplicity of the cell's search: m = every arm tried in
      the cell with at least one read (Extended Data Table 4) plus its single-run variants ('+ N single-run variants'); the Bonferroni
      interval at 1 - 0.05/m; a two-sided bootstrap p value; and Holm's step-down across the six cells of each row (the family 'RL lifts
      every benchmark at this size');
  (b) for the 9B rows' frontier comparisons (Claude Opus 5.5 high; GPT-6 Sol high and at its most accurate setting): the 90% interval
      and the margins it supports -- non-inferiority (the 9B is not worse by more than max(0, -lower bound)) and equivalence (the
      interval lies inside +-max(|lower|, |upper|); the registered two one-sided tests at 3 points pass iff that is <= 3) -- and the 95%
      interval widened by the untrained models' read-to-read SD for that benchmark (an API score is one read and carries read noise too).
Same hierarchical bootstrap as build_table1 (runs, then problems; 10,000 resamples; seed 20261023).
    python3 docs/rl4sci/v05/rigor_v05.py      -> docs/rl4sci/v05/figdata_rigor_v05.csv (repo root; CPU agent)
"""
import contextlib
import csv
import io
import math
import os
import random
import re
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
with contextlib.redirect_stdout(io.StringIO()):
    import build_table1 as bt      # noqa: E402
    import build_ed_table4 as et   # noqa: E402  (its C lists every arm tried per cell)

N, SEED = 10000, 20261023


def boot(runs, base, b):
    """sorted bootstrap distribution of mean(runs) - base (base = one read)"""
    rng = random.Random(SEED); ids = bt.IDS[b]; out = []
    for _ in range(N):
        rs = [rng.choice(runs) for _ in runs]; ps = [rng.choice(ids) for _ in ids]
        out.append(100 * sum(sum(r.get(i, 0) for r in rs) / len(rs) - base.get(i, 0) for i in ps) / len(ps))
    out.sort()
    return out


def q(dist, a):
    return dist[min(N - 1, max(0, int(a * N)))]


def m_cell(row, b):
    """arms with >= 1 read + single-run variants, from Extended Data Table 4's list for this cell"""
    want = {"9B + RL": "9B + RL", "9B distilled": "9B distilled", "4B + RL": "4B + RL"}[row]
    for cell, base_name, bench, arms in et.C:
        if bench == b and cell.startswith(want):
            m = 0
            for arm, reg, desc, dirs, sel in arms:
                if isinstance(dirs, list) and any(x is not None for x in (bt.load(d, b, bt.OPEN_SUB) for d in dirs)):
                    m += 1
                mm = re.match(r"\+ (\d+) single-run", arm)
                if mm:
                    m += int(mm.group(1))
            return max(m, 1)
    return None   # the distilled row repeats the RL cell here


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); k = len(ps); adj = [0.0] * k; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * ps[i])); adj[i] = run
    return adj


SD = {}
try:
    for r in csv.DictReader(open(os.path.join(D, "figdata_base_reps_v05.csv"))):
        SD.setdefault(r["bench"], []).append(float(r["base_sd"]))
except OSError:
    pass
sd_bench = {b: (sum(v) / len(v) if v else 0.0) for b, v in SD.items()}
claude = {b: bt.load("frontier_api/claude-opus-5-5-high", b, bt.API_SUB) for b in bt.BENCHES}
sol = {k: {b: bt.load(d, b, bt.API_SUB) for b in bt.BENCHES} for k, d in bt.SOL_SETTINGS.items()}
taught = {b: ([bt.TAUGHT9[b]] if b in bt.TAUGHT9 else bt.RL9_LIST[b]) for b in bt.BENCHES}
ROWS = [("9B + RL", bt.RL9_LIST, "Qwen3.5-9B"), ("9B distilled", taught, "Qwen3.5-9B"), ("4B + RL", bt.RL4, "Qwen3.5-4B")]
out = []
for row, spec, ref in ROWS:
    cells = []
    for b in bt.BENCHES:
        disp, runs, arm, mean = bt.cell(spec[b], b)
        if len(runs) < 3:
            print(f"{row} {bt.NAME[b]}: {arm} incomplete ({len(runs)}/3)"); continue
        base = bt.base_rows[ref][b][1]
        dist = boot(runs, base, b); d = mean - bt.acc(base, b)
        m = m_cell(row, b) or m_cell("9B + RL", b) or 1
        p = min(1.0, 2 * min(sum(x <= 0 for x in dist), sum(x >= 0 for x in dist)) / N + 1 / N)
        cells.append(dict(row=row, bench=b, arm=arm, kind="R1 vs untrained", ref=ref, d=round(d, 2), lo95=round(q(dist, 0.025), 2),
                          hi95=round(q(dist, 0.975), 2), m=m, bonf_lo=round(q(dist, 0.025 / m), 2), bonf_hi=round(q(dist, 1 - 0.025 / m), 2),
                          p=p, runs=len(runs)))
        if row != "4B + RL":
            refs = [("Claude Opus 5.5 (high)", claude[b])] + [(f"GPT-6 Sol ({k})", sol[k][b]) for k in ("high",)]
            best = max(((bt.acc(sol[k][b], b), k) for k in sol if sol[k][b] is not None), default=None)
            if best and best[1] != "high":
                refs.append((f"GPT-6 Sol (best: {best[1]})", sol[best[1]][b]))
            for lab, r in refs:
                if r is None: continue
                fd = boot(runs, r, b); dd = mean - bt.acc(r, b)
                lo90, hi90, lo95, hi95 = q(fd, 0.05), q(fd, 0.95), q(fd, 0.025), q(fd, 0.975)
                hw = (hi95 - lo95) / 2; wid = math.sqrt(hw ** 2 + (1.96 * sd_bench.get(b, 0.0)) ** 2)
                out.append(dict(row=row, bench=b, arm=arm, kind="vs frontier", ref=lab, d=round(dd, 2), lo90=round(lo90, 2), hi90=round(hi90, 2),
                                noninf_margin=round(max(0.0, -lo90), 2), equiv_margin=round(max(abs(lo90), abs(hi90)), 2),
                                tost3=max(abs(lo90), abs(hi90)) <= 3, lo95=round(lo95, 2), hi95=round(hi95, 2),
                                lo95_widened=round(dd - wid, 2), hi95_widened=round(dd + wid, 2), runs=len(runs)))
    adj = holm([c["p"] for c in cells])
    for c, a in zip(cells, adj):
        c["holm_p"] = round(a, 4); c["holm_ok"] = a < 0.05; c["bonf_r1"] = c["d"] >= 3 and c["bonf_lo"] > 0
        out.insert(0, c)
keys = sorted({k for r in out for k in r}, key=lambda k: ["row", "bench", "arm", "kind", "ref", "runs", "d"].index(k) if k in
              ["row", "bench", "arm", "kind", "ref", "runs", "d"] else 99)
with open(os.path.join(D, "figdata_rigor_v05.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
for r in out:
    if r["kind"].startswith("R1"):
        print(f"R1  {r['row']:12s} {bt.NAME[r['bench']]:14s} {r['arm']:10s} d {r['d']:+6.1f} [{r['lo95']:+.1f},{r['hi95']:+.1f}] m={r['m']:2d} "
              f"Bonf [{r['bonf_lo']:+.1f},{r['bonf_hi']:+.1f}] {'ok' if r['bonf_r1'] else 'NO'} | Holm p {r['holm_p']:.4f} {'ok' if r['holm_ok'] else 'NO'}")
for r in out:
    if r["kind"] == "vs frontier":
        print(f"FR  {r['row']:12s} {bt.NAME[r['bench']]:14s} {r['arm']:10s} vs {r['ref']:26s} d {r['d']:+6.1f} 90% [{r['lo90']:+.1f},{r['hi90']:+.1f}] "
              f"non-inf {r['noninf_margin']:.1f} equiv {r['equiv_margin']:.1f} TOST3 {'pass' if r['tost3'] else 'fail'} | 95% widened "
              f"[{r['lo95_widened']:+.1f},{r['hi95_widened']:+.1f}]")
print("wrote", os.path.join(D, "figdata_rigor_v05.csv"))
