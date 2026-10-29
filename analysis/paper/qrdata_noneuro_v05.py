"""qrdata_noneuro_v05.py (10-28; the 10-27 mock review's concern 6; descriptive). QRData-Verified without its 20 Neuropathic questions
(15 'Neuropathic' + 5 'Neuropathic_pairwise' multiple-choice items, whose generating family also supplies 76 of the 89 QRData questions
outside the verified set that QRREAL4 / QRREAL9 / QRS4 train on): each system's accuracy on the other 231 questions, and the QRData cells'
lifts and frontier differences there with the hierarchical bootstrap (runs, then problems; 10,000 resamples; seed 20261023).
    python3 docs/rl4sci/v05/qrdata_noneuro_v05.py    -> docs/rl4sci/v05/figdata_qrdata_noneuro_v05.csv  (repo root; CPU agent)
"""
import contextlib
import csv
import io
import json
import os
import random
import re
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
with contextlib.redirect_stdout(io.StringIO()):
    import build_table1 as bt   # noqa: E402
B = "qrdata"
P = os.path.join(D, "../../../benchmarks/qrdata_verified/problems")
fam = {}
for k in bt.IDS[B]:
    c = json.load(open(f"{P}/{k}/eval_config.json"))
    fam[k] = re.sub(r"[_\-]?\d+$", "", os.path.splitext((c.get("data_files") or ["?"])[0])[0])
NEURO = {k for k, f in fam.items() if f.startswith("Neuropathic")}
KEEP = [k for k in bt.IDS[B] if k not in NEURO]
assert len(NEURO) == 20, len(NEURO)


def acc(x, ids):
    return 100 * sum(x.get(i, 0) for i in ids) / len(ids)


def ci(runs, base, ids, n=10000, seed=20261023):
    rng = random.Random(seed); out = []
    for _ in range(n):
        rs = [rng.choice(runs) for _ in runs]; ps = [rng.choice(ids) for _ in ids]
        out.append(100 * sum(sum(r.get(i, 0) for r in rs) / len(rs) - base.get(i, 0) for i in ps) / len(ps))
    out.sort(); return out[int(0.025 * n)], out[int(0.975 * n)]


def runs_of(q, run, step=100):
    return [x for x in (bt.load(d, B, bt.OPEN_SUB) for d in bt.seeds(q, run, step)) if x is not None]


SYS = {"Qwen3.5-9B (untrained)": [bt.load("harness_v3_q9think/V3/ext", B, bt.OPEN_SUB)],
       "Qwen3.5-4B (untrained)": [bt.load("harness_v3_q4think/V3/ext", B, bt.OPEN_SUB)],
       "Claude Opus 5.5 (high)": [bt.load("frontier_api/claude-opus-5-5-high", B, bt.API_SUB)],
       "GPT-6 Sol (high)": [bt.load("frontier_api/gpt-6-sol-high", B, bt.API_SUB)],
       "GPT-6 Sol (low)": [bt.load("frontier_api/gpt-6-sol-low", B, bt.API_SUB)],
       "QD9-FG (9B, @100)": runs_of(bt.Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"),
       "QD9-FG-L (9B, @200)": runs_of(bt.Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5", 200),
       "QRREAL9 (9B)": runs_of(bt.Q9, "qrreal89-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"),
       "QRREAL4 (4B cell)": runs_of(bt.Q4, "qrreal89-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")}
rows = []
for name, xs in SYS.items():
    xs = [x for x in xs if x is not None]
    if not xs: continue
    rows.append(dict(system=name, runs=len(xs), all_251=round(sum(acc(x, bt.IDS[B]) for x in xs) / len(xs), 1),
                     without_neuro_231=round(sum(acc(x, KEEP) for x in xs) / len(xs), 1),
                     neuro_20=round(sum(acc(x, sorted(NEURO)) for x in xs) / len(xs), 1)))
    print(f"{name:26s} runs {len(xs)} | all {rows[-1]['all_251']:5.1f} | without Neuropathic {rows[-1]['without_neuro_231']:5.1f} | Neuropathic {rows[-1]['neuro_20']:5.1f}")
pairs = [("QRREAL4 (4B cell)", "Qwen3.5-4B (untrained)"), ("QD9-FG (9B, @100)", "Qwen3.5-9B (untrained)"), ("QD9-FG-L (9B, @200)", "Qwen3.5-9B (untrained)"),
         ("QD9-FG (9B, @100)", "Claude Opus 5.5 (high)"), ("QD9-FG-L (9B, @200)", "Claude Opus 5.5 (high)"), ("QD9-FG-L (9B, @200)", "GPT-6 Sol (low)"),
         ("QRREAL9 (9B)", "Qwen3.5-9B (untrained)")]
for a, r in pairs:
    xa = [x for x in SYS[a] if x is not None]; xr = SYS[r][0]
    if len(xa) < 2 or xr is None: continue
    for lab, ids in (("all 251", bt.IDS[B]), ("without Neuropathic (231)", KEEP)):
        d = sum(acc(x, ids) for x in xa) / len(xa) - acc(xr, ids); lo, hi = ci(xa, xr, ids)
        rows.append(dict(system=f"{a} - {r}", runs=len(xa), set=lab, d=round(d, 1), lo=round(lo, 1), hi=round(hi, 1)))
        print(f"{a} - {r} [{lab}]: {d:+.1f} ({lo:+.1f} to {hi:+.1f}) ({len(xa)} runs)")
keys = ["system", "runs", "all_251", "without_neuro_231", "neuro_20", "set", "d", "lo", "hi"]
with open(os.path.join(D, "figdata_qrdata_noneuro_v05.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote figdata_qrdata_noneuro_v05.csv")
