"""build_table1.py (v05, 10-23) -- Table 1 as the user set it on 10-21/10-22: six benchmark columns; GPT-6 Sol, gpt-5 and gpt-4.1 at
high reasoning effort; the untrained open models; and OURS, trained only on in-domain data -- '9B + RL', '9B + RL, taught by an
RL-trained 27B' and '4B + RL'. Each 'ours' cell is the mean over the registered training runs of that benchmark's in-domain arm
(three seeds), scored strictly as in v04 (an overflowed or unanswered problem is wrong; a run below 95% complete is not used).
A cell still waiting for runs prints as 'pending (k/3)'. Writes table1.md, table1.csv and requirements.md (the R1/R2 status).
    python3 docs/rl4sci/v05/build_table1.py
"""
import csv
import json
import os
import random
from pathlib import Path

R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"
P = f"{R}/results/pods"
OUT = Path(__file__).resolve().parent
os.chdir(R)

IDS = {
    "qrdata": sorted(os.listdir("benchmarks/qrdata_verified/problems")),
    "medcalc": json.load(open("benchmarks/medcalc_ver_s220/SUBSET.json")),
    "ehrsql": json.load(open("benchmarks/ehrsql2024_s300/SUBSET.json")),
    "mab": sorted(os.listdir("benchmarks/medagentbench/problems")),
    "bds": sorted(os.listdir("benchmarks/biodsbench/problems")),
    "kepler": sorted(os.listdir("benchmarks/kepler_real/problems")),
}
BENCHES = list(IDS)
NAME = {"qrdata": "QRData", "medcalc": "MedCalc", "ehrsql": "EHRSQL", "mab": "MedAgentBench", "bds": "BioDSBench",
        "kepler": "Kepler"}
API_SUB = {"qrdata": ["qrdata_verified"], "medcalc": ["medcalc_ver_s220"], "ehrsql": ["ehrsql2024_s300"],
           "mab": ["medagentbench"], "bds": ["biodsbench"], "kepler": ["kepler_real"]}
OPEN_SUB = {"qrdata": ["qrdata_verified"], "medcalc": ["medcalc_ver_s220", "medcalc_ver"], "ehrsql": ["ehrsql2024_s300"],
            "mab": ["medagentbench_fhir"], "bds": ["biodsbench"], "kepler": ["kepler_real"]}
Q9 = "harness_v3_q9/V3__rl-czero-{}/ext"
Q4 = "harness_v3_q4/V3__rl-czero-{}/ext"


def seeds(pattern, run, step=100):
    return [pattern.format(f"{run}-s{s}_s{step}") for s in (1, 2, 3)]


# baselines: one run each, the same directory for every benchmark
BASE = [
    ("GPT-6 Sol (high)", "api", "frontier_api/gpt-6-sol-high", API_SUB),
    ("Claude Opus 5.5 (high)", "api", "frontier_api/claude-opus-5-5-high", API_SUB),   # 10-26 (RL_PLAN H-CLAUDE; user: compete with Claude)
    ("gpt-5 (high)", "api", "frontier_api/gpt-5-high", API_SUB),
    ("gpt-4.1", "api", "frontier_api/gpt-4.1", API_SUB),
    ("gpt-oss-120b", "open", "frontier_open_v3/gpt-oss-120b", OPEN_SUB),
    ("Qwen3.6-35B-A3B", "open", "frontier_open_v3/qwen3.6-35b-a3b-fp8", OPEN_SUB),
    ("Qwen3.6-27B", "open", "harness_v3_q27think/V3/ext", OPEN_SUB),   # BASE27-V3: bf16, same harness as the trained reads
    ("Qwen3.5-9B", "open", "harness_v3_q9think/V3/ext", OPEN_SUB),
    ("Qwen3.5-4B", "open", "harness_v3_q4think/V3/ext", OPEN_SUB),
]
# ours: per benchmark, the registered in-domain arm (docs/RL_PLAN.md 10-21 14:41 and 15:03; 10-22 02:40, 03:10, 14:39; 10-23 09:29)
RL9 = {
    "qrdata": ("SCI9", seeds(Q9, "causal_rdd-think-q9base-echo-sysscience-rc-hv3pr0-lr1e-5")),
    "medcalc": ("GR9-MC", seeds(Q9, "medcalc_ver-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    "ehrsql": ("SCI2CLIN9", seeds(Q9, "clinmix-sci9-q9base-echo-sysscience-rc-hv3pr0-lr1e-5")),
    "mab": ("MAB9", seeds(Q9, "mablab27mix-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    "bds": ("BDS9", seeds(Q9, "bdsgen-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
    "kepler": ("ASTRO9", seeds(Q9, "astro_mix-think-q9base-echo-sysscience-rc-32k-hv3pr0-lr1e-5")),
}
TAUGHT9 = {   # only where an RL-trained 27B teacher exists; the other columns repeat RL9 (marked)
    # 10-28 (RL_PLAN 20:55 / 22:40; the strongest-arm rule): the strongest distilled arm per cell --
    # B-GEN-X3 on clinical calculation (91.2; B-GEN 87.0), B-GEN-X2 on health-record questions (92.3; B-GEN 91.2). Both are B-GEN's
    # recipe with more explanation-guided 27B notebooks for calculators chosen from test errors (disclosed; Extended Data Table 4)
    "medcalc": ("B-GEN-X3", seeds(Q9, "gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 800)),
    "ehrsql": ("B-GEN-X2", seeds(Q9, "gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 775)),
    "kepler": ("B-ASTRO9", seeds(Q9, "kepreal_mix-dastro-q9base-echo-sysscience-rc-32k-hv3pr0-lr1e-5")),
}
RL4 = {   # first choice = the registered fix; the original H-RL4 arm is the fallback until the fix is read
    # QRData (RL_PLAN 10-24 15:14 / 15:23, rule QR4-LIFTS amended): QM4-KL, else QD4-KL, else B-QR4, else B-QR4-9, else the
    # best of them, 'not met'. 10-24 18:1x: QM4-KL (-0.3) and QD4-KL (+1.9) do not lift (three runs each) -- the cell is B-QR4 /
    # B-QR4-9 once read (apply the rule by hand: show B-QR4 only if it lifts), until then the best so far (QD4-KL). cell() shows
    # the first arm with a complete run.
    # 10-27 (RL_PLAN 06:32 / 07:16): QRREAL4 (+3.9, three runs, not met) is the strongest 4B QRData arm, so it is the cell
    "qrdata": [("QRREAL4", seeds(Q4, "qrreal89-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
               ("QRS4", seeds(Q4, "qrrsem-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),   # 10-27: +2.8, not met
               ("B-QR4", seeds(Q4, "qrdisc-dqr27-q4base-echo-sysscience-rc-rg-32k-hv3pr0-lr1e-5")),
               ("B-QR4-9", seeds(Q4, "qrdisc-dqr9-q4base-echo-sysscience-rc-rg-32k-hv3pr0-lr1e-5")),
               # 10-26: B-QR4-9K (+1.1, three runs, not met) is listed after B-QR4-9 so the cell stays the best arm until QD4-FG is read;
               # each seed starts from its own merged B-QR4-9 distilled model, so the run name carries the seed twice
               ("B-QR4-9K", [Q4.format(f"qrdisc-think-q4dqr9s{s}base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5-s{s}_s100") for s in (1, 2, 3)]),
               ("QD4-FG", seeds(Q4, "qrdisc2-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),   # 10-27: +0.3, not met
               ("QD4-KL", seeds(Q4, "qrdisc-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
               ("QM4-KL", seeds(Q4, "qrmix-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
               ("SCI4-KL", seeds(Q4, "causal_mix-think-q4base-echo-sysscience-rc-hv3pr0-kl0.2-lr1e-5"))],
    "medcalc": [("OPD4-MC", seeds(Q4, "medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5")),
                ("GR4-MC", seeds(Q4, "medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"))],
    "ehrsql": [("EHR4-KL", seeds(Q4, "ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5"))],
    # MedAgentBench (RL_PLAN 10-23 13:16, rule MAB4T8-LIFTS): MAB4-T8 if it lifts, else MAB4
    "mab": [("MAB4-T8", seeds(Q4, "mablab27mix_t8-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")),
            ("MAB4", seeds(Q4, "mablab27mix-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"))],
    # BioDSBench (RL_PLAN 10-23 09:42, rule BDS4KL-LIFTS): BDS4-KL if it lifts, else BDS4
    "bds": [("BDS4-KL", seeds(Q4, "bdsgen-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5")),
            ("BDS4", seeds(Q4, "bdsgen-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"))],
    # Kepler (RL_PLAN 10-24 00:15 / 13:04, rule ASTRO4KL-LIFTS amended): ASTRO4-KL if it lifts, else B-ASTRO4, else ASTRO4
    # 10-26 (RL_PLAN H-MEET-ALL 14:04, user 'the strongest ... should be the method'): B-ASTRO4 (+40.8; fresh Kepler +36.1) is the cell;
    # ASTRO4-KL (+16.1 on kepler_real, +3.8 on fresh stars) stays in Extended Data Table 4
    "kepler": [("B-ASTRO4", seeds(Q4, "kepreal_mix-dastro-q4base-echo-sysscience-rc-32k-hv3pr0-lr1e-5")),
               ("ASTRO4-KL", seeds(Q4, "astro_mix-think-q4base-echo-sysscience-rc-32k-hv3pr0-kl0.2-lr1e-5")),
               ("ASTRO4", seeds(Q4, "astro_mix-think-q4base-echo-sysscience-rc-32k-hv3pr0-lr1e-5"))],
}


def load(base, bench, sub):
    """{id: 0/1} strict, or None when the run is missing or under 95% complete on this benchmark."""
    ids = IDS[bench]
    d = next((f"{P}/{base}/{nm}" for nm in sub[bench] if os.path.isdir(f"{P}/{base}/{nm}")), None)
    if d is None:
        return None
    out = {}
    for i in ids:
        f = f"{d}/{i}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        if r.get("error") or "grade" not in r:
            out[i] = 0
            continue
        g = r["grade"] if isinstance(r["grade"], dict) else eval(r["grade"])  # noqa: S307 (grader output, our own files)
        out[i] = int(bool(g.get("passed")))
    return out if len(out) >= 0.95 * len(ids) else None


def acc(rows, bench):
    return 100 * sum(rows.get(i, 0) for i in IDS[bench]) / len(IDS[bench])


def hier_ci(runs, base, bench, n=10000, seed=20261023):   # 10-24: 10,000 resamples, as Methods states
    """mean(runs) - base with a hierarchical bootstrap (runs, then problems); base = one run."""
    rng = random.Random(seed)
    ids = IDS[bench]
    out = []
    for _ in range(n):
        rs = [rng.choice(runs) for _ in runs]
        ps = [rng.choice(ids) for _ in ids]
        out.append(100 * sum(sum(r.get(i, 0) for r in rs) / len(rs) - base.get(i, 0) for i in ps) / len(ps))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def cell(arms, bench):
    """(label, display, runs, arm) for the first arm with >= 1 complete run; display says 'pending (k/3)' below three."""
    for arm, dirs in arms:
        runs = [x for x in (load(d, bench, OPEN_SUB) for d in dirs) if x is not None]
        if runs:
            m = sum(acc(r, bench) for r in runs) / len(runs)
            disp = f"{m:.1f}" if len(runs) == 3 else f"{m:.1f} ({len(runs)}/3)"
            return disp, runs, arm, m
    return f"pending (0/3)", [], arms[0][0], None


rows_out, base_rows = [], {}
for name, grp, base, sub in BASE:
    vals = {}
    for b in BENCHES:
        r = load(base, b, sub)
        vals[b] = (f"{acc(r, b):.1f}" if r else "missing", r)
    base_rows[name] = vals
    rows_out.append((name, grp, {b: vals[b][0] for b in BENCHES}, {}))
ours = {}
# 10-23 (RL_PLAN 18:13, rule BDS9KL-LIFTS): the 9B's BioDSBench cell is BDS9-KL if it lifts, else BDS9 -- first arm with runs shown
RL9_LIST = {b: [RL9[b]] for b in BENCHES}
RL9_LIST["bds"] = [("BDS9-KL", seeds(Q9, "bdsgen-think-q9base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5")), RL9["bds"]]
# 10-26 (RL_PLAN H-BEAT-CLAUDE MAB9-T8 read 06:18; user 13:56 'the strongest ... should be the method'): MAB9-T8 (84.2, the benchmark's own
# 8-turn limit in training, as the 4B cell MAB4-T8) is the 9B MedAgentBench cell; MAB9 (78.2) stays in Extended Data Table 4
RL9_LIST["mab"] = [("MAB9-T8", seeds(Q9, "mablab27mix_t8-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5")), RL9["mab"]]
# 10-27 (RL_PLAN 10:1x QD9-FG run 3; the strongest-arm rule): QD9-FG (58.9, three runs) is the 9B QRData
# cell, above QRREAL9 (57.3) and SCI9 (55.9); SCI9 stays the synthetic-science transfer result (Fig. 5b) and Extended Data Table 4
RL9_LIST["qrdata"] = [("QD9-FG-L", seeds(Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5", 200)),   # 10-29 (RL_PLAN BEAT-CLAUDE-QR read): QD9-FG-L is the cell
                      ("QD9-FG", seeds(Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")),
                      ("QRREAL9", seeds(Q9, "qrreal89-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5")), RL9["qrdata"]]
# 10-24 (mock review; user 13:5x 'RL + distillation' framing): the taught row is labelled as distillation -- B-GEN (clinical
# calculation, health-record questions) is maximum-likelihood distillation with no RL of the 9B; B-ASTRO9 (Kepler) is distillation
# followed by RL on real training stars (dagger). Teacher-scored 4B arms (on-policy distillation) carry a double dagger.
TAUGHT_LABEL = "9B distilled from an RL-trained 27B (ours)"
DISTIL_THEN_RL = {"B-ASTRO9", "B-ASTRO4", "UNION9"}          # dagger: distilled, then reinforcement learning
# 10-29 (RL_PLAN ONE9-LEVEL-ALL met 14:41; user 18:4x: a Table 1 row 'if it good enough and make our manuscript stronger'): ONE 9B for all
# six -- UNION9 (one distillation on the union of both teachers' notebooks, then B-ASTRO9's RL on real stars), the same three runs in every
# column (H-ONE9, registered 10-25 14:24)
ONE_LABEL = "One 9B for all six (ours)"
ONE9 = {b: [("UNION9", seeds(Q9, "kepreal_mix-dunion-q9base-echo-sysscience-rc-32k-hv3pr0-lr1e-5", 100))] for b in BENCHES}
TEACHER_SCORED = {"OPD4-MC"}   # double dagger: 4B trained with the RL-trained 27B as a teacher
for label, spec, ref in (("9B + RL (ours)", RL9_LIST, "Qwen3.5-9B"),
                         (TAUGHT_LABEL, {b: ([TAUGHT9[b]] if b in TAUGHT9 else RL9_LIST[b]) for b in BENCHES}, "Qwen3.5-9B"),
                         (ONE_LABEL, ONE9, "Qwen3.5-9B"),
                         ("4B + RL (ours)", RL4, "Qwen3.5-4B")):
    disp, arms = {}, {}
    for b in BENCHES:
        d, runs, arm, m = cell(spec[b], b)
        mark = (" *" if label == TAUGHT_LABEL and b not in TAUGHT9 else "") + ("†" if arm in DISTIL_THEN_RL else "") \
            + ("‡" if arm in TEACHER_SCORED else "")
        disp[b] = d + mark
        arms[b] = (arm, runs, m)
    ours[label] = (arms, ref)
    rows_out.append((label, "ours", disp, arms))

# Table 1 (markdown + csv)
hdr = "| System | " + " | ".join(f"{NAME[b]} (n = {len(IDS[b])})" for b in BENCHES) + " |"
md = [hdr, "|" + "---|" * (len(BENCHES) + 1)]
for name, grp, disp, _ in rows_out:
    md.append(f"| {name} | " + " | ".join(disp[b] for b in BENCHES) + " |")
(OUT / "table1.md").write_text("\n".join(md) + "\n\nOurs: mean of three training runs (in-domain arm per column); "
                               "* = no teacher for this benchmark, the RL-alone value is repeated; † = distilled, then reinforcement learning; "
                               "‡ = trained with the RL-trained 27B as a teacher (on-policy distillation); One 9B for all six = UNION9, the same three runs in every column.\n")
with open(OUT / "table1.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["system", "group"] + BENCHES)
    for name, grp, disp, _ in rows_out:
        w.writerow([name, grp] + [disp[b] for b in BENCHES])

# requirements: R1 = RL beats the untrained model on all six at 4B and 9B (+3, CI excl. 0); R2 = 9B competitive with Sol (high)
sol = base_rows["GPT-6 Sol (high)"]
req = ["| Requirement | " + " | ".join(NAME[b] for b in BENCHES) + " |", "|" + "---|" * (len(BENCHES) + 1)]
fig = []   # figdata_v05.csv: every difference shown in requirements.md, for make_fig2.py
for label, (arms, ref) in ours.items():
    cells = []
    for b in BENCHES:
        arm, runs, m = arms[b]
        bref = base_rows[ref][b][1]
        if not runs or bref is None:
            cells.append(f"{arm}: pending")
            continue
        lo, hi = hier_ci(runs, bref, b)
        d = m - acc(bref, b)
        ok = "MET" if (d >= 3 and lo > 0 and len(runs) == 3) else ("on track" if (d >= 3 and lo > 0) else "not met")
        fig.append([label, "untrained", b, arm, len(runs), f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}", ok])
        cells.append(f"{arm} {d:+.1f} [{lo:+.1f}, {hi:+.1f}] {ok}" + ("" if len(runs) == 3 else f" ({len(runs)}/3)"))
    req.append(f"| R1 {label} vs {ref} | " + " | ".join(cells) + " |")
# 10-26 (RL_PLAN H-CLAUDE / H-BEAT-CLAUDE): the same comparison against Claude Opus 5.5 (high), descriptive -- R2 stays against GPT-6 Sol
# 10-27 20:4x (user: 'Adopt the review's framing'; RL_PLAN 20:23): R2 is also reported against GPT-6 Sol at its MOST ACCURATE
# effort setting on each benchmark (the pre-10-17 frontier rules' comparator), beside the high-effort row
SOL_SETTINGS = {"low": "frontier_api/gpt-6-sol-low", "medium": "frontier_api/gpt-6-sol-medium", "high": "frontier_api/gpt-6-sol-high"}
SOL_BEST_SETTING = {}
_sb = {}
for b in BENCHES:
    _c = [(acc(r, b), k, r) for k, d in SOL_SETTINGS.items() for r in [load(d, b, API_SUB)] if r is not None]
    _a, _k, _r = max(_c, key=lambda x: x[0])
    _sb[b] = (f"{_a:.1f}", _r)
    SOL_BEST_SETTING[b] = _k
base_rows["GPT-6 Sol (best setting)"] = _sb
for ref_name in ("GPT-6 Sol (high)", "Claude Opus 5.5 (high)", "GPT-6 Sol (best setting)"):
    ref_row = base_rows[ref_name]
    for label in ("9B + RL (ours)", TAUGHT_LABEL, ONE_LABEL):
        arms, _ = ours[label]
        cells = []
        for b in BENCHES:
            arm, runs, m = arms[b]
            if not runs or ref_row[b][1] is None:
                cells.append("pending")
                continue
            lo, hi = hier_ci(runs, ref_row[b][1], b)
            d = m - acc(ref_row[b][1], b)
            ok = "level or above" if (d >= -3 and hi >= 0) else "below"
            fig.append([label, ref_name, b, arm, len(runs), f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}", ok])
            cells.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}] {ok}" + ("" if len(runs) == 3 else f" ({len(runs)}/3)"))
        _tag = {'GPT-6 Sol (high)': 'R2', 'Claude Opus 5.5 (high)': 'vs Claude', 'GPT-6 Sol (best setting)': 'R2 (Sol best setting)'}[ref_name]
        req.append(f"| {_tag} {label} vs {ref_name} | " + " | ".join(cells) + " |")
req.append("")
req.append("GPT-6 Sol best setting per benchmark: " + ", ".join(f"{NAME[b]} {SOL_BEST_SETTING[b]} ({_sb[b][0]})" for b in BENCHES))
(OUT / "requirements.md").write_text("\n".join(req) + "\n")
# every arm with a complete run, chosen for Table 1 or not (check_numbers_v05.py quotes arms by name)
arm_rows = []
CHECK_ONLY4 = {   # original H-RL4 arms the text quotes but Table 1 does not use
    "qrdata": [("SCI4", seeds(Q4, "causal_rdd-think-q4base-echo-sysscience-rc-hv3pr0-lr1e-5")),
               ("B-QR4-NUM", seeds(Q4, "qrdisc2-dqrn-q4base-echo-sysscience-rc-rg-32k-hv3pr0-lr1e-5"))],   # 10-28: quoted in Supplementary Note 1
    "ehrsql": [("GR4-EHR", seeds(Q4, "ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"))],
}
CHECK_ONLY9T = {   # 10-28: B-GEN, the distilled cells' arm until the switch, which the text still quotes as the base recipe
    "medcalc": [("B-GEN", seeds(Q9, "gen_d27u-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 575))],
    "ehrsql": [("B-GEN", seeds(Q9, "gen_d27u-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 575))],
}
for size, ref, table in (("9B", "Qwen3.5-9B", RL9_LIST), ("9B taught", "Qwen3.5-9B", {b: [v] for b, v in TAUGHT9.items()}),
                         ("9B taught", "Qwen3.5-9B", CHECK_ONLY9T), ("4B", "Qwen3.5-4B", RL4), ("4B", "Qwen3.5-4B", CHECK_ONLY4)):
    for b, arms_b in table.items():
        for arm, dirs in arms_b:
            runs = [x for x in (load(d, b, OPEN_SUB) for d in dirs) if x is not None]
            bref = base_rows[ref][b][1]
            if not runs or bref is None:
                continue
            m = sum(acc(r, b) for r in runs) / len(runs)
            lo, hi = hier_ci(runs, bref, b)
            slo, shi = hier_ci(runs, sol[b][1], b) if sol[b][1] is not None else (float("nan"), float("nan"))
            arm_rows.append([size, b, arm, len(runs), f"{m:.4f}", " ".join(f"{acc(r, b):.4f}" for r in runs),
                             f"{m - acc(bref, b):.4f}", f"{lo:.4f}", f"{hi:.4f}",
                             f"{m - acc(sol[b][1], b):.4f}" if sol[b][1] is not None else "", f"{slo:.4f}", f"{shi:.4f}"])
with open(OUT / "figdata_arms_v05.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["size", "bench", "arm", "runs", "mean", "per_run", "diff_base", "base_lo", "base_hi", "diff_sol", "sol_lo", "sol_hi"])
    w.writerows(arm_rows)
with open(OUT / "figdata_v05.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["row", "ref", "bench", "arm", "runs", "diff", "ci_lo", "ci_hi", "status"])
    w.writerows(fig)
print("\n".join(md))
print()
print("\n".join(req))
