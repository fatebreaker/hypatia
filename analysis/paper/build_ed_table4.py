"""Extended Data Table 4 of the v05 manuscript (10-24, mock review: 'show every arm tried per cell'): for every cell of Table 1,
every reinforcement-learning (or, in the distilled row, distillation) arm trained on that benchmark's own domain, in the order
it was registered, with its result against the untrained model of the same size. Controls (shuffled rewards, rejection
sampling, label-free rewards, no-code reward, calculator hold-outs) are not candidates for a cell and are left out; they are
in Extended Data Tables 1 and 3. For the arm Table 1 uses, two further intervals: Bonferroni-adjusted for the number of
arms in the cell with at least one completed run (1 - 0.05/m), and a t interval over the three run means (between-run
variation only).
Registration times (UTC, 2026) are given where the plan's entry names the arm; '–' marks arms registered under a broader
hypothesis whose time is in the plan (Supplementary Data 1).
    python3 docs/rl4sci/v05/build_ed_table4.py      (writes ed_table4.md; make_v05.py inserts it at [[ed_table4]])
"""
import io
import contextlib
import os
import statistics
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
with contextlib.redirect_stdout(io.StringIO()):
    import build_table1 as bt   # noqa: E402  (rebuilds Table 1 on import; reuses its loaders and bootstrap)

Q9, Q4 = bt.Q9, bt.Q4
T975 = {2: 4.303, 1: 12.706}   # t quantiles, 2 and 1 degrees of freedom


def fam(q, run, step=100):
    return [q.format(f"{run}-s{s}_s{step}") for s in (1, 2, 3)]


# (cell, base row name, [(arm, registered (UTC 2026), description, dirs or a status string, selected?)])
C = [
    ("9B + RL · QRData", "Qwen3.5-9B", "qrdata", [
        ("SCI9", "10-01 18:40", "synthetic regression-discontinuity tasks", fam(Q9, "causal_rdd-think-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("SCIMIX9", "10-02 12:48", "all synthetic causal designs", fam(Q9, "causal_mix-think-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("GR-SCI9", "–", "as SCI9, grounded reward", fam(Q9, "causal_rdd-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("+ 13 single-run variants", "–", "other synthetic sciences and settings", "one run each (Extended Data Fig. 4)", False),
        ("QD9-FG", "10-25 21:20", "causal designs + BioDSBench-style statistics + causal discovery as QRData poses it, KL", fam(Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("B-QR9-FG ‡", "10-25 21:45", "distilled from the 27B's numerical QRData notebooks, then RL on QD9-FG's pool", fam(Q9, "qrdisc2-dqrn-q9base-echo-sysscience-rc-rg-32k-hv3pr0-lr1e-5"), False),
        ("QRREAL9", "10-26 14:04", "the 89 QRData questions outside QRData-Verified, KL", fam(Q9, "qrreal89-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("QDS9", "10-27 06:32", "QD9-FG's pool + 800 synthetic layered-diagnosis questions; stopped after run 1 (no gain from the pool)", fam(Q9, "qrdsem-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("B-QDS9 ‡", "10-27 18:37", "B-QR9-FG's distilled start, then RL on QDS9's pool", "stopped at steps 3-15, before any read: QDS9 run 1 and QRS4 showed no gain from the pool", False),
        ("QD9-FG-L", "10-28 05:07", "QD9-FG's runs continued from step 100 to step 200", fam(Q9, "qrdisc2-think-q9base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5", 200), True)]),
    ("9B + RL · clinical calculation", "Qwen3.5-9B", "medcalc", [
        ("MC9", "–", "MedCalc training split, answer reward", fam(Q9, "medcalc_ver-think-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("MC9, no thinking", "10-02 05:35", "as MC9, non-thinking format", fam(Q9, "medcalc_ver-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("GR9-MC", "10-01 19:34", "as MC9, grounded reward", fam(Q9, "medcalc_ver-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), True),
        ("MC9-KL", "–", "as MC9, KL anchor 0.2", fam(Q9, "medcalc_ver-think-q9base-echo-sysscience-rc-hv3pr0-kl0.2-lr1e-5"), False),
        ("GR9-MC-KL", "–", "as GR9-MC, KL anchor 0.2", fam(Q9, "medcalc_ver-think-q9base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5"), False),
        ("GR9-MC, signal pool", "10-07 22:49", "as GR9-MC, unsolved-problem pool", fam(Q9, "medcalc_ver_sig9-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("SCI2CLIN9", "10-02 18:02", "synthetic science, then both clinical splits", fam(Q9, "clinmix-sci9-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("+ 15 single-run variants", "–", "learning rate, KL, hints, overflow masking", "one run each (Extended Data Fig. 5)", False)]),
    ("9B + RL · health-record questions", "Qwen3.5-9B", "ehrsql", [
        ("SCI2CLIN9", "10-02 18:02", "synthetic science, then both clinical splits", fam(Q9, "clinmix-sci9-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), True),
        ("EHR9", "–", "EHRSQL training split, answer reward", fam(Q9, "ehrsql2024-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("GR9-EHR", "–", "as EHR9, grounded reward", fam(Q9, "ehrsql2024-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("SCI9→EHR", "10-03 03:07", "synthetic science, then EHRSQL split", fam(Q9, "ehrsql2024-sci9-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("EHR9, thinking", "–", "EHRSQL split, thinking format, LR 5e-6", fam(Q9, "ehrsql2024-think-q9base-echo-sysscience-rc-hv3pr0-lr5e-6"), False),
        ("SCI69→CLIN", "–", "six synthetic sciences, then both clinical splits", fam(Q9, "clinmix-sci69-q9base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("+ 12 single-run or template-hold-out variants", "–", "", "Supplementary Table 4", False)]),
    ("9B + RL · MedAgentBench", "Qwen3.5-9B", "mab", [
        ("MAB9", "10-21 12:47", "MedAgentBench-style and lab tasks", fam(Q9, "mablab27mix-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("MAB9-T8", "10-25 21:20", "as MAB9, the benchmark's 8 turns", fam(Q9, "mablab27mix_t8-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), True)]),
    ("9B + RL · BioDSBench", "Qwen3.5-9B", "bds", [
        ("BDS9", "10-21 12:47", "BioDSBench-style tasks (49 other studies)", fam(Q9, "bdsgen-think-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("BDS9-KL", "10-23 18:13", "as BDS9, KL anchor 0.2", fam(Q9, "bdsgen-think-q9base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5"), True)]),
    ("9B + RL · Kepler", "Qwen3.5-9B", "kepler", [
        ("ASTRO9", "10-11 02:16", "simulated light curves", fam(Q9, "astro_mix-think-q9base-echo-sysscience-rc-32k-hv3pr0-lr1e-5"), True)]),
    ("9B distilled · clinical calculation", "Qwen3.5-9B", "medcalc", [
        ("B400", "10-08 15:29", "27B clinical-calculation notebooks, 400 steps", fam(Q9, "medcalc_ver_d27-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 400), False),
        ("B-GEN", "10-11 08:13", "27B clinical + EHRSQL notebooks, 575 steps", fam(Q9, "gen_d27u-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 575), False),
        # 10-27 (mock review: every arm tried belongs here): the arms registered to close the gap to Claude Opus 5.5
        ("B-GEN-H", "10-25 21:30", "B-GEN + 106 notebooks of a hinted 27B on problems it had failed", fam(Q9, "gen_d27h-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 600), False),
        ("B-GEN-H-RL †", "10-25 21:39", "B-GEN-H, then grounded RL on the MedCalc split", fam(Q9, "medcalc_ver-dgenh-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("MCW9 †", "10-26 14:04", "B-GEN-H, then grounded RL weighted toward the calculators it fails", fam(Q9, "mcweak3-dgenh-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("B-GEN-X", "10-27 01:42", "B-GEN-H + 279 explanation-guided 27B notebooks for 13 weak calculators", fam(Q9, "gen_d27x-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 675), False),
        ("B-GEN-X-RL †", "10-27 01:42", "B-GEN-X, then grounded RL (MedCalc split + the weak calculators' problems)", fam(Q9, "mcweakx-dgenx-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("B-GEN-X2", "10-27 22:47", "B-GEN-X + 442 explanation-guided 27B notebooks from further attempts and five more calculators", fam(Q9, "gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 775), False),
        ("B-GEN-X3", "10-28 10:27", "B-GEN-X2 + more 27B attempts for the five calculators it still trails Claude on, 120 new CHA2DS2-VASc rows", fam(Q9, "gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 800), True)]),
    # 10-28: the distilled health-record cell lists every distilled arm read on EHRSQL (each was trained on both clinical splits);
    # B-GEN-H-RL and B-GEN-X-RL were not read there
    ("9B distilled · health-record questions", "Qwen3.5-9B", "ehrsql", [
        ("B-GEN", "10-11 08:13", "27B clinical + EHRSQL notebooks, 575 steps", fam(Q9, "gen_d27u-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 575), False),
        ("B-GEN-H", "10-25 21:30", "B-GEN + 106 notebooks of a hinted 27B on problems it had failed", fam(Q9, "gen_d27h-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 600), False),
        ("MCW9 †", "10-26 14:04", "B-GEN-H, then grounded RL weighted toward the calculators it fails", fam(Q9, "mcweak3-dgenh-q9base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("B-GEN-X", "10-27 01:42", "B-GEN-H + 279 explanation-guided 27B notebooks for 13 weak calculators", fam(Q9, "gen_d27x-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 675), False),
        ("B-GEN-X2", "10-27 22:47", "B-GEN-X + 442 explanation-guided 27B notebooks from further attempts and five more calculators", fam(Q9, "gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 775), True),
        ("B-GEN-X3", "10-28 10:27", "B-GEN-X2 + more 27B attempts for the five calculators it still trails Claude on, 120 new CHA2DS2-VASc rows", fam(Q9, "gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5", 800), False)]),
    ("9B distilled · Kepler", "Qwen3.5-9B", "kepler", [
        ("DASTRO9", "10-21 14:07", "27B astronomy notebooks, 400 steps, no RL", fam(Q9, "astro_d27-think-q9base-echo-sysscience-rc-dst-32k-hv3pr0-lr5e-5", 400), False),
        ("B-ASTRO9", "10-21 14:07", "DASTRO9, then RL on real training stars", fam(Q9, "kepreal_mix-dastro-q9base-echo-sysscience-rc-32k-hv3pr0-lr1e-5"), True)]),
    ("4B + RL · QRData", "Qwen3.5-4B", "qrdata", [
        ("SCI4", "10-21 15:03", "the 9B's SCI9 recipe", fam(Q4, "causal_rdd-think-q4base-echo-sysscience-rc-hv3pr0-lr1e-5"), False),
        ("SCI4-KL", "10-22 02:40", "all causal designs, KL anchor 0.2", fam(Q4, "causal_mix-think-q4base-echo-sysscience-rc-hv3pr0-kl0.2-lr1e-5"), False),
        ("OPD4-QR", "10-22 17:24", "on-policy distillation from the RL-trained 27B", "stopped: mechanical failure, then collapse", False),
        ("OPD4-QR-32k", "10-22 21:12", "as OPD4-QR, 32k context", "stopped at step 10-13 (collapse)", False),
        ("OPD4-QR-R", "10-23 03:11", "as OPD4-QR-32k, task reward added", "stopped at step 10-14 (collapse)", False),
        ("OPD4-QR-9", "10-23 09:29", "on-policy distillation from the RL-trained 9B", "stopped at step 13 (collapse)", False),
        ("QM4-KL", "10-23 09:29", "causal designs + BioDSBench-style statistics, KL", fam(Q4, "qrmix-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("QD4-KL", "10-23 17:19", "QM4-KL + synthetic causal discovery", fam(Q4, "qrdisc-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("B-QR4 ‡", "10-24 15:14", "distilled from the RL-trained 27B's notebooks, then QD4-KL's RL without KL", "stopped 10-25 by the first author: its distilled start scored 53.4% and 50.2% on QRData", False),
        ("B-QR4-9 ‡", "10-24 15:23", "as B-QR4, teacher the RL-trained 9B (BDS9-KL)", fam(Q4, "qrdisc-dqr9-q4base-echo-sysscience-rc-rg-32k-hv3pr0-lr1e-5"), False),
        ("B-QR4-9K ‡", "10-25 08:33", "as B-QR4-9, KL anchor held at the distilled start", [Q4.format(f"qrdisc-think-q4dqr9s{s}base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5-s{s}_s100") for s in (1, 2, 3)], False),
        ("QD4-FG", "10-25 18:20", "QD4-KL with causal discovery as QRData poses it", fam(Q4, "qrdisc2-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("QRREAL4", "10-26 14:04", "the 89 QRData questions outside QRData-Verified, KL", fam(Q4, "qrreal89-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), True),
        ("B-QR4-NUM ‡", "10-27 02:58", "distilled from the 27B's numerical QRData notebooks, then RL on QD4-FG's pool", fam(Q4, "qrdisc2-dqrn-q4base-echo-sysscience-rc-rg-32k-hv3pr0-lr1e-5"), False),
        ("QRS4", "10-27 06:32", "QRREAL4's pool + 89 synthetic layered-diagnosis questions", fam(Q4, "qrrsem-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("B-QRS4 ‡", "10-27 18:37", "B-QR4-NUM's distilled start, then RL on QRS4's pool", "stopped at steps 3-15, before any read: QDS9 run 1 and QRS4 showed no gain from the pool", False),
        # 10-28 22:16 (RL_PLAN H-MEET-ALL arm 10): QRREAL4's runs continued to step 200; read at step 200 (rule QR4-LIFTS-L)
        ("QRREAL4-L", "10-28 22:16", "QRREAL4's runs continued from step 100 to step 200", fam(Q4, "qrreal89-think-q4base-echo-sysscience-rc-rg-32k-hv3pr0-kl0.2-lr1e-5", 200), False)]),
    ("4B + RL · clinical calculation", "Qwen3.5-4B", "medcalc", [
        ("GR4-MC", "10-21 15:03", "the 9B's GR9-MC recipe", fam(Q4, "medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("OPD4-MC ‡", "10-22 03:10", "on-policy distillation from the RL-trained 27B", fam(Q4, "medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5"), True)]),
    ("4B + RL · health-record questions", "Qwen3.5-4B", "ehrsql", [
        ("GR4-EHR", "10-21 15:03", "EHRSQL split, grounded reward", fam(Q4, "ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("EHR4-KL", "10-22 14:39", "as GR4-EHR, KL anchor 0.2", fam(Q4, "ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5"), True)]),
    ("4B + RL · MedAgentBench", "Qwen3.5-4B", "mab", [
        ("MAB4", "10-21 15:03", "the 9B's MAB9 recipe (25 training turns)", fam(Q4, "mablab27mix-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("MAB4-T8", "10-23 13:16", "as MAB4, the benchmark's 8 turns", fam(Q4, "mablab27mix_t8-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), True)]),
    ("4B + RL · BioDSBench", "Qwen3.5-4B", "bds", [
        ("BDS4", "10-21 15:03", "the 9B's BDS9 recipe", fam(Q4, "bdsgen-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5"), False),
        ("BDS4-KL", "10-23 09:42", "as BDS4, KL anchor 0.2", fam(Q4, "bdsgen-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5"), True)]),
    ("4B + RL · Kepler", "Qwen3.5-4B", "kepler", [
        ("ASTRO4", "10-21 15:03", "the 9B's ASTRO9 recipe", fam(Q4, "astro_mix-think-q4base-echo-sysscience-rc-32k-hv3pr0-lr1e-5"), False),
        ("ASTRO4-KL", "10-24 00:15", "as ASTRO4, KL anchor 0.2", fam(Q4, "astro_mix-think-q4base-echo-sysscience-rc-32k-hv3pr0-kl0.2-lr1e-5"), False),
        ("B-ASTRO4 ‡", "10-24 13:04", "the 9B's B-ASTRO9 recipe", fam(Q4, "kepreal_mix-dastro-q4base-echo-sysscience-rc-32k-hv3pr0-lr1e-5"), True)]),
]


def ci_level(runs, base, bench, alpha, n=10000, seed=20261023):
    """the hierarchical bootstrap of build_table1, at level 1 - alpha"""
    import random
    rng = random.Random(seed)
    ids = bt.IDS[bench]
    out = []
    for _ in range(n):
        rs = [rng.choice(runs) for _ in runs]
        ps = [rng.choice(ids) for _ in ids]
        out.append(100 * sum(sum(r.get(i, 0) for r in rs) / len(rs) - base.get(i, 0) for i in ps) / len(ps))
    out.sort()
    return out[int(alpha / 2 * n)], out[min(n - 1, int((1 - alpha / 2) * n))]


# 10-28 (the 10-27 mock review's concern 6: 'add a training-data provenance column'): where each arm's training data come from,
# relative to the benchmark it is scored on. S synthetic problems with known answers (no item, file or entity of any benchmark);
# T the benchmark's own training split (its source and task types; every test note or question removed); N the benchmark's own
# items outside the scored subset (QRData's 89 unverified questions: no shared data file, same generating families); O the
# benchmark's task types on entities its test set does not use (other patients, cBioPortal studies or Kepler stars); D(x)
# notebooks of an RL-trained teacher on x, distilled; E the teacher saw the training item's reference explanation (notebooks
# that mention it dropped); 'a -> b' two stages; * pool or length chosen after reading this cell's test results.
DATA = {
    "SCI9": "S", "SCIMIX9": "S", "GR-SCI9": "S", "+ 13 single-run variants": "S", "QD9-FG": "S, O *",
    "B-QR9-FG ‡": "D(S, O) → S, O *", "QRREAL9": "N", "QDS9": "S, O *", "B-QDS9 ‡": "D(S, O) → S, O *", "QD9-FG-L": "S, O *",
    "MC9": "T", "MC9, no thinking": "T", "GR9-MC": "T", "MC9-KL": "T", "GR9-MC-KL": "T", "GR9-MC, signal pool": "T",
    "SCI2CLIN9": "S → T", "+ 15 single-run variants": "T",
    "EHR9": "T", "GR9-EHR": "T", "SCI9→EHR": "S → T", "EHR9, thinking": "T", "SCI69→CLIN": "S → T",
    "+ 12 single-run or template-hold-out variants": "T",
    "MAB9": "O", "MAB9-T8": "O", "BDS9": "O", "BDS9-KL": "O", "ASTRO9": "S",
    "B400": "D(T)", "B-GEN": "D(T)", "B-GEN-H": "D(T)", "B-GEN-H-RL †": "D(T) → T", "MCW9 †": "D(T) → T *",
    "B-GEN-X": "D(T), E *", "B-GEN-X-RL †": "D(T), E → T *", "B-GEN-X2": "D(T), E *", "B-GEN-X3": "D(T), E *",
    "DASTRO9": "D(S)", "B-ASTRO9": "D(S) → O",
    "SCI4": "S", "SCI4-KL": "S", "OPD4-QR": "S, O", "OPD4-QR-32k": "S, O", "OPD4-QR-R": "S, O", "OPD4-QR-9": "S, O",
    "QM4-KL": "S, O", "QD4-KL": "S, O", "B-QR4 ‡": "D(S, O) → S, O", "B-QR4-9 ‡": "D(S, O) → S, O",
    "B-QR4-9K ‡": "D(S, O) → S, O", "QD4-FG": "S, O *", "QRREAL4": "N", "B-QR4-NUM ‡": "D(S, O) → S, O *",
    "QRS4": "N, S *", "B-QRS4 ‡": "D(S, O) → N, S *", "QRREAL4-L": "N *",
    "GR4-MC": "T", "OPD4-MC ‡": "T", "GR4-EHR": "T", "EHR4-KL": "T", "MAB4": "O", "MAB4-T8": "O", "BDS4": "O", "BDS4-KL": "O",
    "ASTRO4": "S", "ASTRO4-KL": "S", "B-ASTRO4 ‡": "D(S) → O",
}
for _cell, _b, _bench, _arms in C:
    for _a in _arms:
        assert _a[0] in DATA, ("no provenance code for", _a[0])
f1 = lambda x: f"{x:+.1f}".replace("-", "−")
# 10-28: the table is computed only when run as a script (rigor_v05.py imports this file for C and DATA)
if __name__ == "__main__":
    rows = ["| Cell | Arm | Registered | Training | Data | Runs | Δ vs untrained (95% CI) | Selected: Bonferroni CI; t interval |",
            "|:--|:--|:-:|:--|:-:|:-:|:--|:--|"]
    for cell, base_name, b, arms in C:
        base = bt.base_rows[base_name][b][1]
        # m = arms of the cell with at least one completed run (the arms a choice could have been made among)
        m = sum(1 for _, _, _, dirs, _ in arms if isinstance(dirs, list)
                and any(x is not None for x in (bt.load(d, b, bt.OPEN_SUB) for d in dirs)))
        first = True
        for arm, reg, desc, dirs, sel in arms:
            if isinstance(dirs, str):
                rows.append(f"| {cell if first else ''} | {arm} | {reg} | {desc} | {DATA[arm]} | – | {dirs} | |")
                first = False
                continue
            runs = [x for x in (bt.load(d, b, bt.OPEN_SUB) for d in dirs) if x is not None]
            if not runs:
                rows.append(f"| {cell if first else ''} | {arm} | {reg} | {desc} | {DATA[arm]} | 0/3 | training | |")
                first = False
                continue
            accs = [bt.acc(r, b) for r in runs]
            d = sum(accs) / len(accs) - bt.acc(base, b)
            lo, hi = bt.hier_ci(runs, base, b)
            cell_ci = f"{f1(d)} ({f1(lo)} to {f1(hi)})" + ("" if len(runs) == 3 else f"; {len(runs)}/3 runs")
            extra = ""
            if sel and len(runs) == 3:
                blo, bhi = ci_level(runs, base, b, 0.05 / max(m, 1)) if m > 1 else (lo, hi)
                diffs = [a - bt.acc(base, b) for a in accs]
                half = T975[2] * statistics.stdev(diffs) / 3 ** 0.5
                extra = (f"m = {m}: {f1(blo)} to {f1(bhi)}; " if m > 1 else "m = 1; ") + \
                    (f"t: {f1(d - half)} to {f1(d + half)}" if half > 0 else "t: the three runs scored the same")
            name = f"**{arm}**" if sel else arm
            rows.append(f"| {cell if first else ''} | {name} | {reg} | {desc} | {DATA[arm]} | {len(runs)} | {cell_ci} | {extra} |")
            first = False
    open(os.path.join(D, "ed_table4.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))
