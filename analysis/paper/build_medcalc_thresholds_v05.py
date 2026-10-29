"""build_medcalc_thresholds.py -- clinical consequence of MedCalc errors (mock review 10-17, Referee 2: 'report errors by clinical
consequence: threshold crossings for scores that drive decisions, and the size of dosing errors'). Descriptive, post hoc, on the
graded answers already on disk (MedCalc-Bench-Verified 220-problem sample; the runs of figdata_fivebench.csv).
For each calculator with a conventional decision threshold (THRESH below; the categories a clinician acts on), an answer makes
a DECISION ERROR when it falls in a different category from the reference answer; an unanswered problem counts as a decision
error (strict scoring). For the three dosing calculators the size of each wrong answer is its absolute relative error.
Systems with several runs: mean over runs.
    python3 docs/rl4sci/v05/build_medcalc_thresholds.py   -> figdata_medcalc_thresholds.csv, supplementary/Supplementary_Table_7_decision_thresholds.csv
"""
import ast
import csv
import json
import os
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(f"{D}/../../..")
SUB = json.load(open(f"{R}/benchmarks/medcalc_ver_s220/SUBSET.json")) if os.path.exists(
    f"{R}/benchmarks/medcalc_ver_s220/SUBSET.json") else sorted(os.listdir(f"{R}/benchmarks/medcalc_ver_s220/problems"))
# calculator -> (category cut points, what the categories mean). A value v falls in category k = number of cut points <= v
# (for '<=' thresholds the cut is placed just above the boundary).
THRESH = {
    "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk": ([2], "anticoagulation considered at >= 2"),
    "CURB-65 Score for Pneumonia Severity": ([2], "hospital admission considered at >= 2"),
    "Wells' Criteria for Pulmonary Embolism": ([4.5], "PE likely at > 4"),
    "Wells' Criteria for DVT": ([2], "DVT likely at >= 2"),
    "MELD Na (UNOS/OPTN)": ([15], "transplant referral at >= 15"),
    "HAS-BLED Score for Major Bleeding Risk": ([3], "high bleeding risk at >= 3"),
    "HEART Score for Major Cardiac Events": ([4, 7], "low 0-3, moderate 4-6, high >= 7"),
    "PERC Rule for Pulmonary Embolism": ([1], "PE not ruled out at >= 1 criterion"),
    "Glasgow Coma Score (GCS)": ([9], "severe at <= 8"),
    "Sequential Organ Failure Assessment (SOFA) Score": ([2], "organ dysfunction at >= 2"),
    "SIRS Criteria": ([2], "SIRS at >= 2 criteria"),
    "Child-Pugh Score for Cirrhosis Mortality": ([7, 10], "class A 5-6, B 7-9, C >= 10"),
    "PSI Score: Pneumonia Severity Index for CAP": ([91], "class IV-V (inpatient) at > 90"),
    "Centor Score (Modified/McIsaac) for Strep Pharyngitis": ([4], "empiric antibiotics considered at >= 4"),
    "FeverPAIN Score for Strep Pharyngitis": ([4], "antibiotics considered at >= 4"),
    "Glasgow-Blatchford Bleeding Score (GBS)": ([2], "outpatient care possible at <= 1"),
    "Revised Cardiac Risk Index for Pre-Operative Risk": ([2], "elevated risk at >= 2"),
    "Caprini Score for Venous Thromboembolism (2005)": ([5], "highest risk at >= 5"),
    "CKD-EPI Equations for Glomerular Filtration Rate": ([30, 60], "eGFR < 30, 30-59, >= 60 (dose adjustment)"),
    "Creatinine Clearance (Cockcroft-Gault Equation)": ([30, 60], "CrCl < 30, 30-59, >= 60 (dose adjustment)"),
    "MDRD GFR Equation": ([30, 60], "eGFR < 30, 30-59, >= 60"),
    "Fibrosis-4 (FIB-4) Index for Liver Fibrosis": ([1.3, 2.67], "low < 1.3, indeterminate, high > 2.67"),
    "Morphine Milligram Equivalents (MME) Calculator": ([50, 90], "< 50, 50-89, >= 90 MME per day"),
}
DOSING = ("Morphine Milligram Equivalents (MME) Calculator", "Steroid Conversion Calculator", "Maintenance Fluids Calculations")
KEYS = [("sol_high", "GPT-6 Sol (high)"), ("gpt5_high", "gpt-5 (high)"), ("gpt41", "gpt-4.1"), ("o4mini", "o4-mini"),
        ("oss120", "gpt-oss-120b"), ("q35a3", "Qwen3.6-35B-A3B"), ("base9", "Qwen3.5-9B, untrained"),
        ("base27", "Qwen3.6-27B, untrained"), ("gr9", "9B + RL (clinical calculation)"), ("mc27", "27B + RL (clinical calculation)"),
        ("b400", "9B distilled (clinical calculation)"), ("bgen", "9B distilled (clinical calculation + EHR)")]
five = {(r["key"], r["bench"], r["metric"]): r for r in csv.DictReader(open(f"{D}/../v04/figdata_fivebench.csv"))}
# 10-24 (v05; mock review: the safety metrics for the Table 1 systems): systems outside the v04 five-benchmark file
EXTRA = {k: {"dirs": ";".join(f"{d}/medcalc_ver_s220" for d in v[1].split(";"))} for k, v in {'base27bf16': ('Qwen3.6-27B, untrained (bf16, Table 1)', 'results/pods/harness_v3_q27think/V3/ext'), 'bgenx2': ('9B distilled, health-record cell (B-GEN-X2)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s775/ext'), 'bgenx3': ('9B distilled, clinical-calculation cell (B-GEN-X3)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s800/ext'), 'base4': ('Qwen3.5-4B, untrained', 'results/pods/harness_v3_q4think/V3/ext'), 'gr4mc': ('4B + RL (clinical calculation; GR4-MC)', 'results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s3_s100/ext'), 'opd4mc': ('4B on-policy distillation (OPD4-MC)', 'results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5-s3_s100/ext')}.items()}
KEYS = KEYS + [(k, v[0]) for k, v in {'base27bf16': ('Qwen3.6-27B, untrained (bf16, Table 1)', 'results/pods/harness_v3_q27think/V3/ext'), 'bgenx2': ('9B distilled, health-record cell (B-GEN-X2)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s775/ext'), 'bgenx3': ('9B distilled, clinical-calculation cell (B-GEN-X3)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s800/ext'), 'base4': ('Qwen3.5-4B, untrained', 'results/pods/harness_v3_q4think/V3/ext'), 'gr4mc': ('4B + RL (clinical calculation; GR4-MC)', 'results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s3_s100/ext'), 'opd4mc': ('4B on-policy distillation (OPD4-MC)', 'results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-medcalc_ver-think-q4base-echo-sysscience-rc-opd-hv3pr0-lr1e-5-s3_s100/ext')}.items()]

meta = {}
for i in SUB:
    c = json.load(open(f"{R}/benchmarks/medcalc_ver/problems/{i}/eval_config.json"))
    pv = ast.literal_eval(c["_provenance"]) if isinstance(c["_provenance"], str) else c["_provenance"]
    gt = ast.literal_eval(c["ground_truth"]) if isinstance(c["ground_truth"], str) else c["ground_truth"]
    meta[i] = (pv.get("calculator"), list(gt.values())[0])


def num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def cat(v, cuts):
    return sum(v >= c for c in cuts)


def load(d):
    out = {}
    for i in SUB:
        f = f"{R}/{d}/{i}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        sub = r.get("submission")
        if isinstance(sub, str):
            try:
                sub = ast.literal_eval(sub)
            except (ValueError, SyntaxError):
                sub = None
        ans = None
        if isinstance(sub, dict):
            a = sub.get("answer", sub)
            ans = list(a.values())[0] if isinstance(a, dict) and a else a
        g = r.get("grade")
        g = ast.literal_eval(g) if isinstance(g, str) else (g or {})
        out[i] = (None if r.get("error") else num(ans), bool(g.get("passed")) and not r.get("error"))
    return out


rows, per_calc = [], []
for key, name in KEYS:
    r = five.get((key, "medcalc", "accuracy")) or EXTRA.get(key)
    if r is None:
        continue
    runs = [load(d) for d in r["dirs"].split(";")]
    dec_err, dec_n, wrong_cross, wrong_n, dose_rel = [], [], [], [], []
    for o in runs:
        de = dn = wc = wn = 0
        for i in SUB:
            calc, gt = meta[i]
            if calc not in THRESH:
                continue
            cuts = THRESH[calc][0]
            g = num(gt)
            v, ok = o.get(i, (None, False))
            dn += 1
            crossed = v is None or g is None or cat(v, cuts) != cat(g, cuts)
            de += crossed
            if not ok:
                wn += 1
                wc += crossed
        dec_err.append(100 * de / dn); dec_n.append(dn); wrong_cross.append(100 * wc / wn if wn else float("nan")); wrong_n.append(wn)
        for i in SUB:
            calc, gt = meta[i]
            v, ok = o.get(i, (None, False))
            if calc in DOSING and not ok and v is not None and num(gt):
                dose_rel.append(abs(v - num(gt)) / abs(num(gt)))
    rows.append({"key": key, "system": name, "runs": len(runs), "decision_problems": dec_n[0],
                 "decision_error_pct": round(st.mean(dec_err), 1), "wrong_answers": round(st.mean(wrong_n), 1),
                 "wrong_that_cross_pct": round(st.mean([x for x in wrong_cross if x == x]), 1) if any(x == x for x in wrong_cross) else "",
                 "dosing_wrong": len(dose_rel), "dosing_median_rel_error_pct": round(100 * st.median(dose_rel), 1) if dose_rel else ""})
with open(f"{D}/figdata_medcalc_thresholds_v05.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(f"{D}/supplementary/Supplementary_Table_7_decision_thresholds.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["calculator", "category cut points", "categories"])
    for c, (cuts, what) in THRESH.items():
        w.writerow([c, "; ".join(str(x) for x in cuts), what])
print(f"{len(THRESH)} decision calculators, {rows[0]['decision_problems']} of the 220 problems")
print(f"{'system':42s} {'runs':>4} {'dec.err%':>8} {'wrong':>6} {'wrong->cross%':>13} {'dose n':>6} {'dose med rel%':>13}")
for r in rows:
    print(f"{r['system']:42s} {r['runs']:4d} {r['decision_error_pct']:8.1f} {r['wrong_answers']:6.1f} {str(r['wrong_that_cross_pct']):>13} "
          f"{r['dosing_wrong']:6d} {str(r['dosing_median_rel_error_pct']):>13}")
