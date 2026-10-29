"""build_ehrsql_rs.py -- EHRSQL-2024's official reliability scores for every system in Table 1 (mock review 10-17, Referee 2:
'EHRSQL-2024 is a reliability task with abstention; report RS(0), RS(10), RS(N) and abstention precision and recall').
Descriptive, post hoc, from the graded answers already on disk (the runs listed in figdata_fivebench.csv, 300-question sample).
Per question, from the official grader's details: a correct result or a correct abstention scores +1, abstaining on an
answerable question 0, a wrong result or any query for an unanswerable question -c. A question left unanswered (time-out,
output limit) has no grader details and is counted as an error (-c), as strict scoring counts it wrong. RS(c) is the mean
over the 300 questions; RS(N) uses c = 300. Abstention precision = correct abstentions / abstentions; recall = correct
abstentions / unanswerable questions. Systems with several runs: mean over runs (range in the CSV).
    python3 docs/rl4sci/v05/build_ehrsql_rs.py     -> docs/rl4sci/v05/figdata_ehrsql_rs.csv
"""
import ast
import csv
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(f"{D}/../../..")
IDS = json.load(open(f"{R}/benchmarks/ehrsql2024_s300/SUBSET.json"))
KEYS = [("sol_high", "GPT-6 Sol (high)"), ("gpt5_high", "gpt-5 (high)"), ("gpt41", "gpt-4.1"), ("oss120", "gpt-oss-120b"),
        ("q35a3", "Qwen3.6-35B-A3B"), ("base9", "Qwen3.5-9B, untrained"), ("base27", "Qwen3.6-27B, untrained"),
        ("gr9", "9B + RL (clinical calculation)"), ("sci2clin9", "9B + RL (science, then clinical)"),
        ("mc27", "27B + RL (clinical calculation)"), ("b400", "9B distilled (clinical calculation)"),
        ("bgen", "9B distilled (clinical calculation + EHR)")]
five = {(r["key"], r["bench"], r["metric"]): r for r in csv.DictReader(open(f"{D}/../v04/figdata_fivebench.csv"))}
# 10-24 (v05; mock review: the safety metrics for the Table 1 systems): systems outside the v04 five-benchmark file
EXTRA = {k: {"dirs": ";".join(f"{d}/ehrsql2024_s300" for d in v[1].split(";"))} for k, v in {'base27bf16': ('Qwen3.6-27B, untrained (bf16, Table 1)', 'results/pods/harness_v3_q27think/V3/ext'), 'bgenx2': ('9B distilled, health-record cell (B-GEN-X2)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s775/ext'), 'bgenx3': ('9B distilled, clinical-calculation cell (B-GEN-X3)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s800/ext'), 'base4': ('Qwen3.5-4B, untrained', 'results/pods/harness_v3_q4think/V3/ext'), 'gr4ehr': ('4B + RL (health-record questions; GR4-EHR)', 'results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s3_s100/ext'), 'ehr4kl': ('4B + RL with KL anchor (EHR4-KL)', 'results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5-s3_s100/ext')}.items()}
KEYS = KEYS + [(k, v[0]) for k, v in {'base27bf16': ('Qwen3.6-27B, untrained (bf16, Table 1)', 'results/pods/harness_v3_q27think/V3/ext'), 'bgenx2': ('9B distilled, health-record cell (B-GEN-X2)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s775/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x2-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s775/ext'), 'bgenx3': ('9B distilled, clinical-calculation cell (B-GEN-X3)', 'results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s1_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s2_s800/ext;results/pods/harness_v3_q9/V3__rl-czero-gen_d27x3-think-q9base-echo-sysscience-rc-dst-hv3pr0-lr5e-5-s3_s800/ext'), 'base4': ('Qwen3.5-4B, untrained', 'results/pods/harness_v3_q4think/V3/ext'), 'gr4ehr': ('4B + RL (health-record questions; GR4-EHR)', 'results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s3_s100/ext'), 'ehr4kl': ('4B + RL with KL anchor (EHR4-KL)', 'results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5-s1_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5-s2_s100/ext;results/pods/harness_v3_q4/V3__rl-czero-ehrsql2024-think-q4base-echo-sysscience-rc-rg-hv3pr0-kl0.2-lr1e-5-s3_s100/ext')}.items()]


def outcomes(d):
    """per question: 'correct' | 'abstain_wrong' (abstained on an answerable one) | 'error'; plus answerable flags"""
    out, ans = {}, {}
    for i in IDS:
        f = f"{R}/{d}/{i}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        g = r.get("grade")
        if isinstance(g, str):
            try:
                g = ast.literal_eval(g)
            except (ValueError, SyntaxError):
                g = None
        det = (g or {}).get("details") or []
        if r.get("error") or not det:
            out[i] = "error"     # unanswered: scored as an error
            continue
        x = det[0]
        ans[i] = bool(x.get("answerable"))
        if x.get("correct"):
            out[i] = "correct"
        elif x.get("abstained") and x.get("answerable"):
            out[i] = "abstain_wrong"
        else:
            out[i] = "error"
        out[i + "#abst"] = bool(x.get("abstained"))
    return out, ans


rows = []
UNANS = None
for key, name in KEYS:
    r = five.get((key, "ehrsql", "accuracy")) or EXTRA.get(key)
    if r is None:
        continue
    per = []
    for d in r["dirs"].split(";"):
        o, ans = outcomes(d)
        n = len(IDS)
        corr = sum(o.get(i) == "correct" for i in IDS)
        err = sum(o.get(i, "error") == "error" for i in IDS)
        un = [i for i in IDS if i in ans and not ans[i]]
        UNANS = UNANS or len(un)
        abst = [i for i in IDS if o.get(i + "#abst")]
        corr_abst = [i for i in abst if i in ans and not ans[i]]
        per.append({"rs0": 100 * corr / n, "rs10": 100 * (corr - 10 * err) / n, "rsn": 100 * (corr - n * err) / n,
                    "err": err, "abst_p": 100 * len(corr_abst) / len(abst) if abst else float("nan"),
                    "abst_r": 100 * len(corr_abst) / len(un) if un else float("nan"), "n_abst": len(abst)})
    m = {k: sum(p[k] for p in per) / len(per) for k in per[0]}
    rows.append({"key": key, "system": name, "runs": len(per), "RS0": round(m["rs0"], 1), "RS10": round(m["rs10"], 1),
                 "RSN": round(m["rsn"], 1), "errors": round(m["err"], 1), "abstentions": round(m["n_abst"], 1),
                 "abst_precision": round(m["abst_p"], 1), "abst_recall": round(m["abst_r"], 1),
                 "RS10_runs": ";".join(f"{p['rs10']:.1f}" for p in per)})
with open(f"{D}/figdata_ehrsql_rs_v05.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"EHRSQL-2024 s300: {UNANS} unanswerable questions; RS in points (x100)")
print(f"{'system':40s} {'runs':>4} {'RS(0)':>6} {'RS(10)':>7} {'RS(N)':>9} {'errors':>6} {'abst':>5} {'abstP':>6} {'abstR':>6}")
for r in rows:
    print(f"{r['system']:40s} {r['runs']:4d} {r['RS0']:6.1f} {r['RS10']:7.1f} {r['RSN']:9.1f} {r['errors']:6.1f} "
          f"{r['abstentions']:5.1f} {r['abst_precision']:6.1f} {r['abst_recall']:6.1f}")
