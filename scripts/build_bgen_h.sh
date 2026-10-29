#!/bin/bash
# build_bgen_h.sh (10-25, docs/RL_PLAN.md H-BEAT-CLAUDE arm 3, B-GEN-H) -- after both hinted-teacher halves (HV3_1378/1379) have results:
#   1. scratchpad/build_distill_set.py --keep-unlabeled on the hinted teacher reads, with --inst-root = the ORIGINAL MedCalc training
#      instances, so every rebuilt episode carries the hint-free task -> results/pods/distill27genh/rescued_raw.jsonl;
#   2. drops any episode whose assistant text REFERS TO a reference / hint (the student's prompt has none). 10-26 amendment (RL_PLAN):
#      the first filter (any 40-character span shared with the hint, or 'the note') kept 5 of 197 -- it matched calculator and criterion
#      names and the patient note; formula knowledge from MedCalc's TRAIN explanations may stay in an episode, a reference to it may not;
#   3. the union with B-GEN's set (results/pods/distill27genh/gen_h.jsonl) and its pool generated/gen_d27h (symlinks to the original
#      instances); STEPS = 25 x round(2 N / 8 / 25) (B-GEN's two epochs) -> results/pods/distill27genh/STEPS.
# Refuses to overwrite. Run on any pod (CPU work) or the CPU agent.
set -euo pipefail
R=/project/pi_hongyu_umass_edu/hieutran/deep/deeprs; cd $R
S=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
G=/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated; O=results/pods/distill27genh
T=results/pods/harness_v3_q27/V3__rl-czero-medcalc_ver-think-q27base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s1_s100/ext
for h in a b; do [ -f $T/mc_hint_teach_$h/summary.json ] || { echo "teacher half $h has no summary.json yet"; exit 3; }; done
[ -e $O/gen_h.jsonl ] && { echo "$O/gen_h.jsonl exists -- nothing to do"; exit 0; }
mkdir -p $O; rm -rf $G/medcalc_genh_rescued $O/rescued_raw.jsonl
GBP_SYSTEM=science venv/bin/python $S/build_distill_set.py --keep-unlabeled --inst-root $G/medcalc_ver/train \
  --out $O/rescued_raw.jsonl --pool $G/medcalc_genh_rescued $T/mc_hint_teach_a $T/mc_hint_teach_b
venv/bin/python - <<'PY'
import ast, json, os, re
R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"; G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
O = f"{R}/results/pods/distill27genh"
REF = re.compile(r"\b(reference (note|formula|text|material|document|card|sheet|table|says|states|gives|provided)|the reference\b|"
                 r"re-?read(ing)? the reference|according to the (reference|hint|provided)|the hint|hint says|as hinted|"
                 r"provided (reference|formula|note|hint)|documentation provided|reference for this calculator|"
                 r"the formula (given|provided) (above|in the (prompt|task|reference)))", re.I)
keep, drop = [], {"refers to a reference or hint": 0, "hint in prompt": 0}
for l in open(f"{O}/rescued_raw.jsonl"):
    r = json.loads(l)
    asst = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant" and isinstance(m["content"], str))
    prompt = " ".join(m["content"] for m in r["messages"] if m["role"] != "assistant" and isinstance(m["content"], str))
    if "Reference for this calculator" in prompt:
        drop["hint in prompt"] += 1; continue
    if REF.search(asst):
        drop["refers to a reference or hint"] += 1; continue
    keep.append(l)
gen = [l for l in open(f"{R}/results/pods/distill27gen/gen_u.jsonl") if l.strip()]
names = {json.loads(l)["instance"] for l in gen}
keep = [l for l in keep if json.loads(l)["instance"] not in names]
open(f"{O}/gen_h.jsonl.tmp", "w").write("".join(gen) + "".join(keep))
os.rename(f"{O}/gen_h.jsonl.tmp", f"{O}/gen_h.jsonl")
pool = f"{G}/gen_d27h/train"; os.makedirs(pool)
for l in gen + keep:
    i = json.loads(l)["instance"]
    src = os.path.realpath(f"{G}/gen_d27u/train/{i}") if os.path.exists(f"{G}/gen_d27u/train/{i}") else os.path.realpath(f"{G}/medcalc_ver/train/{i}")
    assert os.path.isfile(f"{src}/eval_config.json"), i
    os.symlink(src, f"{pool}/{i}")
n = len(gen) + len(keep); steps = 25 * round(2 * n / 8 / 25)
open(f"{O}/STEPS", "w").write(str(steps))
json.dump({"gen_u": len(gen), "rescued_raw": sum(1 for _ in open(f"{O}/rescued_raw.jsonl")), "rescued_kept": len(keep), "dropped": drop,
           "n": n, "steps": steps}, open(f"{O}/MANIFEST.json", "w"), indent=1)
print(json.dumps(json.load(open(f"{O}/MANIFEST.json"))))
PY
