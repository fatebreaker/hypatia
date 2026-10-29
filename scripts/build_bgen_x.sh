#!/bin/bash
# build_bgen_x.sh (10-27, docs/RL_PLAN.md H-MEET-ALL arm 5, B-GEN-X) -- after the four explanation-teacher parts (HV3_1423-1426) have results:
#   1. scratchpad/build_distill_set.py --keep-unlabeled on the teacher reads, with --inst-root = generated/medcalc_xw/train (the same
#      problems WITHOUT the explanation), so every rebuilt episode carries the explanation-free task -> results/pods/distill27genx/expl_raw.jsonl;
#   2. drops any episode whose assistant text refers to a reference / explanation / hint (B-GEN-H's amended filter + 'explanation',
#      'worked solution');
#   3. the union with B-GEN-H's set (results/pods/distill27genh/gen_h.jsonl) -> results/pods/distill27genx/gen_x.jsonl and its pool
#      generated/gen_d27x (symlinks); STEPS = 25 x round(2 N / 8 / 25) -> results/pods/distill27genx/STEPS. Refuses to overwrite.
set -euo pipefail
R=/project/pi_hongyu_umass_edu/hieutran/deep/deeprs; cd $R
S=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
G=/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated; O=results/pods/distill27genx
T=results/pods/harness_v3_q27/V3__rl-czero-medcalc_ver-think-q27base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s1_s100/ext
for h in a b c d; do [ -f $T/mc_expl_teach_$h/summary.json ] || { echo "teacher part $h has no summary.json yet"; exit 3; }; done
[ -e $O/gen_x.jsonl ] && { echo "$O/gen_x.jsonl exists -- nothing to do"; exit 0; }
mkdir -p $O; rm -rf $G/medcalc_genx_expl $O/expl_raw.jsonl
GBP_SYSTEM=science venv/bin/python $S/build_distill_set.py --keep-unlabeled --inst-root $G/medcalc_xw/train \
  --out $O/expl_raw.jsonl --pool $G/medcalc_genx_expl $T/mc_expl_teach_a $T/mc_expl_teach_b $T/mc_expl_teach_c $T/mc_expl_teach_d
venv/bin/python - <<'PY'
import ast, json, os, re
R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"; G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
O = f"{R}/results/pods/distill27genx"
REF = re.compile(r"\b(reference (note|formula|text|material|document|card|sheet|table|says|states|gives|provided)|the reference\b|"
                 r"re-?read(ing)? the reference|according to the (reference|hint|provided)|the hint|hint says|as hinted|"
                 r"provided (reference|formula|note|hint)|documentation provided|reference for this calculator|"
                 r"the formula (given|provided) (above|in the (prompt|task|reference))|the (worked )?explanation|worked solution|explanation (says|states|gives|provided)|"
                 r"according to the explanation|as (the )?explained)", re.I)
keep, drop = [], {"refers to a reference, explanation or hint": 0, "explanation in prompt": 0}
for l in open(f"{O}/expl_raw.jsonl"):
    r = json.loads(l)
    asst = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant" and isinstance(m["content"], str))
    prompt = " ".join(m["content"] for m in r["messages"] if m["role"] != "assistant" and isinstance(m["content"], str))
    if "Worked explanation of this problem" in prompt:
        drop["explanation in prompt"] += 1; continue
    if REF.search(asst):
        drop["refers to a reference, explanation or hint"] += 1; continue
    keep.append(l)
gen = [l for l in open(f"{R}/results/pods/distill27genh/gen_h.jsonl") if l.strip()]   # B-GEN-H's set (2,388)
names = {json.loads(l)["instance"] for l in gen}
keep = [l for l in keep if json.loads(l)["instance"] not in names]
open(f"{O}/gen_x.jsonl.tmp", "w").write("".join(gen) + "".join(keep))
os.rename(f"{O}/gen_x.jsonl.tmp", f"{O}/gen_x.jsonl")
pool = f"{G}/gen_d27x/train"; os.makedirs(pool)
for l in gen + keep:
    i = json.loads(l)["instance"]
    src = next(os.path.realpath(f"{G}/{p}/train/{i}") for p in ("gen_d27u", "medcalc_ver", "medcalc_xw") if os.path.exists(f"{G}/{p}/train/{i}"))
    assert os.path.isfile(f"{src}/eval_config.json"), i
    os.symlink(src, f"{pool}/{i}")
n = len(gen) + len(keep); steps = 25 * round(2 * n / 8 / 25)
open(f"{O}/STEPS", "w").write(str(steps))
json.dump({"gen_h": len(gen), "expl_raw": sum(1 for _ in open(f"{O}/expl_raw.jsonl")), "expl_kept": len(keep), "dropped": drop,
           "n": n, "steps": steps}, open(f"{O}/MANIFEST.json", "w"), indent=1)
print(json.dumps(json.load(open(f"{O}/MANIFEST.json"))))
PY
