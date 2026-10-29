#!/bin/bash
# build_bgen_x3.sh (10-28, docs/RL_PLAN.md H-MEET-ALL arm 9, B-GEN-X3) -- after the eight teacher parts (HV3_1578-1585) have results:
#   1. scratchpad/build_distill_set.py --keep-unlabeled on the teacher reads IN ATTEMPT ORDER (mc_expl_teach4_a-d, then mc_expl_teach5_a-d), with
#      --inst-root = generated/medcalc_xwall3/train (the same problems WITHOUT the explanation) -> results/pods/distill27genx3/expl_raw.jsonl;
#   2. build_bgen_x.sh's filter, verbatim (no explanation in the rebuilt prompt; no mention of a reference / explanation / hint);
#   3. AT MOST ONE notebook per problem (the first kept attempt), none for a problem B-GEN-X2 already kept;
#   4. the union with B-GEN-X2's set (results/pods/distill27genx2/gen_x2.jsonl) -> results/pods/distill27genx3/gen_x3.jsonl (written last) and
#      its pool generated/gen_d27x3 (symlinks); STEPS = 25 x round(2 N / 8 / 25) -> results/pods/distill27genx3/STEPS. Refuses to overwrite.
set -euo pipefail
R=/project/pi_hongyu_umass_edu/hieutran/deep/deeprs; cd $R
S=/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad
G=/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated; O=results/pods/distill27genx3
T=results/pods/harness_v3_q27/V3__rl-czero-medcalc_ver-think-q27base-echo-sysscience-rc-rg-hv3pr0-lr1e-5-s1_s100/ext
DIRS=()
for k in 4 5; do for h in a b c d; do
  [ -f $T/mc_expl_teach${k}_$h/summary.json ] || { echo "teacher part ${k}$h has no summary.json yet"; exit 3; }
  DIRS+=($T/mc_expl_teach${k}_$h)
done; done
[ -e $O/gen_x3.jsonl ] && { echo "$O/gen_x3.jsonl exists -- nothing to do"; exit 0; }
[ -e $G/gen_d27x3 ] && { echo "$G/gen_d27x3 exists without $O/gen_x3.jsonl -- look before rebuilding"; exit 1; }
mkdir -p $O
GBP_SYSTEM=science venv/bin/python $S/build_distill_set.py --keep-unlabeled --inst-root $G/medcalc_xwall3/train \
  --out $O/expl_raw.jsonl --pool $G/medcalc_genx3_expl "${DIRS[@]}"
venv/bin/python - <<'PY'
import collections, json, os, re
R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"; G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
O = f"{R}/results/pods/distill27genx3"
REF = re.compile(r"\b(reference (note|formula|text|material|document|card|sheet|table|says|states|gives|provided)|the reference\b|"
                 r"re-?read(ing)? the reference|according to the (reference|hint|provided)|the hint|hint says|as hinted|"
                 r"provided (reference|formula|note|hint)|documentation provided|reference for this calculator|"
                 r"the formula (given|provided) (above|in the (prompt|task|reference))|the (worked )?explanation|worked solution|explanation (says|states|gives|provided)|"
                 r"according to the explanation|as (the )?explained)", re.I)   # build_bgen_x.sh's, verbatim
gen = [l for l in open(f"{R}/results/pods/distill27genx2/gen_x2.jsonl") if l.strip()]   # B-GEN-X2's set (3,109)
names = {json.loads(l)["instance"] for l in gen}
keep, seen = [], set()
drop = {"explanation in prompt": 0, "refers to a reference, explanation or hint": 0, "problem already kept in B-GEN-X2": 0,
        "a later attempt of a kept problem": 0}
for l in open(f"{O}/expl_raw.jsonl"):
    r = json.loads(l)
    asst = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant" and isinstance(m["content"], str))
    prompt = " ".join(m["content"] for m in r["messages"] if m["role"] != "assistant" and isinstance(m["content"], str))
    if "Worked explanation of this problem" in prompt:
        drop["explanation in prompt"] += 1; continue
    if REF.search(asst):
        drop["refers to a reference, explanation or hint"] += 1; continue
    if r["instance"] in names:
        drop["problem already kept in B-GEN-X2"] += 1; continue
    if r["instance"] in seen:
        drop["a later attempt of a kept problem"] += 1; continue
    seen.add(r["instance"]); keep.append(l)
tmp = f"{G}/gen_d27x3.tmp"; pool = f"{tmp}/train"; os.makedirs(pool, exist_ok=True)
for l, base in [(l, f"{G}/gen_d27x2/train") for l in gen] + [(l, f"{G}/medcalc_xwall3/train") for l in keep]:
    i = json.loads(l)["instance"]
    src = os.path.realpath(f"{base}/{i}")
    assert os.path.isfile(f"{src}/eval_config.json"), i
    if not os.path.lexists(f"{pool}/{i}"):
        os.symlink(src, f"{pool}/{i}")
os.rename(tmp, f"{G}/gen_d27x3")
calc = collections.Counter(json.load(open(f"{G}/medcalc_xwall3/train/{json.loads(l)['instance']}/eval_config.json"))["_provenance"]["calculator"]
                           for l in keep)
n = len(gen) + len(keep); steps = 25 * round(2 * n / 8 / 25)
open(f"{O}/STEPS", "w").write(str(steps))
json.dump({"gen_x2": len(gen), "expl_raw": sum(1 for _ in open(f"{O}/expl_raw.jsonl")), "kept": len(keep),
           "kept_reattempted": sum(1 for l in keep if not json.loads(l)["instance"].startswith("mcx3-")),
           "kept_new_rows": sum(1 for l in keep if json.loads(l)["instance"].startswith("mcx3-")),
           "kept_per_calculator": dict(calc.most_common()), "dropped": drop, "n": n, "steps": steps},
          open(f"{O}/MANIFEST.json", "w"), indent=1)
open(f"{O}/gen_x3.jsonl.tmp", "w").write("".join(gen) + "".join(keep))
os.rename(f"{O}/gen_x3.jsonl.tmp", f"{O}/gen_x3.jsonl")   # last: its existence means the build is complete
print(json.dumps(json.load(open(f"{O}/MANIFEST.json"))))
PY
