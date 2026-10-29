"""build_mcweak.py (10-26, docs/RL_PLAN.md H-MEET-ALL 14:04, arm MCW9) -- the training pool mcweak3: all 2,152 MedCalc TRAINING problems
(generated/medcalc_ver/train) plus two extra copies of every TRAINING problem whose calculator the RL-trained 27B teacher answered usably on
fewer than 60% of its TRAINING problems (kept = results/pods/distill27fullu/medcalc_train_2152u.jsonl, the set B-GEN distils). The rule reads
training data only. Copies are extra symlinks (<id>__w1, <id>__w2) to the same instance, so those calculators are drawn three times as often.
Refuses to overwrite.   python3 scripts/build_mcweak.py"""
import ast, collections, json, os, sys

R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"
G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
SRC, OUT = f"{G}/medcalc_ver/train", f"{G}/mcweak3/train"
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists -- not overwriting")
ids = sorted(os.listdir(SRC))
assert len(ids) == 2152, len(ids)
kept = {json.loads(l)["instance"] for l in open(f"{R}/results/pods/distill27fullu/medcalc_train_2152u.jsonl") if l.strip()}
assert kept <= set(ids), "teacher set names do not match medcalc_ver/train"


def calc(i):
    cfg = json.load(open(f"{SRC}/{i}/eval_config.json")); p = cfg.get("_provenance")
    p = ast.literal_eval(p) if isinstance(p, str) else (p or {}); return p.get("calculator") or "?"


cal = {i: calc(i) for i in ids}
n, ok = collections.Counter(cal.values()), collections.Counter(cal[i] for i in kept)
weak = sorted(c for c in n if ok[c] / n[c] < 0.60)
os.makedirs(OUT)
extra = 0
for i in ids:
    src = os.path.realpath(f"{SRC}/{i}")
    os.symlink(src, f"{OUT}/{i}")
    if cal[i] in weak:
        for k in (1, 2):
            os.symlink(src, f"{OUT}/{i}__w{k}"); extra += 1
print(json.dumps({"train": len(ids), "weak_calculators": {c: f"{ok[c]}/{n[c]}" for c in weak}, "weak_problems": extra // 2,
                  "pool_size": len(ids) + extra, "pool": OUT}, indent=1))
