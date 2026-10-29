"""build_mc_hint_teach.py (10-25, docs/RL_PLAN.md H-BEAT-CLAUDE arm 3, B-GEN-H) -- the MedCalc TRAINING problems the RL-trained 27B teacher
did not answer usably (the 2,152 of medcalc_train_1000_a/_b + medcalc_train_rest_a/_b minus the 1,608 kept in B-FULL-U's set,
results/pods/distill27fullu/medcalc_train_2152u.jsonl), each copied with its calculator's TRAINING-split formula hint appended to the task
(results/pods/hints/medcalc_formula_hints.json; rl/grpo_hint.py's HINT_HEAD, exactly), into generated/medcalc_hint_teach/train, and two
teacher wrappers benchmarks/mc_hint_teach_a / _b (both trees; problems/ symlinks, the MedCalc grader). Problems whose calculator has no
hint are listed and skipped. Refuses to overwrite.   python3 scripts/build_mc_hint_teach.py"""
import ast, json, os, shutil, sys
R = "/project/pi_hongyu_umass_edu/hieutran/deep/deeprs"
G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
TREES = (f"{R}/benchmarks", "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/deeprs_benchmarks")
HINT_HEAD = ("\n\nReference for this calculator (documentation for your own use; do not mention or quote this note in "
             "your answer):\n")   # rl/grpo_hint.py, verbatim
src = open(f"{R}/rl/grpo_hint.py").read()
assert 'HINT_HEAD = ("\\n\\nReference for this calculator (documentation for your own use; do not mention or quote this note in "' in src
OUT = f"{G}/medcalc_hint_teach/train"
for x in [OUT] + [f"{t}/mc_hint_teach_{h}" for t in TREES for h in "ab"]:
    if os.path.exists(x):
        sys.exit(f"{x} exists -- not overwriting")
allp = []
for b in ("medcalc_train_1000_a", "medcalc_train_1000_b", "medcalc_train_rest_a", "medcalc_train_rest_b"):
    d = f"{R}/benchmarks/{b}/problems"
    allp += [(i, os.path.realpath(f"{d}/{i}")) for i in sorted(os.listdir(d))]
assert len(allp) == 2152 and len({i for i, _ in allp}) == 2152, len(allp)
kept = {json.loads(l)["instance"] for l in open(f"{R}/results/pods/distill27fullu/medcalc_train_2152u.jsonl") if l.strip()}
hints = json.load(open(f"{R}/results/pods/hints/medcalc_formula_hints.json"))["hints"]
fail = [(i, p) for i, p in allp if i not in kept]
os.makedirs(OUT)
made, nohint = [], {}
for i, p in fail:
    cfg = json.load(open(f"{p}/eval_config.json"))
    prov = cfg.get("_provenance"); prov = ast.literal_eval(prov) if isinstance(prov, str) else (prov or {})
    calc = prov.get("calculator") if isinstance(prov, dict) else None
    if not calc or calc not in hints:
        nohint[calc] = nohint.get(calc, 0) + 1; continue
    d = f"{OUT}/{i}"; os.makedirs(d)
    cfg["task"] = cfg["task"] + HINT_HEAD + hints[calc].strip()
    json.dump(cfg, open(f"{d}/eval_config.json", "w"))
    if os.path.isdir(f"{p}/data_files"):
        shutil.copytree(f"{p}/data_files", f"{d}/data_files")
    made.append(i)
half = (len(made) + 1) // 2
for t in TREES:
    for h, part in (("a", made[:half]), ("b", made[half:])):
        w = f"{t}/mc_hint_teach_{h}"; os.makedirs(f"{w}/problems")
        for i in part:
            os.symlink(f"{OUT}/{i}", f"{w}/problems/{i}")
        os.symlink(f"{R}/benchmarks/medcalc_ver/reference_grader.py", f"{w}/reference_grader.py")
json.dump({"built": "2026-10-25", "registration": "docs/RL_PLAN.md H-BEAT-CLAUDE arm 3 (2026-10-25 21:30 UTC)", "train_total": 2152,
           "kept_by_teacher": len(kept), "failed": len(fail), "hinted": len(made), "no_hint": nohint, "instances": made},
          open(f"{G}/medcalc_hint_teach/MANIFEST.json", "w"), indent=1)
print(f"failed {len(fail)}, hinted {len(made)} (a {half}, b {len(made) - half}), no hint {sum(nohint.values())}: {nohint}")
