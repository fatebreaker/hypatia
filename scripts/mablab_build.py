"""mablab_build.py (10-19, user: 'fill both data gaps') -- MedAgentBench LAB-task TRAINING tasks on the synthetic lab histories of
scripts/mablab_synth.py (300 patients that no test task and no generated/mabgen2 task names; 50 per lab task type).

Run on a node with a FRESH server whose overlay then receives the synthetic records (scripts/mablab_load.py): each task uses the
test tasks' own wording for its type (first test case, patient substituted, as mabgen_build2.py), the official refsol is the oracle
(fetch_medagentbench.reference_solution), its GETs against the AUGMENTED server are recorded, and the task is kept only if the
oracle passes refsol live and replayed AND its outcome (measured or not; number of orders) is the one planned for that patient.
Grading replays the recorded GETs, so it needs no server; training rollouts need the augmented server (the item loads the records).
Output: generated/mablab/train/mabl-<type>_<k>/ (eval_config.json + data_files/medagentbench_fhir.py) + MANIFEST.json.
    eval "$(scripts/medagentbench_server.sh start)"   # with MAB_STATE_DIR set
    venv/bin/python scripts/mablab_build.py; scripts/medagentbench_server.sh stop
"""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
OUT = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated/mablab")
spec = importlib.util.spec_from_file_location("fetch_mab", R / "scripts" / "fetch_medagentbench.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
spec2 = importlib.util.spec_from_file_location("mablab_load", R / "scripts" / "mablab_load.py")
LD = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(LD)

base = os.environ.get("FHIR_API_BASE") or sys.exit("FHIR_API_BASE is not set (start a fresh server first)")
state = os.environ.get("MAB_STATE_DIR") or sys.exit("MAB_STATE_DIR is not set")
if (OUT / "train").exists():
    sys.exit(f"{OUT / 'train'} exists -- not overwriting")
recs_blob = (OUT / "synthetic_labs.json").read_text()
recs = json.loads(recs_blob)
plan = json.load(open(OUT / "plan.json"))
assert plan["sha1"] == hashlib.sha1(recs_blob.encode()).hexdigest(), "plan.json does not match synthetic_labs.json"
n = LD.load(recs, state, base)
print(f"loaded {n} synthetic lab records into this server (overlay; discarded at stop)")

tmpl = {}
for pdir in sorted((R / "benchmarks" / "medagentbench" / "problems").iterdir()):
    cfg = json.loads((pdir / "eval_config.json").read_text())
    case = next(iter(cfg["grader"]["config"]["keys"].values()))["case"]
    tmpl.setdefault(case["id"].split("_")[0], case)
funcs = json.loads((F.CACHE / "repo" / "data" / "medagentbench" / "funcs_v1.json").read_text())
fn = json.dumps(funcs)
g = F.load_grader()
EXPECT = {("task4", "value"): "value/0", ("task4", "none_old"): "none/0", ("task4", "none_absent"): "none/0",
          ("task5", "low"): "value/1", ("task5", "normal"): "value/0", ("task5", "none_old"): "none/0", ("task5", "none_absent"): "none/0",
          ("task6", "value"): "value/0", ("task6", "none_old"): "none/0",
          ("task7", "value"): "value/0", ("task7", "none_absent"): "none/0",
          ("task9", "low"): "value/2", ("task9", "normal"): "value/0",
          ("task10", "recent"): "value/0", ("task10", "old"): "value/1", ("task10", "none_absent"): "none/1"}

(OUT / "train").mkdir(parents=True)
bad, kept, counts = [], 0, {}
k_by_type = {}
for p in plan["plan"]:
    t, mrn = p["type"], p["mrn"]
    c0 = tmpl[t]
    c = copy.deepcopy(c0)
    k_by_type[t] = k_by_type.get(t, 0) + 1
    c["id"] = f"{t}_l{k_by_type[t]:03d}"
    old = c0.get("eval_MRN")
    c["instruction"] = c0["instruction"].replace(old, mrn)
    c["context"] = str(c0.get("context") or "").replace(old, mrn)
    c["eval_MRN"] = mrn
    assert mrn in c["instruction"]
    rec: dict = {}
    live_ns, _ = g.load_refsol({}, base, live=True, record_into=rec)
    finish, posts = F.reference_solution(c, live_ns["send_get_request"], base)
    outcome = ("none" if finish in ([-1], []) else "value") + f"/{len(posts)}"
    gt = {"finish": finish, "fhir_posts": posts}
    res_posts = [dict(x, accepted=True) for x in posts]
    live_ns2, _ = g.load_refsol({}, base, live=True, record_into=rec)
    v_live = g.official_eval(live_ns2, c, g.build_results(gt, res_posts, base), base)
    replay_ns, _ = g.load_refsol(rec, base)
    v_replay = g.official_eval(replay_ns, c, g.build_results(gt, res_posts, base), base)
    if not (v_live and v_replay) or outcome != EXPECT[(t, p["planned"])]:
        bad.append((c["id"], mrn, p["planned"], outcome, v_live, v_replay))
        continue
    pid = f"mabl-{c['id']}"
    key = f"{pid}-q1"
    pdir = OUT / "train" / pid
    (pdir / "data_files").mkdir(parents=True)
    shutil.copy(R / "benchmarks" / "medagentbench" / "data" / "medagentbench_fhir.py", pdir / "data_files" / "medagentbench_fhir.py")
    cfg = {"id": pid, "task": F.TASK.format(functions=fn, context=c["context"], question=c["instruction"], key=key),
           "data_files": ["medagentbench_fhir.py"], "ground_truth": {key: gt},
           "grader": {"type": "medagentbench_refsol", "config": {"keys": {key: {"case": c, "recorded_gets": rec}}}},
           "_provenance": {"benchmark": "MedAgentBench-style LAB TRAINING task on synthetic lab records (H-MAB-LAB27)",
                           "source_template": c0["id"], "patient_not_in_any_test_task": True, "planned_outcome": p["planned"],
                           "outcome": outcome, "synthetic_labs_sha1": plan["sha1"], "refsol_sha1": F.REFSOL_SHA1,
                           "needs_server": "medagentbench_server.sh start + scripts/mablab_load.py (training rollouts only)"}}
    (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
    kept += 1
    counts[f"{t} {outcome}"] = counts.get(f"{t} {outcome}", 0) + 1
json.dump({"built": "2026-10-19", "n_tasks": kept, "outcomes": counts, "dropped": bad, "synthetic_labs_sha1": plan["sha1"]},
          open(OUT / "MANIFEST.json", "w"), indent=1)
print(f"built {kept} lab training tasks in {OUT}/train; outcomes {counts}; dropped {len(bad)}: {bad[:8]}")
