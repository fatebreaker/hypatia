"""mabgen_build.py (10-17, H-GEN27 clean training data) -- MedAgentBench-style TRAINING tasks on patients that no test task uses.

MedAgentBench's FHIR sandbox holds 695 patients; its 300 test tasks name 99 of them (scratchpad/mabgen/test_mrns.json: every
eval_MRN and every MRN in an instruction). This builds N_PER training tasks per task type (task1..task10) on the OTHER patients,
with the test tasks' own wording (the instruction and context of that type's first test case, with the patient substituted),
and grades them exactly as the test tasks are graded: the official refsol run on GETs recorded here from the PRISTINE server
(scripts/fetch_medagentbench.py's machinery, imported, unchanged). task1 ('MRN of the patient with name X and DOB Y') also gets
some patients that do not exist (answer "Patient not found"), as the test set has. Every oracle answer is checked to pass refsol
live and replayed. Output: generated/mabgen/train/<id>/ (eval_config.json + data_files/medagentbench_fhir.py) + MANIFEST.json.
Needs a pristine server on this node:
    eval "$(scripts/medagentbench_server.sh start)"; venv/bin/python scripts/mabgen_build.py; scripts/medagentbench_server.sh stop
"""
import copy
import importlib.util
import json
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path

R = Path(__file__).resolve().parent.parent
S = "/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad"
OUT = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated/mabgen")
N_PER, N_NOTFOUND, SEED = 60, 4, 20261017
spec = importlib.util.spec_from_file_location("fetch_mab", R / "scripts" / "fetch_medagentbench.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

base = os.environ.get("FHIR_API_BASE") or sys.exit("FHIR_API_BASE is not set (start the pristine server first)")
if OUT.exists():
    sys.exit(f"{OUT} exists -- not overwriting")
test_mrns = set(json.load(open(f"{S}/mabgen/test_mrns.json")))


def get(path):
    with urllib.request.urlopen(path if path.startswith("http") else base + path, timeout=120) as r:
        return json.loads(r.read())


# every patient on the server (paged)
pts, url = [], base + "Patient?_count=500&_format=json"
while url:
    d = get(url)
    pts += [e["resource"] for e in d.get("entry", [])]
    url = next((l["url"] for l in d.get("link", []) if l.get("relation") == "next"), None)
info = {}
for p in pts:
    mrn = next((i["value"] for i in p.get("identifier", []) if i.get("value")), None)
    name = (p.get("name") or [{}])[0]
    info[mrn] = {"given": " ".join(name.get("given", [])), "family": name.get("family", ""), "dob": p.get("birthDate")}
train_mrns = sorted(m for m in info if m and m not in test_mrns and info[m]["dob"] and info[m]["family"])
print(f"{len(info)} patients on the server; {len(test_mrns)} named by test tasks; {len(train_mrns)} available for training")

# the test tasks' wording, one template per type (from the built test problems)
tmpl = {}
for pdir in sorted((R / "benchmarks" / "medagentbench" / "problems").iterdir()):
    cfg = json.loads((pdir / "eval_config.json").read_text())
    case = next(iter(cfg["grader"]["config"]["keys"].values()))["case"]
    t = case["id"].split("_")[0]
    tmpl.setdefault(t, case)
funcs = json.loads((F.CACHE / "repo" / "data" / "medagentbench" / "funcs_v1.json").read_text())   # the cached repo; F.fetch()
#                                    would re-copy files into the live benchmark folder, which we do not touch
fn = json.dumps(funcs)
g = F.load_grader()
rng = random.Random(SEED)
first_names = sorted({v["given"] for v in info.values() if v["given"]})
last_names = sorted({v["family"] for v in info.values() if v["family"]})
existing = {(v["given"], v["family"], v["dob"]) for v in info.values()}

cases = []
for t in sorted(tmpl, key=lambda x: int(x[4:])):
    c0 = tmpl[t]
    chosen = rng.sample(train_mrns, N_PER)
    for k, mrn in enumerate(chosen):
        c = copy.deepcopy(c0)
        c["id"] = f"{t}_t{k + 1:03d}"
        old = c0.get("eval_MRN")
        if t == "task1":
            pi = info[mrn]
            if k < N_NOTFOUND:            # a patient that does not exist: real-looking name and DOB, no match on the server
                while True:
                    gvn, fam = rng.choice(first_names), rng.choice(last_names)
                    dob = f"{rng.randint(1930, 2005)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                    if (gvn, fam, dob) not in existing:
                        break
                sol = ["Patient not found"]
            else:
                gvn, fam, dob = pi["given"], pi["family"], pi["dob"]
                sol = [mrn]
            o = tmpl["task1"]
            oi = info[o["sol"][0]] if o["sol"][0] in info else None
            ins = o["instruction"]
            if oi:
                ins = ins.replace(f"{oi['given']} {oi['family']}", f"{gvn} {fam}").replace(oi["dob"], dob)
            c["instruction"], c["sol"], c["eval_MRN"] = ins, sol, sol[0] if sol[0] != "Patient not found" else None
            assert ins != o["instruction"], "task1 template substitution failed"
        else:
            c["instruction"] = c0["instruction"].replace(old, mrn)
            c["context"] = str(c0.get("context") or "").replace(old, mrn)
            c["eval_MRN"] = mrn
            assert mrn in c["instruction"], (t, c["instruction"][:120])
        cases.append(c)

(OUT / "train").mkdir(parents=True)
disagree = []
for case in cases:
    rec: dict = {}
    live_ns, _ = g.load_refsol({}, base, live=True, record_into=rec)
    finish, posts = F.reference_solution(case, live_ns["send_get_request"], base)
    gt = {"finish": finish, "fhir_posts": posts}
    res_posts = [dict(p, accepted=True) for p in posts]
    live_ns2, _ = g.load_refsol({}, base, live=True, record_into=rec)
    v_live = g.official_eval(live_ns2, case, g.build_results(gt, res_posts, base), base)
    replay_ns, _ = g.load_refsol(rec, base)
    v_replay = g.official_eval(replay_ns, case, g.build_results(gt, res_posts, base), base)
    if not (v_live and v_replay):
        disagree.append((case["id"], v_live, v_replay))
    pid = f"mabt-{case['id']}"
    key = f"{pid}-q1"
    pdir = OUT / "train" / pid
    (pdir / "data_files").mkdir(parents=True)
    shutil.copy(R / "benchmarks" / "medagentbench" / "data" / "medagentbench_fhir.py", pdir / "data_files" / "medagentbench_fhir.py")
    cfg = {"id": pid, "task": F.TASK.format(functions=fn, context=case["context"], question=case["instruction"], key=key),
           "data_files": ["medagentbench_fhir.py"], "ground_truth": {key: gt},
           "grader": {"type": "medagentbench_refsol", "config": {"keys": {key: {"case": case, "recorded_gets": rec}}}},
           "_provenance": {"benchmark": "MedAgentBench-style TRAINING task (H-GEN27)", "source_template": tmpl[case["id"].split("_")[0]]["id"],
                           "patient_not_in_any_test_task": True, "refsol_sha1": F.REFSOL_SHA1}}
    (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
assert not disagree, f"oracle not graded correct live AND replayed: {disagree[:5]}"
json.dump({"built": "2026-10-17", "seed": SEED, "n_per_type": N_PER, "n_notfound_task1": N_NOTFOUND,
           "n_patients_server": len(info), "test_mrns_excluded": len(test_mrns), "n_tasks": len(cases)},
          open(OUT / "MANIFEST.json", "w"), indent=1)
print(f"built {len(cases)} MedAgentBench-style training tasks in {OUT}; the oracle passes refsol live and replayed on all")
