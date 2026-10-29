"""Adapt MedAgentBench into harness-compatible problems (healthcare benchmark: an agent working in a FHIR EHR).

SOURCE    github.com/stanfordmlgroup/MedAgentBench (Jiang, Black, Geng, Park, Zou, Ng, Chen; NEJM AI 2025,
          doi 10.1056/AIdbp2500144; arXiv 2501.14654). LICENSE MIT (repo). The EHR is the authors' Docker image
          jyxsu6/medagentbench (HAPI FHIR JPA starter v7.6.0, Apache-2.0, + a 4.4 GB H2 database of 695 synthetic-profile
          patients / 563,426 Observations); the graders are refsol.py, distributed separately via the Stanford Box link
          in the README (fetched here, pinned by sha1, kept under the git-ignored benchmarks/medagentbench/data/).
PINNED    repo REPO_REV (data/medagentbench/test_data_v2.json -- the file configs/tasks/medagentbench.yaml evaluates --
          and funcs_v1.json); image by digest (scripts/medagentbench_server.sh); refsol.py by sha1.
TASKS     n = 300 = 10 task types x 30: query tasks 1,2,4,6,7 (MRN lookup, age, last Mg in 24 h, mean glucose in 24 h,
          last glucose) and action tasks 3,5,8,9,10 (record a BP; order IV magnesium if low; orthopaedic referral;
          order potassium + a morning K draw if low; order an HbA1c if the last is >1 year old), each graded by refsol.

HOW IT RUNS HERE (all opt-in; nothing existing changed except the runner's SUBMISSION_FILES hook, see below)
  * scripts/medagentbench_server.sh start  -- runs the image under apptainer INSIDE the job that runs the harness
    (fresh writable overlay per start = pristine EHR; GET-only proxy in front; prints `export FHIR_API_BASE=...`).
  * the agent works in the usual Python sandbox and reaches the EHR through data_files/medagentbench_fhir.py:
    fhir_get() sends GETs exactly as the official harness does; fhir_post() RECORDS a POST (url + JSON payload) and
    answers with the official "POST request accepted..." -- the official harness never executes POSTs either, it grades
    them from the conversation. The record is fhir_posts.jsonl in the sandbox's working directory.
  * harness/runner.py attaches that file to the submission (the grader declares SUBMISSION_FILES -- an opt-in hook
    added for code/action benchmarks; no other grader declares it), and the final answer is the list the official
    agent would put in FINISH([...]).
  * reference_grader.py (= scripts/medagentbench_reference_grader.py) runs refsol's task function on an
    official-shaped (result, history) rebuilt from that answer and that record. The graders' own GETs (reference values)
    are answered from responses recorded here from the pristine server, so grading needs no server and is deterministic.

DIFFERENCES FROM THE OFFICIAL PROTOCOL (the tasks, EHR and graders are identical; the agent loop is ours):
  the official agent emits ONE `GET url` / `POST url + JSON` / `FINISH([...])` line per round for at most 8 rounds;
  ours writes Python cells (one cell may issue several GETs) for up to GBP_MAX_TURNS turns -- set GBP_MAX_TURNS=8 to
  match the round cap. Scores are therefore close to, not identical with, the published protocol.

PUBLISHED NUMBERS (arXiv 2501.14654 v2 main table; overall / query / action success rate; 8 rounds, temperature 0
  except o3-mini; which test-data file (v1/v2) the table used is not stated on the page -- UNVERIFIED):
      Claude 3.5 Sonnet v2 69.67 / 85.33 / 54.00     GPT-4o 64.00 / 72.00 / 56.00     DeepSeek-V3 62.67 / 70.67 / 54.67
      Gemini-1.5 Pro 62.00 / 52.67 / 71.33           GPT-4o-mini 56.33 / 59.33 / 53.33 o3-mini 51.67 / 54.67 / 48.67
      Qwen2.5 51.33 / 38.67 / 64.00                  Llama 3.3 46.33 / 50.00 / 42.67  Gemini 2.0 Flash 38.33
      Gemma2 19.33                                   Gemini 2.0 Pro 18.00             Mistral v0.3 4.00
  (model sizes for Qwen2.5 / Llama 3.3 / Gemma2 / Mistral are not given on that page -- UNVERIFIED.)

BUILD (needs a live server, i.e. a compute node; ~5 min):
  eval "$(scripts/medagentbench_server.sh start)"
  venv/bin/python scripts/fetch_medagentbench.py            # build + verify
  scripts/medagentbench_server.sh stop
  venv/bin/python scripts/fetch_medagentbench.py --verify-only    # replayed grading only, no server needed
EVALUATE (same job/node as the server):
  eval "$(scripts/medagentbench_server.sh start)"
  cd harness && GBP_MAX_TURNS=8 ../venv/bin/python runner.py --agent llm:MODEL --problems ../benchmarks/medagentbench \\
      --workers 8 --outdir OUT
  ../venv/bin/python ../scripts/fetch_medagentbench.py --report OUT     # overall / query / action success rates
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, data_root  # noqa: E402

REPO = "https://github.com/stanfordmlgroup/MedAgentBench.git"
REPO_REV = "99260117137b09f04837a8c18d18a1107efa55ae"          # 2025-11-21 "Update paper link"
REFSOL_URL = ("https://stanfordmedicine.app.box.com/index.php?rm=box_download_shared_file"
              "&shared_name=fizv0unyjgkb1r3a83rfn5p3dc673uho&file_id=f_1756907296228")
REFSOL_SHA1 = "1440e8b982059a3fc730a26f448465f549910b36"
CACHE = data_root() / "cache" / "medagentbench"
ROOT = PROJECT_ROOT / "benchmarks" / "medagentbench"
HERE = Path(__file__).resolve().parent
QUERY_TASKS = {"task1", "task2", "task4", "task6", "task7"}

TASK = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set \
of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the \
purpose.

You work in a Python notebook. The EHR is a FHIR R4 server whose base URL -- the api_base -- is in the environment \
variable FHIR_API_BASE (it ends with a slash). Use the helper module in data_files/:
    import sys; sys.path.insert(0, "data_files")
    from medagentbench_fhir import API_BASE, fhir_get, fhir_post
1. GET:  fhir_get(API_BASE + "Patient", {{"identifier": "S1234567"}}) returns the server's JSON response (a dict).
2. POST: fhir_post(API_BASE + "<ResourceType>", payload) with the payload as a JSON object. This is the ONLY way to make
   a POST: it is recorded and graded exactly as sent (URL and payload), and the server itself rejects direct writes.
   Every fhir_post() call counts, so do not post drafts or tests -- post only what you intend to order or record.
3. When you have got answers for all the questions and finished all the requested tasks, give the final answer -- the
   list the official harness expects in FINISH([answer1, answer2, ...]) -- as JSON:
   {{"answer": {{"{key}": [answer1, answer2, ...]}}}}

Here is a list of functions in JSON format that you can invoke. Note that you should use API_BASE as the api_base.
{functions}

Context: {context}
Question: {question}"""


def fetch() -> tuple[list, list]:
    repo = CACHE / "repo"
    if not (repo / ".git").exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", REPO_REV], check=True)
    ref = CACHE / f"refsol_{REFSOL_SHA1[:8]}.py"
    if not ref.exists():
        urllib.request.urlretrieve(REFSOL_URL, ref.with_suffix(".part"))
        ref.with_suffix(".part").rename(ref)
    got = hashlib.sha1(ref.read_bytes()).hexdigest()
    assert got == REFSOL_SHA1, f"refsol.py sha1 {got} != pinned {REFSOL_SHA1} (the Box file changed?)"
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    for src, dst in ((ref, ROOT / "data" / "refsol.py"),
                     (HERE / "medagentbench_fhir_client.py", ROOT / "data" / "medagentbench_fhir.py")):
        tmp = dst.with_suffix(".part")
        shutil.copy2(src, tmp)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(tmp, dst)
    shutil.copy2(HERE / "medagentbench_reference_grader.py", ROOT / "reference_grader.py")
    data = json.loads((repo / "data/medagentbench/test_data_v2.json").read_text())
    funcs = json.loads((repo / "data/medagentbench/funcs_v1.json").read_text())
    return data, funcs


def load_grader():
    spec = importlib.util.spec_from_file_location("mab_grader", ROOT / "reference_grader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# reference solution (the ORACLE): the correct FINISH list and POSTs by each task's rules, from the same GETs refsol
# makes. It exists to prove the grading pipeline end to end; refsol, not this, is the grader.
# --------------------------------------------------------------------------- #
NOW = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
AUTHORED = "2023-11-13T10:15:00+00:00"


def _obs(get, base, mrn, code):
    r = get(f"{base}Observation?patient={mrn}&code={code}&_count=5000&_format=json")
    return json.loads(r["data"]).get("entry", [])


def _last(entries, since=None):
    last_t = last_v = last_s = None
    for e in entries:
        t = datetime.fromisoformat(e["resource"]["effectiveDateTime"])
        if since is not None and t < since:
            continue
        if last_t is None or t > last_t:
            last_t, last_v, last_s = t, e["resource"]["valueQuantity"]["value"], e["resource"]["effectiveDateTime"]
    return last_t, last_v, last_s


def reference_solution(case: dict, get, base: str) -> tuple[list, list]:
    t, mrn = case["id"].split("_")[0], case.get("eval_MRN")
    subj = {"reference": f"Patient/{mrn}"}
    if t == "task1":
        return case["sol"], []
    if t == "task2":
        dob = datetime.strptime(json.loads(get(f"{base}Patient?identifier={mrn}&_format=json")["data"])
                                ["entry"][0]["resource"]["birthDate"], "%Y-%m-%d")
        today = datetime(2023, 11, 13)
        return [today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))], []
    if t == "task3":
        return [], [{"url": "Observation", "payload": {
            "resourceType": "Observation",
            "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category", "code": "vital-signs",
                                      "display": "Vital Signs"}]}],
            "code": {"text": "BP"}, "effectiveDateTime": AUTHORED, "status": "final", "valueString": "118/77 mmHg",
            "subject": subj}}]
    if t in ("task4", "task5"):
        _, v, _ = _last(_obs(get, base, mrn, "MG"), NOW - timedelta(hours=24))
        if t == "task4" or v is None or v > 1.9:
            return [v if v is not None else -1], []
        dose = 4 if v < 1 else 2 if v < 1.5 else 1
        return [v], [{"url": "MedicationRequest", "payload": {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"}]},
            "authoredOn": AUTHORED,
            "dosageInstruction": [{"route": "IV", "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "g"},
                                                                   "rateQuantity": {"value": dose, "unit": "h"}}]}],
            "status": "active", "intent": "order", "subject": subj}}]
    if t == "task6":
        vals = [e["resource"]["valueQuantity"]["value"] for e in _obs(get, base, mrn, "GLU")
                if datetime.fromisoformat(e["resource"]["effectiveDateTime"]) >= NOW - timedelta(hours=24)]
        return [sum(vals) / len(vals) if vals else -1], []
    if t == "task7":
        _, v, _ = _last(_obs(get, base, mrn, "GLU"))
        return [v if v is not None else -1], []
    if t == "task8":
        return [], [{"url": "ServiceRequest", "payload": {
            "resourceType": "ServiceRequest",
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "306181000000106"}]},
            "authoredOn": AUTHORED, "status": "active", "intent": "order", "priority": "stat",
            "note": {"text": "Situation: acute left knee injury, Background: radiology report indicates ACL tear. "
                             "Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to "
                             "evaluate and provide management recommendations."},
            "subject": subj}}]
    if t == "task9":
        _, v, _ = _last(_obs(get, base, mrn, "K"))
        if v is None or v >= 3.5:
            return [v if v is not None else -1], []
        return [v], [
            {"url": "MedicationRequest", "payload": {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc",
                                                          "code": "40032-917-01"}]},
                "authoredOn": AUTHORED,
                "dosageInstruction": [{"route": "oral", "doseAndRate": [{"doseQuantity": {
                    "value": (3.5 - v) / 0.1 * 10, "unit": "mEq"}}]}],
                "status": "active", "intent": "order", "subject": subj}},
            {"url": "ServiceRequest", "payload": {
                "resourceType": "ServiceRequest",
                "code": {"coding": [{"system": "http://loinc.org", "code": "2823-3"}]},
                "authoredOn": AUTHORED, "status": "active", "intent": "order", "priority": "stat",
                "subject": subj, "occurrenceDateTime": "2023-11-14T08:00:00+00:00"}}]
    if t == "task10":
        last_t, v, s = _last(_obs(get, base, mrn, "A1C"))
        finish = [-1] if v is None else [v, s]
        if v is not None and last_t >= datetime.fromisoformat("2022-11-13T10:15:00+00:00"):
            return finish, []
        return finish, [{"url": "ServiceRequest", "payload": {
            "resourceType": "ServiceRequest",
            "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
            "authoredOn": AUTHORED, "status": "active", "intent": "order", "priority": "stat", "subject": subj}}]
    raise ValueError(t)


def build(limit: int | None) -> int:
    base = os.environ.get("FHIR_API_BASE")
    if not base:
        sys.exit("FHIR_API_BASE is not set: run `eval \"$(scripts/medagentbench_server.sh start)\"` first "
                 "(on a compute node), or use --verify-only")
    data, funcs = fetch()
    assert len(data) == 300, len(data)
    if limit:
        data = data[:limit]
    g = load_grader()
    fn = json.dumps(funcs)
    if (ROOT / "problems").exists():
        shutil.rmtree(ROOT / "problems")
    disagree, n_bytes = [], 0
    for case in data:
        rec: dict = {}
        live_ns, _ = g.load_refsol({}, base, live=True, record_into=rec)
        finish, posts = reference_solution(case, live_ns["send_get_request"], base)
        gt = {"finish": finish, "fhir_posts": posts}
        res_posts = [dict(p, accepted=True) for p in posts]
        # refsol LIVE on the oracle (records every GET refsol itself makes), then REPLAYED from the record only
        live_ns2, _ = g.load_refsol({}, base, live=True, record_into=rec)
        v_live = g.official_eval(live_ns2, case, g.build_results(gt, res_posts, base), base)
        replay_ns, _ = g.load_refsol(rec, base)
        v_replay = g.official_eval(replay_ns, case, g.build_results(gt, res_posts, base), base)
        if not (v_live and v_replay):
            disagree.append((case["id"], v_live, v_replay))
        pid = f"mab-{case['id']}"
        key = f"{pid}-q1"
        pdir = ROOT / "problems" / pid
        (pdir / "data_files").mkdir(parents=True)
        os.link(ROOT / "data" / "medagentbench_fhir.py", pdir / "data_files" / "medagentbench_fhir.py")
        cfg = {
            "id": pid,
            "task": TASK.format(functions=fn, context=case["context"], question=case["instruction"], key=key),
            "data_files": ["medagentbench_fhir.py"],
            "ground_truth": {key: gt},        # the oracle's FINISH list + POSTs (inline form; see the grader)
            "grader": {"type": "medagentbench_refsol", "config": {"keys": {key: {
                "case": case, "recorded_gets": rec}}}},
            "_provenance": {"benchmark": "MedAgentBench", "repo": f"{REPO}@{REPO_REV}",
                            "data_file": "data/medagentbench/test_data_v2.json", "refsol_sha1": REFSOL_SHA1,
                            "task_type": case["id"].split("_")[0],
                            "kind": "query" if case["id"].split("_")[0] in QUERY_TASKS else "action",
                            "paper": "arXiv:2501.14654 / NEJM AI doi:10.1056/AIdbp2500144"},
        }
        text = json.dumps(cfg, indent=1)
        n_bytes += len(text)
        (pdir / "eval_config.json").write_text(text)
    assert not disagree, f"oracle not graded correct live AND replayed: {disagree[:5]}"
    print(f"[medagentbench] built {len(data)} problems ({n_bytes / 1e6:.1f} MB of configs); refsol grades the "
          f"reference solution correct on all of them, live and replayed")
    return len(data)


def verify() -> None:
    g = load_grader()
    cfgs = [json.loads(p.read_text()) for p in sorted((ROOT / "problems").glob("*/eval_config.json"))]
    ok, null_pass, junk_pass, nofinish = 0, [], [], 0
    for cfg in cfgs:
        key = next(iter(cfg["ground_truth"]))
        ok += g.evaluate(cfg, {"answer": dict(cfg["ground_truth"])})["passed"]
        if g.evaluate(cfg, {"answer": {key: 0}})["passed"]:                        # what the harness null agent sends
            null_pass.append(cfg["id"])
        junk = {"answer": {key: ["banana"]}, "_files": {"fhir_posts.jsonl": json.dumps(
            {"url": "http://localhost:8080/fhir/Observation", "accepted": True, "payload": {"x": 1},
             "api_base": "http://localhost:8080/fhir/"})}}
        if g.evaluate(cfg, junk)["passed"]:
            junk_pass.append(cfg["id"])
        nofinish += g.evaluate(cfg, {"answer": {}})["passed"]
    n = len(cfgs)
    assert ok == n, f"oracle passes only {ok}/{n}"
    assert nofinish == 0, "an episode with no FINISH answer passed"
    assert not junk_pass, f"a wrong answer + a bogus POST passed: {junk_pass[:5]}"
    # refsol.task5 grades ANY finished answer correct when no Mg was measured in the last 24 h and nothing was ordered
    # ("We don't care the finish"): the only way a do-nothing answer can pass. Check that is exactly what happens.
    odd = [i for i in null_pass if not i.startswith("mab-task5_")]
    assert not odd, f"the null answer passed outside the task5 no-measurement rule: {odd}"
    print(f"[medagentbench] {n} problems: oracle passes {ok}/{n} (replayed GETs, no server); a wrong answer with a bogus "
          f"POST passes 0; no FINISH passes 0; the null answer passes {len(null_pass)} -- all task5 cases with no Mg in "
          f"the last 24 h, which refsol.task5 grades correct for any finished answer without an order (official rule)")


def report(outdir: str) -> None:
    recs = [json.loads(f.read_text()) for f in Path(outdir).glob("mab-*.json")]
    by = {"query": [0, 0], "action": [0, 0]}
    for r in recs:
        kind = "query" if r["id"].split("-")[1].split("_")[0] in QUERY_TASKS else "action"
        by[kind][0] += bool(r["grade"]["passed"])
        by[kind][1] += 1
    tot = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    print(json.dumps({"n": tot[1], "success_rate": tot[0] / max(tot[1], 1),
                      **{f"{k}_success_rate": v[0] / max(v[1], 1) for k, v in by.items()},
                      **{f"n_{k}": v[1] for k, v in by.items()}}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--report", default=None, help="runner outdir -> overall / query / action success rates")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        sys.exit(0)
    if not a.verify_only:
        build(a.limit)
    verify()
