"""MedAgentBench grading (copied into benchmarks/medagentbench/reference_grader.py by scripts/fetch_medagentbench.py).

THE OFFICIAL GRADERS, UNCHANGED: refsol.py (task1..task10) as the authors distribute it (Stanford Box link in the
MedAgentBench README; pinned by sha1 REFSOL_SHA1, downloaded by the fetch script to data/refsol.py and NOT
redistributed here). Each refsol task function is called exactly as the official `eval()` calls it -- (case_data,
results, fhir_api_base) -- and a task passes iff it returns True; the official `calculate_overall` only grades episodes
that FINISHed, so no answer = fail. Two inputs are rebuilt for our Python-sandbox agent:

  results.result   the text the official agent puts inside FINISH(...): our JSON answer's list, json.dumps()-ed
                   (a string value is used verbatim, as FINISH's text would be).
  results.history  the official conversation shape refsol inspects: for every POST the agent RECORDED with
                   data_files/medagentbench_fhir.py (harness/runner.py attaches fhir_posts.jsonl -- the
                   SUBMISSION_FILES opt-in), an agent turn "POST <url>\\n<payload JSON>" followed by the official
                   "POST request accepted and executed successfully..." (or "Invalid POST request") reply, then the
                   agent's FINISH turn. So refsol's extract_posts()/check_has_post() run as written.
  fhir_api_base    the base URL the agent's episode used (recorded with each POST; else the official default).

THE GRADERS' OWN GET REQUESTS (the reference values: last magnesium in 24 h, average glucose, ...) are answered from
responses recorded from the PRISTINE server at build time (grader config "recorded_gets", keyed by URL without the
base). Grading is therefore deterministic, needs no running server, and cannot be moved by an episode that reached
the server some other way. fetch_medagentbench.py --verify checks refsol gives the same verdict live and replayed.
Set MAB_LIVE_GETS=1 (with FHIR_API_BASE) to let an unrecorded GET go to the live server instead of failing.

refsol's own print() calls are captured into the grade details instead of the runner's stdout.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFSOL = HERE / "data" / "refsol.py"
REFSOL_SHA1 = "1440e8b982059a3fc730a26f448465f549910b36"
OFFICIAL_BASE = "http://localhost:8080/fhir/"          # configs/tasks/medagentbench.yaml
ACCEPTED = ("POST request accepted and executed successfully. Please call FINISH if you have got answers for all the "
            "questions and finished all the requested tasks")
INVALID = "Invalid POST request"

SUBMISSION_FILES = ("fhir_posts.jsonl",)               # attached by harness/runner.py (opt-in)


def _refsol_source() -> str:
    raw = REFSOL.read_bytes()
    if hashlib.sha1(raw).hexdigest() != REFSOL_SHA1:
        raise RuntimeError(f"{REFSOL} is not the pinned refsol.py (sha1 {REFSOL_SHA1}); rerun fetch_medagentbench.py")
    src = raw.decode()
    assert "from .utils import *" in src
    return src.replace("from .utils import *", "# (utils names injected by the harness grader)")


def _key(url: str, base: str) -> str:
    return url[len(base):] if url.startswith(base) else url


def load_refsol(recorded: dict, base: str, live: bool = False, record_into: dict | None = None):
    """A fresh refsol namespace (thread-safe: nothing shared between calls) whose GETs are replayed/recorded."""
    out: list[str] = []

    def send_get_request(url, params=None, headers=None):
        k = _key(url, base)
        if not live and k in recorded:
            return {"status_code": 200, "data": recorded[k]}
        if live or os.environ.get("MAB_LIVE_GETS") == "1":
            import requests                                  # official utils.send_get_request, verbatim logic
            try:
                live_url = url if live else os.environ["FHIR_API_BASE"] + k
                response = requests.get(live_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json() if response.headers.get('Content-Type') == 'application/json' else response.text
                if record_into is not None:
                    record_into[k] = data
                return {"status_code": response.status_code, "data": data}
            except Exception as e:                           # noqa: BLE001
                return {"error": str(e)}
        return {"error": f"GET not recorded at build time: {k}"}

    def verify_fhir_server(fhir_api_base):
        return True

    ns = {"__name__": "medagentbench_refsol", "send_get_request": send_get_request,
          "verify_fhir_server": verify_fhir_server,
          "print": lambda *a, **kw: out.append(" ".join(str(x) for x in a))}
    exec(compile(_refsol_source(), str(REFSOL), "exec"), ns)
    return ns, out


def official_eval(ns, case_data, results, fhir_api_base) -> bool:
    """src/server/tasks/medagentbench/eval.py::eval, verbatim logic."""
    task_id = case_data['id'].split('_')[0]
    grader_func = ns[task_id]
    try:
        if grader_func(case_data, results, fhir_api_base) is True:
            return True
    except Exception as e:                                   # noqa: BLE001
        ns["print"](e)
        return False
    return False


def _posts_from(obj_value, files: dict) -> tuple[list[dict], str]:
    if "fhir_posts.jsonl" in files:
        recs = []
        for line in files["fhir_posts.jsonl"].splitlines():
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    recs.append({"url": "?", "accepted": False, "raw": line})
        return recs, "file"
    if isinstance(obj_value, dict) and "fhir_posts" in obj_value:          # inline form (oracle / other harnesses)
        return [dict(p, accepted=p.get("accepted", True)) for p in obj_value["fhir_posts"]], "inline"
    return [], "none"


def build_results(value, posts: list[dict], base: str):
    """(results, fhir_api_base) in the shape refsol expects."""
    if isinstance(value, dict) and "finish" in value:
        value = value["finish"]
    result = None if value is None else (value if isinstance(value, str) else json.dumps(value))
    msg = types.SimpleNamespace
    history = []
    for p in posts:
        url = str(p.get("url", ""))
        if not url.startswith(("http://", "https://")):
            url = base + url.lstrip("/")
        if p.get("accepted", True) and "payload" in p:
            history += [msg(role="agent", content=f"POST {url}\n{json.dumps(p['payload'])}"),
                        msg(role="user", content=ACCEPTED)]
        else:
            history += [msg(role="agent", content=f"POST {url}\n{p.get('raw', '')}"),
                        msg(role="user", content=INVALID)]
    if result is not None:
        history.append(msg(role="agent", content=f"FINISH({result})"))
    return types.SimpleNamespace(result=result, history=history)


def evaluate(config: dict, submission: dict) -> dict:
    """Harness entry point -- same contract as the other vendored graders."""
    grader = config.get("grader") or {}
    if grader.get("type") != "medagentbench_refsol":
        raise ValueError(f"unsupported grader type: {grader.get('type')}")
    gcfg = grader.get("config") or {}
    obj = submission or {}
    files = obj.get("_files") or {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    details, correct = [], 0
    for k, spec in (gcfg.get("keys") or {}).items():
        value = obj.get(k)
        posts, src = _posts_from(value, files)
        base = next((p["api_base"] for p in posts if p.get("api_base")), None) or OFFICIAL_BASE
        results = build_results(value, posts, base)
        ns, out = load_refsol(spec["recorded_gets"], base)
        ok = results.result is not None and official_eval(ns, spec["case"], results, base)
        correct += bool(ok)
        details.append({"key": k, "task": spec["case"]["id"], "correct": bool(ok), "finished": results.result is not None,
                        "finish": results.result, "n_posts": len(posts), "posts_from": src,
                        "refsol_output": out[-10:]})
    n = max(len(details), 1)
    return {"eval_id": config.get("id"), "grader_type": "medagentbench_refsol",
            "score": correct / n, "passed": correct == n, "correct": correct, "total": n, "details": details}
