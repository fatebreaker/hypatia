"""FHIR access for a MedAgentBench task (shipped to the agent as data_files/medagentbench_fhir.py).

    import sys; sys.path.insert(0, "data_files")
    from medagentbench_fhir import API_BASE, fhir_get, fhir_post

    fhir_get("Patient", {"identifier": "S1234567"})           # GET {API_BASE}Patient?identifier=S1234567&_format=json
    fhir_post(API_BASE + "Observation", {...})                  # RECORDED, not executed (see below)

GET  -- sent to the live FHIR server exactly as the official harness sends it (the URL plus `_format=json`); returns
        the parsed JSON (a dict), or {"error": "..."} on failure.
POST -- as in the official MedAgentBench harness, a POST is NEVER executed: it is checked for a JSON payload, recorded,
        and answered with the official acceptance message. The grader reads the record (fhir_posts.jsonl in the working
        directory). The server itself refuses POST/PUT/DELETE. Relative URLs ("Observation") are resolved against
        API_BASE; the grader compares the full URL with API_BASE + "<ResourceType>", as the official grader does.

The harness sandbox re-runs every earlier notebook cell each time a new cell runs, so a naive log would repeat each
POST once per later cell. The record is therefore rebuilt on every run of the notebook: it always holds exactly the
POSTs that one pass through the current notebook makes.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import time

import requests

API_BASE = os.environ.get("FHIR_API_BASE", "")
POST_LOG = os.path.abspath("fhir_posts.jsonl")     # the sandbox working directory, where the harness collects it
ACCEPTED = ("POST request accepted and executed successfully. Please call FINISH if you have got answers for all the "
            "questions and finished all the requested tasks")
INVALID = "Invalid POST request"

# one id per notebook run (one process), kept across importlib.reload
if not hasattr(sys, "_medagentbench_run_id"):
    sys._medagentbench_run_id = f"{os.getpid()}-{time.time_ns()}"

    @atexit.register
    def _drop_stale_record() -> None:
        """A run that made no POST leaves no record (e.g. the only posting cell timed out and left the notebook)."""
        try:
            with open(POST_LOG) as fh:
                first = fh.readline()
            if first.strip() and json.loads(first).get("run") != sys._medagentbench_run_id:
                os.remove(POST_LOG)
        except (OSError, ValueError):
            pass


def _full(url: str) -> str:
    if not API_BASE:
        raise RuntimeError("FHIR_API_BASE is not set: start the server with scripts/medagentbench_server.sh")
    url = str(url).strip()
    return url if url.startswith(("http://", "https://")) else API_BASE + url.lstrip("/")


def fhir_get(url: str, params: dict | None = None) -> dict:
    """GET {url}[?params]&_format=json -> parsed JSON dict (or {"error": ...})."""
    try:
        r = requests.get(_full(url), params={**(params or {}), "_format": "json"}, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}


def _record(entry: dict) -> None:
    run = sys._medagentbench_run_id
    lines = []
    if os.path.exists(POST_LOG):
        with open(POST_LOG) as fh:
            lines = [ln for ln in fh if ln.strip()]
        if not lines or json.loads(lines[0]).get("run") != run:
            lines = []                                           # a new notebook run: rebuild the record
    entry = {"run": run, "api_base": API_BASE, **entry}
    lines.append(json.dumps(entry) + "\n")
    with open(POST_LOG + ".tmp", "w") as fh:
        fh.writelines(lines)
    os.replace(POST_LOG + ".tmp", POST_LOG)


def fhir_post(url: str, payload) -> str:
    """Record a POST {url} with a JSON payload (dict, list or JSON string). Returns the official reply."""
    full = _full(url)
    try:
        body = json.loads(payload) if isinstance(payload, str) else json.loads(json.dumps(payload))
    except Exception:                                            # noqa: BLE001
        _record({"url": full, "accepted": False, "raw": str(payload)[:20000]})
        return INVALID
    _record({"url": full, "accepted": True, "payload": body})
    return ACCEPTED
