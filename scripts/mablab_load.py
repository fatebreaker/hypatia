"""mablab_load.py (10-19) -- write the synthetic lab records (generated/mablab/synthetic_labs.json, scripts/mablab_synth.py) into a
RUNNING MedAgentBench server, for lab-task TRAINING only. The agents' proxy is GET-only, so the records go to HAPI on its internal
port, read from the proxy's command line ($MAB_STATE_DIR/proxy.pid). The server's overlay is discarded at stop, so every training
start must load again (the training items do, after scripts/medagentbench_server.sh start and after every watchdog restart).
Never run this against the server a TEST evaluation uses.
    MAB_STATE_DIR=... FHIR_API_BASE=... venv/bin/python scripts/mablab_load.py [records.json]
"""
import json
import os
import sys
import urllib.request

DEFAULT = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated/mablab/synthetic_labs.json"


def hapi_base(state_dir: str) -> str:
    pid = open(f"{state_dir}/proxy.pid").read().strip()
    cmd = open(f"/proc/{pid}/cmdline").read().split("\0")
    return cmd[cmd.index("--upstream") + 1].rstrip("/") + "/fhir/"


def load(records: list, state_dir: str, proxy_base: str) -> int:
    hapi = hapi_base(state_dir)
    n = 0
    for r in records:
        req = urllib.request.Request(hapi + "Observation", data=json.dumps(r).encode(),
                                     headers={"Content-Type": "application/fhir+json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"POST failed: {resp.status}")
        n += 1
    # check through the agents' proxy: every patient's record count is visible
    per = {}
    for r in records:
        m = r["subject"]["reference"].split("/")[-1]
        per[(m, r["code"]["text"])] = per.get((m, r["code"]["text"]), 0) + 1
    for (m, code), k in list(per.items())[:: max(1, len(per) // 25)]:
        with urllib.request.urlopen(f"{proxy_base}Observation?patient={m}&code={code}&_count=5000&_format=json", timeout=120) as resp:
            got = len(json.loads(resp.read()).get("entry", []))
        if got < k:
            raise RuntimeError(f"{m} {code}: {got} records visible, {k} loaded")
    return n


if __name__ == "__main__":
    recs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
    n = load(recs, os.environ["MAB_STATE_DIR"], os.environ["FHIR_API_BASE"])
    print(f"mablab_load: {n} synthetic lab records loaded and checked through the proxy")
