"""mablab_synth.py (10-19, user: 'fill both data gaps') -- SYNTHETIC LAB RESULTS for MedAgentBench-style lab-task TRAINING.

Why: MedAgentBench's sandbox keeps its lab data (magnesium MG, capillary glucose GLU, potassium K, HbA1c A1C) almost entirely on the
98 patients its test tasks name; of the 597 others, 1-2 have a recent value (scripts/mabgen_build2.py's survey), so the lab task
types (task4/5/6/7/9/10, 60% of the test set) had almost no clean training data. This writes realistic lab histories for 300
patients that NO test task and NO earlier training task (generated/mabgen2) names: 50 per lab task type, each patient planned for
one outcome (measured or not; normal or low -> an order; recent or stale HbA1c), with older readings outside every time window
as distractors. The records copy the sandbox's own Observation format (code coding loinc.org MG/GLU/K/A1C, units mg/dL / mg/dL /
mmol/L / %, laboratory category, subject reference + identifier, effectiveDateTime / issued with +00:00).

They are loaded only into a PRIVATE copy of the server used for TRAINING (scripts/mablab_load.py, after medagentbench_server.sh
start); the test evaluation keeps the pristine server, and no test patient receives any record. Deterministic (seed 20261019).
Output: generated/mablab/synthetic_labs.json (records) + plan.json (type, MRN, planned outcome).
    venv/bin/python scripts/mablab_synth.py
"""
import datetime as dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path

S = "/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad"
G = Path("/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated")
OUT = G / "mablab"
SEED, N_PER = 20261019, 50
NOW = dt.datetime(2023, 11, 13, 10, 15, tzinfo=dt.timezone.utc)      # the benchmark's 'now' (refsol cutoff)
UNIT = {"MG": "mg/dL", "GLU": "mg/dL", "K": "mmol/L", "A1C": "%"}
# planned outcomes per lab task type (counts sum to N_PER); both acting and not acting occur in every type with an action
PLAN = {
    "task4": {"value": 25, "none_old": 15, "none_absent": 10},                       # Mg within 24 h? value : -1
    "task5": {"low": 20, "normal": 15, "none_old": 10, "none_absent": 5},            # Mg within 24 h low -> IV order
    "task6": {"value": 42, "none_old": 8},                                            # mean CBG over 24 h
    "task7": {"value": 45, "none_absent": 5},                                         # most recent CBG
    "task9": {"low": 25, "normal": 25},                                               # most recent K low -> order + lab
    "task10": {"recent": 20, "old": 15, "none_absent": 15},                          # A1C older than 1 y -> order
}
CODE = {"task4": "MG", "task5": "MG", "task6": "GLU", "task7": "GLU", "task9": "K", "task10": "A1C"}


def iso(t):
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def obs(mrn, code, t, value, rng):
    issued = t + dt.timedelta(minutes=rng.randint(5, 90))
    return {"resourceType": "Observation", "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory",
                                      "display": "Laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": code}], "text": code},
            "subject": {"reference": f"Patient/{mrn}", "identifier": {"system": "http://terminology.hl7.org/CodeSystem/v2-0203", "value": mrn}},
            "effectiveDateTime": iso(t), "issued": iso(min(issued, NOW)),
            "valueQuantity": {"value": value, "unit": UNIT[code], "system": "http://unitsofmeasure.org", "code": UNIT[code]}}


def within(rng, hours_lo, hours_hi):
    return NOW - dt.timedelta(minutes=rng.randint(int(hours_lo * 60), int(hours_hi * 60)))


def older(rng, n, min_h=26, max_days=120):
    return [NOW - dt.timedelta(minutes=rng.randint(min_h * 60, max_days * 24 * 60)) for _ in range(n)]


def make(t, outcome, mrn, rng):
    code, recs = CODE[t], []
    if outcome == "none_absent":
        return recs
    if t in ("task4", "task5"):
        for when in older(rng, rng.randint(1, 4)):                       # stale readings (distractors / the 'none_old' case)
            recs.append(obs(mrn, code, when, round(rng.uniform(1.4, 2.6), 1), rng))
        if outcome in ("value", "normal", "low"):
            n_in = rng.randint(1, 3)
            times = sorted(within(rng, 0.5, 23.5) for _ in range(n_in))
            for i, when in enumerate(times):
                last = i == len(times) - 1
                if outcome == "low" and last:
                    v = rng.choice([round(rng.uniform(0.7, 0.99), 2), round(rng.uniform(1.0, 1.49), 1), round(rng.uniform(1.5, 1.8), 1)])
                elif outcome == "normal" and last:
                    v = round(rng.uniform(2.0, 2.6), 1)
                else:
                    v = round(rng.uniform(1.4, 2.6), 1)
                recs.append(obs(mrn, code, when, v, rng))
    elif t == "task6":
        for when in older(rng, rng.randint(1, 5)):
            recs.append(obs(mrn, code, when, float(rng.randint(70, 320)), rng))
        if outcome == "value":
            for when in sorted(within(rng, 0.3, 23.7) for _ in range(rng.randint(2, 6))):
                recs.append(obs(mrn, code, when, float(rng.randint(70, 320)), rng))
    elif t == "task7":
        times = sorted((NOW - dt.timedelta(minutes=rng.randint(30, 60 * 24 * 60)) for _ in range(rng.randint(1, 6))))
        for when in times:
            recs.append(obs(mrn, code, when, float(rng.randint(70, 320)), rng))
    elif t == "task9":
        times = sorted((NOW - dt.timedelta(minutes=rng.randint(60, 60 * 24 * 90)) for _ in range(rng.randint(1, 5))))
        for i, when in enumerate(times):
            last = i == len(times) - 1
            if last:
                v = round(rng.uniform(2.8, 3.4), 1) if outcome == "low" else round(rng.uniform(3.5, 5.2), 1)
            else:
                v = round(rng.uniform(3.0, 5.2), 1)
            recs.append(obs(mrn, code, when, v, rng))
    elif t == "task10":
        if outcome == "recent":
            last_t = NOW - dt.timedelta(days=rng.randint(10, 330))
        else:                                                             # old: more than a year before 'now'
            last_t = NOW - dt.timedelta(days=rng.randint(380, 900))
        prev = [last_t - dt.timedelta(days=rng.randint(60, 500)) for _ in range(rng.randint(0, 3))]
        for when in prev + [last_t]:
            recs.append(obs(mrn, code, when, round(rng.uniform(4.8, 11.5), 1), rng))
    return recs


def main():
    if (OUT / "synthetic_labs.json").exists():
        sys.exit(f"{OUT}/synthetic_labs.json exists -- not overwriting")
    test = set(json.load(open(f"{S}/mabgen/test_mrns.json")))
    used = set()
    for d in sorted((G / "mabgen2" / "train").iterdir()):                # patients an earlier training task already names
        cfg = json.loads((d / "eval_config.json").read_text())
        case = next(iter(cfg["grader"]["config"]["keys"].values()))["case"]
        if case.get("eval_MRN"):
            used.add(case["eval_MRN"])
        used |= {s for s in case.get("sol", []) if isinstance(s, str) and s.startswith("S")}
    allp = json.load(open(f"{S}/mablab/all_patients.json"))             # every MRN on the server (written by the probe/build step)
    pool = sorted(m for m in allp if m not in test and m not in used)
    need = N_PER * len(PLAN)
    print(f"{len(allp)} patients on the server; {len(test)} test, {len(used)} used by mabgen2 -> {len(pool)} free; need {need}")
    assert len(pool) >= need, "not enough free patients"
    rng = random.Random(SEED)
    chosen = rng.sample(pool, need)
    recs, plan, k = [], [], 0
    for t in sorted(PLAN, key=lambda x: int(x[4:])):
        outs = [o for o, n in PLAN[t].items() for _ in range(n)]
        for o in outs:
            mrn = chosen[k]
            k += 1
            r = make(t, o, mrn, rng)
            recs += r
            plan.append({"type": t, "mrn": mrn, "planned": o, "n_records": len(r)})
    OUT.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(recs, sort_keys=True)
    (OUT / "synthetic_labs.json").write_text(blob)
    json.dump({"built": "2026-10-19", "seed": SEED, "n_records": len(recs), "sha1": hashlib.sha1(blob.encode()).hexdigest(),
               "plan": plan, "patients_excluded": {"test": len(test), "mabgen2": len(used)}}, open(OUT / "plan.json", "w"), indent=1)
    print(f"{len(recs)} synthetic lab records for {len(plan)} patients -> {OUT}")


if __name__ == "__main__":
    main()
