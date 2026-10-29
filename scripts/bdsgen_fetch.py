"""bdsgen_fetch.py (10-17, H-GEN27 clean training data) -- choose public cBioPortal studies that BioDSBench does NOT use and fetch
their clinical and mutation tables, for BioDSBench-style TRAINING tasks (scripts/bdsgen_build.py).

Selection: public studies (not NCI-TARGET, whose terms exclude methods/tool development), none of BioDSBench's 14 named studies
and none whose PubMed ID is one of BioDSBench's 38 (python and R tasks; scratchpad/bdsgen_meta/biodsbench_exclusions.json),
40-1,500 sequenced samples, and patient-level OS_MONTHS + OS_STATUS (checked through the API). Up to N_MAX studies, smallest
tarballs first is not knowable, so we prefer non-TCGA studies (smaller archives) and cap each download at 400 MB.
Download: https://datahub.assets.cbioportal.org/<study>.tar.gz; only data_clinical_patient.txt, data_clinical_sample.txt and
data_mutations*.txt are kept (generated/bdsgen_raw/<study>/). Run on a node with internet (the CPU agent):
    python3 scripts/bdsgen_fetch.py
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

S = "/scratch/login/claude-2899/-project-pi-hongyu-umass-edu-hieutran-deep-deeprs/bed53d0c-1958-433d-b7ec-a923d9217fff/scratchpad"
OUT = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated/bdsgen_raw"
API = "https://www.cbioportal.org/api"
N_MAX = 60
EXCL = json.load(open(f"{S}/bdsgen_meta/biodsbench_exclusions.json"))
studies = json.load(open(f"{S}/cbio_studies.json"))


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


cands = []
for st in studies:
    sid, pm = st["studyId"], str(st.get("pmid") or "")
    if not st.get("publicStudy") or "TARGET" in (st.get("groups") or "") or sid in EXCL["studies"]:
        continue
    if pm and any(p in EXCL["pmids"] for p in pm.split(",")):
        continue
    n = st.get("sequencedSampleCount") or 0
    if not 40 <= n <= 1500:
        continue
    cands.append(st)
cands.sort(key=lambda x: ("tcga" in x["studyId"], x["sequencedSampleCount"]))
print(f"{len(cands)} candidate studies after exclusions", flush=True)
os.makedirs(OUT, exist_ok=True)
kept = []
for st in cands:
    if len(kept) >= N_MAX:
        break
    sid = st["studyId"]
    try:
        attrs = {a["clinicalAttributeId"] for a in get(f"{API}/studies/{sid}/clinical-attributes")}
    except Exception as e:  # noqa: BLE001
        print(f"  {sid}: attribute query failed ({e})", flush=True)
        continue
    if not {"OS_MONTHS", "OS_STATUS"} <= attrs:
        continue
    d = f"{OUT}/{sid}"
    if os.path.isdir(d) and os.path.exists(f"{d}/data_clinical_patient.txt"):
        kept.append(sid)
        continue
    tgz = f"{OUT}/{sid}.tar.gz"
    rc = subprocess.run(["curl", "-s", "-L", "-m", "900", "--max-filesize", str(400 * 2 ** 20), "-o", tgz,
                         f"https://datahub.assets.cbioportal.org/{sid}.tar.gz"]).returncode
    if rc != 0 or not os.path.exists(tgz):
        print(f"  {sid}: download failed (curl rc {rc})", flush=True)
        if os.path.exists(tgz):
            os.remove(tgz)
        continue
    os.makedirs(d, exist_ok=True)
    try:
        with tarfile.open(tgz) as tf:
            for m in tf.getmembers():
                base = os.path.basename(m.name)
                if m.isfile() and (base in ("data_clinical_patient.txt", "data_clinical_sample.txt")
                                   or (base.startswith("data_mutations") and base.endswith(".txt"))):
                    src = tf.extractfile(m)
                    with open(f"{d}/{base}", "wb") as fh:
                        shutil.copyfileobj(src, fh)
    except Exception as e:  # noqa: BLE001
        print(f"  {sid}: extraction failed ({e})", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    finally:
        os.remove(tgz)
    if os.path.exists(f"{d}/data_clinical_patient.txt") and any(f.startswith("data_mutations") for f in os.listdir(d)):
        kept.append(sid)
        json.dump({"studyId": sid, "name": st["name"], "pmid": st.get("pmid"), "citation": st.get("citation"),
                   "sequencedSampleCount": st["sequencedSampleCount"], "cancerType": st["cancerType"]["name"]},
                  open(f"{d}/STUDY.json", "w"), indent=1)
        print(f"  kept {sid} ({st['sequencedSampleCount']} sequenced samples) [{len(kept)}/{N_MAX}]", flush=True)
    else:
        shutil.rmtree(d, ignore_errors=True)
json.dump({"kept": kept, "excluded_pmids": EXCL["pmids"], "excluded_studies": EXCL["studies"]},
          open(f"{OUT}/SELECTION.json", "w"), indent=1)
print(f"{len(kept)} studies kept in {OUT}")
