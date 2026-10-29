#!/usr/bin/env python
"""Build the exact S3 key list for PMC articles cited by Open Targets evidence.

Why not just mirror PMC: the bucket is keyed per-article-VERSION
(PMC10000000.1/PMC10000000.1.txt), ~8M article versions, no bulk tarballs.
Open Targets cites 1.87M distinct PMCIDs -- that's the corpus the agent would
actually encounter, and it's ~47 GB as plain text instead of TBs.

Also: OT gives bare PMCIDs with no version suffix, so we cannot construct keys
by hand. The S3 inventory report is the only reliable PMCID -> key mapping.
When an article has several versions we keep the highest.

Steps:
  1. download the 5 inventory CSVs (243 MB) listed in manifest.json
  2. keep .txt keys only (plain text; skip pdf/xml/images)
  3. intersect with PMCIDs cited in OT evidence_europepmc
  4. emit keys.txt for the parallel fetcher

Run:  venv/bin/python scripts/pmc_build_keylist.py
"""
import csv
import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

BASE = str(data_root())
PMC = f"{BASE}/pmc"
INV = f"{PMC}/inventory"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"

# The bucket publishes exactly one inventory configuration, and it covers the
# metadata/ prefix only (metadata/PMC10000000.1.json) -- there is no
# full-object inventory. That is still sufficient: it enumerates every
# article-version, and the text object is at a deterministic path
#   PMC<id>.<ver>/PMC<id>.<ver>.txt   (verified: HTTP 206 on range request)
# so we recover the version from the metadata key and construct the txt key.
# Articles that are PDF-only will 404 at fetch time; curl -f drops those.
KEY_RE = re.compile(r"^metadata/(PMC\d+)\.(\d+)\.json$")


def sh(cmd):
    subprocess.run(cmd, shell=True, check=False)


# The inventory lives under `inventory-reports/`, NOT `inventory/`, and each
# run is published to its own timestamped prefix. Discovering the newest one is
# the only way to avoid pinning a date that silently goes stale.
INV_PREFIX = "inventory-reports/pmc-oa-opendata/metadata/"
SNAPSHOT_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}Z)/")


def latest_manifest_url() -> str:
    """URL of the most recent inventory manifest.

    Previously this function assumed `manifest.json` had already been placed in
    the pmc/ directory by hand -- an undocumented manual step, so the script
    died with FileNotFoundError before fetching anything.
    """
    import urllib.request

    url = (f"{S3}/?list-type=2&prefix={INV_PREFIX}&delimiter=/")
    with urllib.request.urlopen(url, timeout=120) as r:
        xml = r.read().decode()
    snaps = sorted(set(SNAPSHOT_RE.findall(xml)))
    if not snaps:
        raise RuntimeError(f"no inventory snapshots under {INV_PREFIX}")
    print(f"[1] newest inventory snapshot: {snaps[-1]}  ({len(snaps)} available)")
    return f"{S3}/{INV_PREFIX}{snaps[-1]}/manifest.json"


def step1_download_inventory():
    os.makedirs(INV, exist_ok=True)
    man_path = f"{PMC}/manifest.json"
    if not os.path.exists(man_path) or os.path.getsize(man_path) == 0:
        sh(f"wget -q -O {man_path} '{latest_manifest_url()}'")
    man = json.load(open(man_path))
    print(f"[1] {len(man['files'])} inventory CSVs")
    for f in man["files"]:
        dest = os.path.join(INV, os.path.basename(f["key"]))
        if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
            print(f"    have {os.path.basename(dest)}")
            continue
        print(f"    fetch {os.path.basename(dest)} ({f['size']/1e6:.0f} MB)")
        sh(f"wget -c -q -O {dest} '{S3}/{f['key']}'")


def step2_extract_txt_keys():
    """PMCID -> highest-version .txt key."""
    best = {}
    files = sorted(f for f in os.listdir(INV) if f.endswith(".csv.gz"))
    for i, fn in enumerate(files, 1):
        print(f"[2] parsing {fn} ({i}/{len(files)})")
        with gzip.open(os.path.join(INV, fn), "rt", newline="") as fh:
            for row in csv.reader(fh):
                # schema: Bucket, Key, LastModifiedDate, ETag
                if len(row) < 2:
                    continue
                m = KEY_RE.match(row[1])
                if not m:
                    continue
                pmcid, ver = m.group(1), int(m.group(2))
                cur = best.get(pmcid)
                if cur is None or ver > cur[0]:
                    best[pmcid] = (ver, f"{pmcid}.{ver}/{pmcid}.{ver}.txt")
    print(f"[2] {len(best):,} distinct PMCIDs with a .txt key")
    return best


def step3_open_targets_pmcids():
    import duckdb
    ot = f"{BASE}/opentargets/26.06/evidence_europepmc/*.parquet"
    print("[3] extracting cited PMCIDs from Open Targets")
    rows = duckdb.connect().execute(f"""
        SELECT DISTINCT unnest(pmcIds) AS p
        FROM read_parquet('{ot}') WHERE pmcIds IS NOT NULL
    """).fetchall()
    ids = {r[0] for r in rows if r[0]}
    ids = {i if i.startswith("PMC") else f"PMC{i}" for i in ids}
    print(f"[3] {len(ids):,} distinct PMCIDs cited")
    return ids


def main():
    step1_download_inventory()
    keymap = step2_extract_txt_keys()
    wanted = step3_open_targets_pmcids()

    hit = wanted & set(keymap)
    miss = wanted - set(keymap)
    print(f"\n[4] cited={len(wanted):,}  available_as_txt={len(hit):,}  "
          f"not_in_bucket={len(miss):,}  ({100*len(hit)/max(len(wanted),1):.1f}% coverage)")

    out = f"{PMC}/keys.txt"
    with open(out, "w") as fh:
        for pmcid in sorted(hit):
            fh.write(keymap[pmcid][1] + "\n")
    print(f"[4] wrote {out}")

    with open(f"{PMC}/missing_pmcids.txt", "w") as fh:
        fh.write("\n".join(sorted(miss)))
    print(f"[4] wrote {PMC}/missing_pmcids.txt "
          f"(closed-access or non-OA -- abstracts only for these)")


if __name__ == "__main__":
    sys.exit(main())
