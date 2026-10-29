#!/usr/bin/env python
"""Compare Open Targets schemas across the four layout eras.

The backtest needs one normalized view spanning 21.06 -> 26.06, but the
release layout changed four times. This reports, per era, the schema of the
datasets that matter most:

  label source  : knownDrugsAggregated (21-24) / known_drug (25) / clinical_* (26)
  entities      : targets|target, diseases|disease
  evidence      : evidence/sourceId=* (21-25) / evidence_<source>/ (26)

Run:  venv/bin/python scripts/inspect_schemas.py
"""
import glob
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

BASE = str(data_root() / "opentargets")

# dataset role -> per-release directory name
ROLES = {
    "labels": {
        "21.06": "knownDrugsAggregated", "22.06": "knownDrugsAggregated",
        "23.06": "knownDrugsAggregated", "24.06": "knownDrugsAggregated",
        "25.06": "known_drug",           "26.06": "clinical_indication",
    },
    "target": {
        "21.06": "targets", "22.06": "targets", "23.06": "targets",
        "24.06": "targets", "25.06": "target",  "26.06": "target",
    },
    "disease": {
        "21.06": "diseases", "22.06": "diseases", "23.06": "diseases",
        "24.06": "diseases", "25.06": "disease",  "26.06": "disease",
    },
    "assoc_by_source": {
        "21.06": "associationByDatasourceDirect",
        "22.06": "associationByDatasourceDirect",
        "23.06": "associationByDatasourceDirect",
        "24.06": "associationByDatasourceDirect",
        "25.06": "association_by_datasource_direct",
        "26.06": "association_by_datasource_direct",
    },
}

RELEASES = ["21.06", "22.06", "23.06", "24.06", "25.06", "26.06"]


def first_parquet(d):
    """Find one parquet under d, descending into Hive partitions if needed."""
    if not os.path.isdir(d):
        return None
    hits = sorted(glob.glob(os.path.join(d, "*.parquet")))
    if hits:
        return hits[0]
    hits = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    return hits[0] if hits else None


def schema_of(path):
    try:
        return {n: str(t) for n, t in zip(pq.ParquetFile(path).schema_arrow.names,
                                          pq.ParquetFile(path).schema_arrow.types)}
    except Exception as e:                                    # noqa: BLE001
        return {"__error__": str(e)[:80]}


def main():
    for role, mapping in ROLES.items():
        print("=" * 78)
        print(f"ROLE: {role}")
        print("=" * 78)
        schemas = {}
        for rel in RELEASES:
            d = os.path.join(BASE, rel, mapping[rel])
            p = first_parquet(d)
            if p is None:
                print(f"  {rel:<7} {mapping[rel]:<34} MISSING")
                continue
            s = schema_of(p)
            schemas[rel] = s
            print(f"  {rel:<7} {mapping[rel]:<34} {len(s)} cols")

        if len(schemas) < 2:
            print()
            continue

        # Which columns are stable across every release we have?
        common = set.intersection(*(set(s) for s in schemas.values()))
        print(f"\n  STABLE across {len(schemas)} releases ({len(common)}):")
        print("    " + ", ".join(sorted(common)) or "    (none)")

        for rel, s in schemas.items():
            uniq = set(s) - common
            if uniq:
                print(f"  ONLY in {rel} ({len(uniq)}): {', '.join(sorted(uniq))}")
        print()

    # Evidence partitions available per release -- the provenance axis.
    print("=" * 78)
    print("EVIDENCE DATASOURCES PER RELEASE")
    print("=" * 78)
    for rel in RELEASES:
        if rel == "26.06":
            srcs = sorted(os.path.basename(p).replace("evidence_", "")
                          for p in glob.glob(os.path.join(BASE, rel, "evidence_*")))
        else:
            srcs = sorted(os.path.basename(p).replace("sourceId=", "")
                          for p in glob.glob(os.path.join(BASE, rel, "evidence", "sourceId=*")))
        print(f"  {rel:<7} {len(srcs):>2} sources: {', '.join(srcs) if srcs else '(none yet)'}")


if __name__ == "__main__":
    sys.exit(main())
