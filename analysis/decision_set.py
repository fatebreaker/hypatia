"""Build the decision set an agent is asked to rank, with evidence dossiers.

The test cohort is 30,370 targets with 215 positives (0.7%). Running an agent
over all of it is not affordable, and a plain random sample would contain ~3
positives -- useless. So we case-control sample: every positive plus a random
draw of negatives.

That inflates the base rate (0.7% -> ~21%), which is fine PROVIDED the
classifiers are scored on the identical set. Enrichment is always reported
against the set's own base rate, never the cohort's, so the agent and the
baselines are compared on exactly the same decision problem. Reporting an
agent's enrichment against the cohort base rate while it only saw the sample
would inflate it ~30x, and is the obvious way to accidentally cheat here.

Each dossier contains only what was visible at the cutoff release:
  * per-datasource evidence -- score, count, how many diseases it spans
  * target annotations from the same release (symbol, biotype, tractability,
    genetic constraint, known safety liabilities)
  * clinical state at T, which an analyst would also have known

Nothing derived from a later release ever enters a dossier. `phase_final`
lives on the record for scoring but is never rendered into agent-visible text.

Run:  venv/bin/python analysis/decision_set.py --n-neg 785
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.normalize_ot import OpenTargets  # noqa: E402
from analysis.split import CLINICAL_SOURCES, TemporalSplit  # noqa: E402
from paths import data_root  # noqa: E402

# Human-readable gloss per datasource, so the agent is judging evidence rather
# than decoding column names. Wording is deliberately neutral about strength.
SOURCE_DESC = {
    "ot_genetics_portal": "GWAS locus-to-gene (common-variant association)",
    "gene_burden": "rare-variant burden test",
    "eva": "ClinVar germline variant-disease assertions",
    "eva_somatic": "ClinVar somatic variant assertions",
    "genomics_england": "Genomics England PanelApp gene-disease panel",
    "gene2phenotype": "curated developmental-disorder gene-disease pairs",
    "clingen": "ClinGen expert-curated gene-disease validity",
    "orphanet": "Orphanet rare-disease gene associations",
    "uniprot_literature": "UniProt curated literature annotation",
    "uniprot_variants": "UniProt curated variant annotation",
    "impc": "mouse knockout phenotype (IMPC)",
    "expression_atlas": "differential expression in disease tissue",
    "europepmc": "text-mined co-occurrence in literature",
    "crispr": "CRISPR loss-of-function screen",
    "cancer_gene_census": "COSMIC Cancer Gene Census",
    "intogen": "IntOGen cancer driver analysis",
    "cancer_biomarkers": "cancer biomarker association",
    "slapenrich": "pathway-level somatic mutation enrichment",
    "reactome": "Reactome pathway involvement",
    "sysbio": "systems-biology gene signature",
    "progeny": "PROGENy pathway activity",
    "chembl": "EXISTING CLINICAL ACTIVITY: a drug against this target has "
              "reached the stated trial phase",
}


def target_annotations(ot: OpenTargets, release: str) -> pd.DataFrame:
    """Target-level annotations as of `release` (never a later one)."""
    d = ot._dir(release, "target", "targets")
    if d is None:
        return pd.DataFrame(columns=["targetId"])
    cols = list(ot.con.execute(
        f"SELECT * FROM read_parquet('{d}/*.parquet') LIMIT 0").df().columns)
    # Identifiers must be quoted: `constraint` is a DuckDB reserved word.
    sel = [f'"{c}"' for c in ("id", "approvedSymbol", "approvedName", "biotype")
           if c in cols]
    # constraint / safetyLiabilities / tractability are nested LISTs of STRUCTs.
    # Pulling them raw drags heavy objects into pandas for no benefit, so
    # reduce each to the scalar an analyst would actually act on.
    if "safetyLiabilities" in cols:
        sel.append('len(COALESCE("safetyLiabilities", [])) AS n_safety_liabilities')
    if "constraint" in cols:
        sel.append('list_aggregate(list_transform(COALESCE("constraint", []), '
                   'x -> x.upperBin6), \'min\') AS constraint_bin')
    if "tractability" in cols:
        sel.append('len(list_filter(COALESCE("tractability", []), '
                   'x -> x.value)) AS n_tractability_flags')
    q = ", ".join(sel)
    try:
        df = ot.con.execute(f"SELECT {q} FROM read_parquet('{d}/*.parquet')").df()
    except Exception as e:                                        # noqa: BLE001
        print(f"  WARN: target annotations reduced ({str(e)[:90]})")
        df = ot.con.execute(
            f'SELECT "id", "approvedSymbol", "approvedName", "biotype" '
            f"FROM read_parquet('{d}/*.parquet')").df()
    return df.rename(columns={"id": "targetId"})


def build(cutoff: str, final: str, n_neg: int, seed: int,
          out: Path) -> pd.DataFrame:
    ot = OpenTargets(memory_limit="16GB").build(verbose=False)
    ts = TemporalSplit(ot, final=final).build(cutoff)
    ts.check_leakage(verbose=False)
    tbl = ts.table().df()

    pos = tbl[tbl.advanced == 1]
    neg = tbl[tbl.advanced == 0].sample(
        n=min(n_neg, (tbl.advanced == 0).sum()), random_state=seed)
    ds = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    ann = target_annotations(ot, cutoff)
    ds = ds.merge(ann, on="targetId", how="left")

    print(f"decision set: {len(ds)} targets  "
          f"({int(ds.advanced.sum())} advanced = {ds.advanced.mean():.1%} base rate)")
    print(f"  cohort base rate was {tbl.advanced.mean():.3%} "
          f"-- score enrichment against the SET rate, not the cohort rate")

    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out / f"decision_set_{cutoff}_{final}.parquet", index=False)

    dossiers = [dossier(r) for _, r in ds.iterrows()]
    with open(out / f"dossiers_{cutoff}_{final}.jsonl", "w") as fh:
        for d in dossiers:
            fh.write(json.dumps(d) + "\n")
    print(f"wrote {out}/dossiers_{cutoff}_{final}.jsonl")
    return ds


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def dossier(row: pd.Series) -> dict:
    """Agent-visible record. Label fields are kept OUT of `text`."""
    srcs = []
    for c in row.index:
        if not c.startswith("src_"):
            continue
        v = row[c]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        name = c[4:]
        srcs.append({"datasource": name,
                     "description": SOURCE_DESC.get(name, name),
                     "max_score": float(v),
                     "is_clinical": name in CLINICAL_SOURCES})
    srcs.sort(key=lambda d: -d["max_score"])

    lines = [
        f"TARGET: {row.get('approvedSymbol') or row['targetId']} "
        f"({row['targetId']})",
        f"name: {_fmt(row.get('approvedName'))}",
        f"biotype: {_fmt(row.get('biotype'))}",
        f"clinical stage at cutoff: phase {_fmt(row.get('phase_t'))}",
        f"datasources with evidence: {int(row.get('n_sources', 0))}, "
        f"total evidence records: {int(row.get('total_evidence', 0))}, "
        f"distinct diseases: {int(row.get('n_diseases', 0))}",
        "",
        "EVIDENCE BY DATASOURCE (association score 0-1, as of the cutoff):",
    ]
    for s in srcs:
        flag = "  [clinical]" if s["is_clinical"] else ""
        lines.append(f"  {s['datasource']:<22} {s['max_score']:.3f}  "
                     f"{s['description']}{flag}")

    return {
        "targetId": row["targetId"],
        "symbol": row.get("approvedSymbol"),
        "phase_t": float(row.get("phase_t", 0)),
        "sources": srcs,
        "text": "\n".join(lines),
        # scoring only -- never shown to the agent
        "_label_advanced": int(row["advanced"]),
        "_label_late_stage": int(row.get("reached_late_stage", 0)),
        "_phase_final": float(row.get("phase_final", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="24.06")
    ap.add_argument("--final", default="26.06")
    ap.add_argument("--n-neg", type=int, default=785)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else data_root() / "backtest"
    ds = build(args.cutoff, args.final, args.n_neg, args.seed, out)
    print("\nsample dossier:\n" + "-" * 60)
    print(dossier(ds.iloc[0])["text"][:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
