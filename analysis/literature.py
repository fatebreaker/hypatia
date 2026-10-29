"""Augment the backtest dossiers with the literature the agent is supposed to read.

WHY THIS EXISTS
---------------
`analysis/decision_set.py` renders the classifier's own feature vector as text:
per-datasource association scores, target annotations, clinical stage. So the
arm-A result (RESULTS §12) measures an LLM reasoning over a *tabulation* — and
the question on the front page of the README, whether an agent that **reads
scientific evidence** beats one that merely **tabulates** it, has never been
tested. This module is what tests it.

The existing dossiers become the control condition: same 1,000 targets, same
classifiers, same paired bootstrap, one variable changed. That two-point
ablation is a much stronger design than starting here would have been.

THE LEAK, AND WHY THE GUARD IS THE POINT OF THIS FILE
-----------------------------------------------------
A dossier is supposed to contain only what was knowable at the cutoff. Evidence
rows in the cutoff release can still reference papers published AFTER it, and
`publicationYear` is dirty in ways that defeat the obvious filter. Measured on
OT 24.06 `evidence_europepmc`:

    2022  1,308,927
    2023  1,190,046
    2024    397,606      partial year -- the cutoff is JUNE 2024
    2025          4      impossible in a 24.06 release
    2119/2120/2121/2122  63 rows of garbage

So `publicationYear <= 2024` leaks both the second half of 2024 and every
garbage row. We keep **strictly earlier than the cutoff year**, which discards
all of 2024 rather than guess months from a year-granular field. That is
deliberately conservative: losing six months of real papers is cheap, and a
leak here would look like a spectacular agent win — the A6-class failure this
project has already been burned by twice.

`verify()` asserts the maximum publication year across every emitted snippet is
below the cutoff year. If that assertion ever fires, every number produced
after it is wrong.

WHAT THE AGENT GETS
-------------------
Top-N text-mined sentences per target by `resourceScore`, each tagged with its
section (abstract / results / discuss) and year. These are the sentences Open
Targets itself mined to justify the `europepmc` association — i.e. the evidence
*behind* the score the control dossier reports as a bare number.

Output is a drop-in replacement dossier set, so it runs unchanged:

    venv/bin/python analysis/literature.py --out-root <dir>
    venv/bin/python analysis/agent_backtest.py --agent llm:... --root <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

# Anything at or above the cutoff year is dropped. Year granularity cannot
# resolve a mid-year cutoff, and the safe direction is obvious.
MAX_SENTENCE_CHARS = 400
PER_PAPER = 2          # sentences kept per paper, so one study cannot dominate


def cutoff_year(cutoff: str) -> int:
    """'24.06' -> 2024. Release tags are YY.MM."""
    return 2000 + int(cutoff.split(".")[0])


def fetch(cutoff: str, targets: list[str], top_n: int) -> dict[str, list[dict]]:
    """Top-N pre-cutoff mined sentences per target, best score first."""
    import duckdb

    year = cutoff_year(cutoff)
    ev = data_root() / "opentargets" / cutoff / "evidence" / "sourceId=europepmc" / "*.parquet"
    if not list((data_root() / "opentargets" / cutoff / "evidence" /
                 "sourceId=europepmc").glob("*.parquet")):
        raise FileNotFoundError(f"no europepmc evidence for {cutoff}")

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE t(targetId VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?)", [(x,) for x in targets])
    # Two-stage ranking. Ranking sentences by score alone lets ONE high-scoring
    # paper fill the entire quota -- observed: 6 of 6 snippets from a single
    # 2018 study -- so the agent reads one paper rather than a body of
    # evidence, which is the opposite of what this experiment is testing.
    # Cap each paper at PER_PAPER sentences first, then take the best across
    # papers.
    rows = con.execute(f"""
        WITH s AS (
            SELECT e.targetId, e.publicationYear AS yr, e.resourceScore AS score,
                   e.diseaseId, u.s.section AS section, u.s.text AS text,
                   coalesce(e.literature[1], '?') AS paper
            FROM read_parquet('{ev}') e
            JOIN t ON t.targetId = e.targetId,
                 UNNEST(e.textMiningSentences) AS u(s)
            WHERE e.publicationYear IS NOT NULL
              AND e.publicationYear < {year}      -- STRICTLY before: see docstring
              AND e.publicationYear > 1800        -- drop garbage low end too
              AND u.s.text IS NOT NULL AND length(u.s.text) > 40
        ), capped AS (
            SELECT * FROM s
            QUALIFY row_number() OVER (
                PARTITION BY targetId, paper
                -- abstracts and conclusions carry the claim; methods rarely do
                ORDER BY CASE section WHEN 'abstract' THEN 0 WHEN 'concl' THEN 1
                                      WHEN 'discuss' THEN 2 ELSE 3 END
            ) <= {PER_PAPER}
        )
        SELECT targetId, yr, score, diseaseId, section, text FROM capped
        QUALIFY row_number() OVER (
            PARTITION BY targetId ORDER BY score DESC, yr DESC
        ) <= {top_n}
    """).fetchall()

    out: dict[str, list[dict]] = {}
    for tid, yr, score, dis, section, text in rows:
        out.setdefault(tid, []).append(
            {"year": int(yr), "score": float(score or 0.0), "disease": dis,
             "section": section or "?", "text": " ".join(str(text).split())[:MAX_SENTENCE_CHARS]})
    return out


def render(snips: list[dict]) -> str:
    lines = ["", "LITERATURE (text-mined sentences behind the europepmc score,",
             "            published BEFORE the cutoff):"]
    for s in snips:
        lines.append(f"  [{s['year']}, {s['section']}] {s['text']}")
    return "\n".join(lines)


def verify(path: Path, year: int) -> int:
    """The assertion this module exists for: nothing from on or after the cutoff."""
    rows = [json.loads(l) for l in open(path)]
    years = [s["year"] for r in rows for s in r.get("_literature", [])]
    withlit = sum(1 for r in rows if r.get("_literature"))
    print(f"dossiers: {len(rows)}   with literature: {withlit}   "
          f"snippets: {len(years)}")
    if not years:
        print("verify: FAIL -- no literature attached to any dossier")
        return 1
    print(f"publication years: min {min(years)}  max {max(years)}  "
          f"(cutoff year {year})")
    if max(years) >= year:
        print(f"verify: FAIL -- LEAK: a snippet from {max(years)} is at or after "
              f"the cutoff year {year}. Every downstream number is invalid.")
        return 1
    print(f"verify: PASS -- max year {max(years)} < {year}, no future leakage")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="24.06")
    ap.add_argument("--final", default="26.06")
    ap.add_argument("--top-n", type=int, default=12, help="sentences per target")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    src = data_root() / "backtest"
    dst = Path(args.out_root) if args.out_root else data_root() / "backtest_lit"
    name = f"dossiers_{args.cutoff}_{args.final}.jsonl"
    year = cutoff_year(args.cutoff)

    if args.verify_only:
        return verify(dst / name, year)

    dst.mkdir(parents=True, exist_ok=True)
    ds_name = f"decision_set_{args.cutoff}_{args.final}.parquet"
    shutil.copy(src / ds_name, dst / ds_name)      # same targets, same labels

    dos = [json.loads(l) for l in open(src / name)]
    targets = [d["targetId"] for d in dos]
    print(f"{len(targets)} targets; pulling <= {args.top_n} pre-{year} sentences each")
    lit = fetch(args.cutoff, targets, args.top_n)
    print(f"literature found for {len(lit)}/{len(targets)} targets")

    with open(dst / name, "w") as fh:
        for d in dos:
            snips = lit.get(d["targetId"], [])
            if snips:
                d["text"] = d["text"] + "\n" + render(snips)
            d["_literature"] = snips          # scoring/verification only
            fh.write(json.dumps(d) + "\n")
    print(f"wrote {dst / name}")

    added = [len(lit.get(t, [])) for t in targets]
    chars = [len(d["text"]) for d in dos]
    print(f"snippets per target: mean {sum(added)/len(added):.1f}, max {max(added)}")
    print(f"dossier chars: mean {sum(chars)//len(chars):,}, max {max(chars):,}")
    print()
    return verify(dst / name, year)


if __name__ == "__main__":
    raise SystemExit(main())
