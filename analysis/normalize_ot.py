"""Normalise six Open Targets releases into one queryable view.

The backtest asks: given evidence frozen at time T, would an agent have picked
the targets that later succeeded? That needs a single table spanning releases,
but the release layout changed four times between 2021 and 2026.

What actually varies, after inspecting all six on disk:

  evidence   core keys are STABLE -- targetId, diseaseId, datasourceId,
             datatypeId, score exist in every release (column count grows
             66 -> 88, but we only need those five). Location differs:
               21.06-25.06  <rel>/evidence/sourceId=<src>/
               26.06+       <rel>/evidence_<src>/
  labels     knownDrugsAggregated (21-24) and known_drug (25) are IDENTICAL
             17-column schemas. 26.06 replaced them with clinical_target,
             which drops `phase` for `maxClinicalStage` (a string enum) and
             nests diseases in a list.
  entities   targets/diseases -> target/disease from 25.03.

Two things this module deliberately does NOT do:

  * It does not use `clinical_report.year` -- that field is 99.8% null
    (561 of 289,955). Use trialStartDate.
  * It does not treat "trial stopped" as failure. Only Negative (efficacy) and
    Safety_Sideeffects are target-quality signals; Insufficient_Enrollment and
    Business_Administrative dominate by count and are noise.

Usage:
    from analysis.normalize_ot import OpenTargets
    ot = OpenTargets()
    ot.build()                       # register views (lazy, no copying)
    ot.con.execute("SELECT * FROM evidence LIMIT 5").df()
    ot.frozen_evidence("22.06")      # evidence as of a release
    ot.first_seen()                  # earliest release per (target,disease,source)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import data_root  # noqa: E402

# Release -> approximate data cut. Open Targets tags releases YY.MM, and the
# data reflects sources ingested shortly before. Good enough to order a split;
# do not treat as an exact publication cutoff.
RELEASES: dict[str, str] = {
    "21.06": "2021-06-01",
    "22.06": "2022-06-01",
    "23.06": "2023-06-01",
    "24.06": "2024-06-01",
    "25.06": "2025-06-01",
    "26.06": "2026-06-01",
}

# 26.06 clinical_target.maxClinicalStage -> the integer `phase` used by every
# earlier release. Combined stages map to the highest reached.
STAGE_TO_PHASE = {
    "PRECLINICAL": 0, "IND": 0, "EARLY_PHASE_1": 0,
    "PHASE_1": 1, "PHASE_1_2": 2, "PHASE_2": 2,
    "PHASE_2_3": 3, "PHASE_3": 3,
    "PREAPPROVAL": 4, "APPROVAL": 4,
}

# Failure categories that say something about the TARGET, as opposed to
# enrollment, funding or business decisions.
INFORMATIVE_STOP = ("Negative", "Safety_Sideeffects")


@dataclass
class OpenTargets:
    root: Path = field(default=None)  # type: ignore[assignment]
    releases: tuple[str, ...] = tuple(RELEASES)
    con: duckdb.DuckDBPyConnection = field(default=None)  # type: ignore[assignment]
    memory_limit: str = "48GB"
    threads: int = 8
    temp_dir: str = "/tmp/duckdb_spill"

    def __post_init__(self) -> None:
        if self.root is None:
            self.root = data_root() / "opentargets"
        if self.con is None:
            self.con = duckdb.connect()
        self.con.execute("PRAGMA enable_progress_bar=false")
        # Explicit bounds: the un-normalised scan OOM-killed a login node.
        self.con.execute(f"SET memory_limit='{self.memory_limit}'")
        self.con.execute(f"SET threads={self.threads}")
        self.con.execute(f"SET temp_directory='{self.temp_dir}'")

    # ------------------------------------------------------------------ paths
    def _dir(self, rel: str, *names: str) -> Path | None:
        """First existing directory among candidate names for a release."""
        for n in names:
            p = self.root / rel / n
            if p.is_dir() and any(p.rglob("*.parquet")):
                return p
        return None

    def _evidence_glob(self, rel: str) -> str | None:
        """Evidence parquet glob, handling both partition layouts."""
        part = self.root / rel / "evidence"
        if part.is_dir() and any(part.glob("sourceId=*")):
            return f"{part}/sourceId=*/*.parquet"
        split = sorted(self.root.glob(f"{rel}/evidence_*"))
        if split:
            return f"{self.root / rel}/evidence_*/*.parquet"
        return None

    def available(self) -> list[str]:
        return [r for r in self.releases if self._evidence_glob(r)]

    # -------------------------------------------------------------- materialise
    def materialize(self, out: Path | None = None, verbose: bool = True) -> Path:
        """One-time ETL to compact parquet. Run this before build().

        A view over the raw releases is not viable: reconciling 66-89 column
        schemas across thousands of files with union_by_name forces DuckDB to
        read every schema and buffer wide rows, which OOMs even for a simple
        count. Since only five columns are ever needed, project them once per
        release into a narrow file. Downstream queries then hit ~7 columns
        instead of ~90, which is what makes repeated backtest queries cheap.
        """
        out = Path(out) if out else (self.root.parent / "normalized")
        out.mkdir(parents=True, exist_ok=True)
        for r in self.available():
            dest = out / f"evidence_{r}.parquet"
            if dest.exists():
                if verbose:
                    print(f"  {r}: cached")
                continue
            g = self._evidence_glob(r)
            if verbose:
                print(f"  {r}: projecting ...", flush=True)
            # Per-release so a failure costs one release, not the whole ETL.
            self.con.execute(f"""
                COPY (
                    SELECT '{r}' AS release, DATE '{RELEASES[r]}' AS release_date,
                           targetId, diseaseId, datasourceId, datatypeId,
                           CAST(score AS DOUBLE) AS score
                    FROM read_parquet('{g}', union_by_name=true)
                    WHERE targetId IS NOT NULL AND diseaseId IS NOT NULL
                ) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
            if verbose:
                n = self.con.execute(
                    f"SELECT count(*) FROM read_parquet('{dest}')").fetchone()[0]
                print(f"  {r}: {n:,} rows -> {dest.name} "
                      f"({dest.stat().st_size/1e6:.0f} MB)", flush=True)
        return out

    # ------------------------------------------------------------------ build
    def build(self, normalized: Path | None = None, verbose: bool = True) -> "OpenTargets":
        rels = self.available()
        if not rels:
            raise FileNotFoundError(f"no releases with evidence under {self.root}")

        norm = Path(normalized) if normalized else (self.root.parent / "normalized")
        files = sorted(norm.glob("evidence_*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"no normalized evidence in {norm} -- run materialize() first")
        self.con.execute(
            f"CREATE OR REPLACE VIEW evidence AS "
            f"SELECT * FROM read_parquet('{norm}/evidence_*.parquet')")

        # --- labels: knownDrugsAggregated / known_drug / clinical_target ---
        lparts = []
        for r in rels:
            d = self._dir(r, "knownDrugsAggregated", "known_drug")
            if d is not None:
                lparts.append(f"""
                    SELECT '{r}' AS release, DATE '{RELEASES[r]}' AS release_date,
                           targetId, diseaseId, drugId,
                           CAST(phase AS DOUBLE) AS phase
                    FROM read_parquet('{d}/*.parquet', union_by_name=true)
                    WHERE targetId IS NOT NULL AND diseaseId IS NOT NULL""")
                continue
            ct = self._dir(r, "clinical_target")
            if ct is not None:
                # 26.06 nests diseases as a LIST of STRUCTs
                # (diseaseFromSource, diseaseId) -- not a list of ids, which
                # silently yields struct-typed diseaseId and breaks joins
                # against every earlier release. Project the field.
                cases = " ".join(
                    f"WHEN '{k}' THEN {v}" for k, v in STAGE_TO_PHASE.items())
                lparts.append(f"""
                    SELECT '{r}' AS release, DATE '{RELEASES[r]}' AS release_date,
                           targetId, unnest(diseases).diseaseId AS diseaseId, drugId,
                           CAST(CASE maxClinicalStage {cases} ELSE NULL END AS DOUBLE) AS phase
                    FROM read_parquet('{ct}/*.parquet', union_by_name=true)
                    WHERE targetId IS NOT NULL""")
        if lparts:
            self.con.execute("CREATE OR REPLACE VIEW labels AS " + "\nUNION ALL\n".join(lparts))

        # --- trial outcomes (26.06 only): the failure-reason ground truth ---
        cr = self._dir("26.06", "clinical_report")
        if cr is not None:
            self.con.execute(f"""
                CREATE OR REPLACE VIEW trials AS
                SELECT id, trialPhase, trialOverallStatus, trialWhyStopped,
                       trialStopReasonCategories, trialStartDate,
                       CAST(substr(CAST(trialStartDate AS VARCHAR),1,4) AS INTEGER) AS start_year
                FROM read_parquet('{cr}/*.parquet', union_by_name=true)""")

        if verbose:
            n = self.con.execute("SELECT count(*) FROM evidence").fetchone()[0]
            print(f"releases : {', '.join(rels)}")
            print(f"evidence : {n:,} rows")
            if lparts:
                m = self.con.execute("SELECT count(*) FROM labels").fetchone()[0]
                print(f"labels   : {m:,} rows")
            if cr is not None:
                t = self.con.execute("SELECT count(*) FROM trials").fetchone()[0]
                print(f"trials   : {t:,} rows")
        return self

    # ------------------------------------------------------------- backtesting
    def frozen_evidence(self, release: str):
        """Evidence as it stood at `release` -- the agent's information set."""
        return self.con.execute(
            "SELECT * FROM evidence WHERE release = ?", [release]
        )

    def first_seen(self):
        """Earliest release in which each (target, disease, source) appears.

        This is what makes a leak-free split possible: an association whose
        first_seen is later than T did not exist at T and must be invisible.
        """
        self.con.execute("""
            CREATE OR REPLACE VIEW first_seen AS
            SELECT targetId, diseaseId, datasourceId,
                   min(release)      AS first_release,
                   min(release_date) AS first_date,
                   count(DISTINCT release) AS n_releases
            FROM evidence GROUP BY 1,2,3""")
        return self.con.execute("SELECT * FROM first_seen")

    def outcome_labels(self):
        """Target-level outcomes with only scientifically informative failures."""
        cats = ", ".join(f"'{c}'" for c in INFORMATIVE_STOP)
        self.con.execute(f"""
            CREATE OR REPLACE VIEW outcomes AS
            SELECT id, trialPhase, trialStartDate, start_year,
                   list_contains(trialStopReasonCategories, 'Negative')            AS failed_efficacy,
                   list_contains(trialStopReasonCategories, 'Safety_Sideeffects')  AS failed_safety
            FROM trials
            WHERE trialStopReasonCategories IS NOT NULL
              AND len(list_filter(trialStopReasonCategories,
                                  x -> x IN ({cats}))) > 0""")
        return self.con.execute("SELECT * FROM outcomes")


if __name__ == "__main__":
    import os
    ot = OpenTargets(memory_limit=os.environ.get("DUCKDB_MEM", "48GB"),
                     threads=int(os.environ.get("DUCKDB_THREADS", "8")))
    print("materialising normalized evidence ...")
    ot.materialize()
    ot.build()
    print("\n=== evidence rows per release ===")
    print(ot.con.execute(
        "SELECT release, count(*) n, count(DISTINCT targetId) targets, "
        "count(DISTINCT datasourceId) sources FROM evidence "
        "GROUP BY 1 ORDER BY 1").df().to_string(index=False))
