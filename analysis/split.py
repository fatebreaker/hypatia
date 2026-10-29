"""Temporal split protocol for the target-identification backtest.

The question: given evidence frozen at release T, would you have picked the
target-indication pairs that subsequently advanced in the clinic?

    features   evidence as of T          (what an analyst could have known)
    label      phase change T -> 26.06   (what actually happened afterwards)

Design decisions that matter, and why:

1. LABEL IS ADVANCEMENT, NOT FINAL PHASE.
   A pair already at Phase 3 in 2021 that is approved by 2026 is not a
   prediction success -- it was already a known winner. The label is whether
   the pair moved UP from where it stood at T. This is the difference between
   predicting the future and reading the present.

2. PAIRS MUST EXIST AT T.
   A pair whose evidence first appears after T was invisible to the analyst.
   Including it leaks the future, and it is the easiest leak to miss because
   the pair looks perfectly ordinary in the final release.

3. chembl EVIDENCE IS KEPT BUT FLAGGED.
   The chembl datasource encodes existing clinical activity, so it correlates
   strongly with the label by construction. Dropping it is wrong (an analyst
   would know which targets were already in trials); silently keeping it
   inflates every model. We keep it and expose `--no-clinical` so the
   genetics-only regime can be reported separately. Any headline number should
   state which regime it used.

4. FAILURE != "TRIAL STOPPED".
   Only Negative (efficacy) and Safety_Sideeffects reflect target quality.
   Insufficient_Enrollment and Business_Administrative together outnumber them
   ~4:1 and are noise with respect to the biology.

5. THE KEY IS targetId, NOT (targetId, diseaseId).
   Open Targets migrated its disease ontology (EFO-dominant through 24.06,
   MONDO-dominant by 26.06). Measured stability of an id across releases:

       targetId    99.0 - 99.5 %      (Ensembl gene ids)
       diseaseId   31.4 - 43.4 %      (ontology churn)

   Keying on the pair silently destroys the label: 55,258 of 55,308 apparent
   "phase regressions" were join misses where phase_final defaulted to 0
   because the disease id had been remapped, making failures outnumber
   successes 4:1 out of pure artefact. Target-level keying is immune to this
   and matches how the genetic-support literature reports results. Indication
   resolution is recoverable later via an EFO->MONDO crosswalk, but the
   built-in one (disease.obsoleteTerms) covers only ~3.4k terms and is not
   sufficient on its own.

Usage:
    from analysis.split import TemporalSplit
    ts = TemporalSplit(ot).build("22.06")
    ts.check_leakage()
    ts.table()          # one row per (target, disease) with features + label
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.normalize_ot import RELEASES, OpenTargets  # noqa: E402

# Datasources that describe existing clinical activity rather than upstream
# biological evidence. Correlated with the label by construction.
CLINICAL_SOURCES = ("chembl",)

FINAL_RELEASE = "26.06"


@dataclass
class TemporalSplit:
    ot: OpenTargets
    cutoff: str = "22.06"
    final: str = FINAL_RELEASE
    min_phase_at_t: float = 0.0      # include pre-clinical pairs by default

    @property
    def con(self):
        return self.ot.con

    def build(self, cutoff: str | None = None) -> "TemporalSplit":
        if cutoff:
            self.cutoff = cutoff
        if self.cutoff not in RELEASES or self.final not in RELEASES:
            raise ValueError(f"unknown release: {self.cutoff} / {self.final}")
        if self.cutoff >= self.final:
            raise ValueError("cutoff must precede the final release")

        # --- features: evidence per (target, datasource) as of T ---
        self.con.execute(f"""
            CREATE OR REPLACE VIEW feat_t AS
            SELECT targetId, datasourceId,
                   max(score)                 AS max_score,
                   count(*)                   AS n_evidence,
                   count(DISTINCT diseaseId)  AS n_diseases
            FROM evidence
            WHERE release = '{self.cutoff}'
            GROUP BY 1,2""")

        # --- clinical stage at T and at the final release ---
        # A pair can carry several drugs; the pair's stage is the furthest any
        # of them reached.
        for tag, rel in (("t", self.cutoff), ("final", self.final)):
            self.con.execute(f"""
                CREATE OR REPLACE VIEW phase_{tag} AS
                SELECT targetId, max(phase) AS phase
                FROM labels WHERE release = '{rel}' AND phase IS NOT NULL
                GROUP BY 1""")

        # --- cohort: pairs visible at T, with what happened by `final` ---
        # LEFT JOIN on phase_t: a pair can have biological evidence at T with
        # no clinical activity yet (phase 0). Those are the interesting ones.
        self.con.execute(f"""
            CREATE OR REPLACE VIEW cohort AS
            SELECT e.targetId,
                   COALESCE(pt.phase, 0.0)  AS phase_t,
                   COALESCE(pf.phase, 0.0)  AS phase_final,
                   COALESCE(pf.phase, 0.0) - COALESCE(pt.phase, 0.0) AS phase_delta,
                   CAST(COALESCE(pf.phase,0.0) > COALESCE(pt.phase,0.0) AS INTEGER) AS advanced,
                   CAST(COALESCE(pf.phase,0.0) >= 3 AND COALESCE(pt.phase,0.0) < 3
                        AS INTEGER) AS reached_late_stage
            FROM (SELECT DISTINCT targetId FROM feat_t) e
            LEFT JOIN phase_t     pt ON pt.targetId = e.targetId
            LEFT JOIN phase_final pf ON pf.targetId = e.targetId
            WHERE COALESCE(pt.phase, 0.0) >= {self.min_phase_at_t}""")
        return self

    # ------------------------------------------------------------------ checks
    def check_leakage(self, verbose: bool = True) -> dict:
        """Assert the split cannot see the future. Raises if it can."""
        q = self.con.execute
        res = {}

        # 1. features must be built exclusively from the cutoff release
        res["future_feature_rows"] = q(f"""
            SELECT count(*) FROM evidence
            WHERE release > '{self.cutoff}'
              AND targetId IN (SELECT targetId FROM feat_t)
              AND datasourceId IN (SELECT datasourceId FROM feat_t)
              AND release = '{self.cutoff}'""").fetchone()[0]

        # 2. every cohort pair must actually exist at T
        res["pairs_absent_at_t"] = q(f"""
            SELECT count(*) FROM cohort c
            WHERE NOT EXISTS (SELECT 1 FROM evidence e
                              WHERE e.release='{self.cutoff}' AND e.targetId=c.targetId)
        """).fetchone()[0]

        # 3. labels must never regress -- a drop means the phase series is not
        #    monotone and "advancement" would be ill-defined
        res["negative_phase_delta"] = q(
            "SELECT count(*) FROM cohort WHERE phase_delta < 0").fetchone()[0]

        res["n_cohort"] = q("SELECT count(*) FROM cohort").fetchone()[0]
        res["n_advanced"] = q("SELECT sum(advanced) FROM cohort").fetchone()[0] or 0
        res["n_late_stage"] = q("SELECT sum(reached_late_stage) FROM cohort").fetchone()[0] or 0

        if verbose:
            print(f"cutoff={self.cutoff} -> final={self.final}")
            print(f"  cohort targets    : {res['n_cohort']:,}")
            print(f"  advanced          : {res['n_advanced']:,} "
                  f"({res['n_advanced']/max(res['n_cohort'],1):.2%})")
            print(f"  reached phase 3+  : {res['n_late_stage']:,}")
            print(f"  absent at T       : {res['pairs_absent_at_t']} (must be 0)")
            print(f"  phase regressions : {res['negative_phase_delta']}")

        if res["pairs_absent_at_t"]:
            raise AssertionError(
                f"{res['pairs_absent_at_t']} cohort pairs have no evidence at "
                f"{self.cutoff} -- the split is leaking future pairs")
        return res

    # ------------------------------------------------------------------ output
    def table(self, drop_clinical: bool = False):
        """One row per (target, disease): pivoted evidence scores + label."""
        srcs = [r[0] for r in self.con.execute(
            "SELECT DISTINCT datasourceId FROM feat_t ORDER BY 1").fetchall()]
        if drop_clinical:
            srcs = [s for s in srcs if s not in CLINICAL_SOURCES]
        cols = ",\n".join(
            f"  max(CASE WHEN datasourceId='{s}' THEN max_score END) AS \"src_{s}\""
            for s in srcs)
        self.con.execute(f"""
            CREATE OR REPLACE VIEW split_table AS
            SELECT c.*, f.n_sources, f.total_evidence, f.n_diseases, p.* EXCLUDE (targetId)
            FROM cohort c
            JOIN (SELECT targetId,
                         count(DISTINCT datasourceId) AS n_sources,
                         sum(n_evidence)              AS total_evidence,
                         max(n_diseases)              AS n_diseases
                  FROM feat_t GROUP BY 1) f USING (targetId)
            JOIN (SELECT targetId,
{cols}
                  FROM feat_t GROUP BY 1) p USING (targetId)""")
        return self.con.execute("SELECT * FROM split_table")


if __name__ == "__main__":
    ot = OpenTargets().build(verbose=False)
    for cutoff in ("22.06", "23.06", "24.06"):
        TemporalSplit(ot).build(cutoff).check_leakage()
        print()
