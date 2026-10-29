"""Feature-based baselines on the temporal target-identification split.

These stand in for the published classifiers the agent has to beat:

  * medRxiv 2026.02.19.26346536 -- MR-informed XGBoost over 11,482
    target-indication pairs, reporting ~6.4x enrichment over base rate.
  * arXiv 2606.14823 -- 26,278 target-disease pairs, post-2015 temporal
    validation, concluding genetic evidence has "limited practical predictive
    value".

We cannot rerun their code on their data, so we reproduce the *class* of model
-- a gradient-boosted tree over per-datasource evidence scores -- on our split,
and report the same family of metrics. That gives the agent an honest,
same-data bar rather than a number quoted from a different cohort.

Evaluation is genuinely temporal, not cross-validated:

    train   features @ 22.06  ->  advanced by 24.06
    test    features @ 24.06  ->  advanced by 26.06

Training on one horizon and testing on a later, disjoint one means no label
from the test period is visible at fit time. A random split of a single cutoff
would leak: targets are correlated through shared evidence, and the model
would be scored on the same horizon it was tuned on.

Metrics: with a ~1% positive rate, ROC AUC is misleading (it is dominated by
the vast negative class). Average precision and enrichment@k are the honest
summaries, so those lead; AUC is reported only because the literature does.

Run:  venv/bin/python analysis/baselines.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.normalize_ot import OpenTargets  # noqa: E402
from analysis.split import CLINICAL_SOURCES, TemporalSplit  # noqa: E402

warnings.filterwarnings("ignore")

TRAIN = ("22.06", "24.06")   # features at, label horizon
TEST = ("24.06", "26.06")
KS = (50, 100, 250, 500)


def frame(ot: OpenTargets, cutoff: str, final: str) -> pd.DataFrame:
    ts = TemporalSplit(ot, final=final).build(cutoff)
    ts.check_leakage(verbose=False)
    return ts.table().df()


def features(df: pd.DataFrame, drop_clinical: bool) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in df.columns if c.startswith("src_")]
    if drop_clinical:
        cols = [c for c in cols if c[4:] not in CLINICAL_SOURCES]
    cols = cols + ["n_sources", "total_evidence", "n_diseases", "phase_t"]
    if drop_clinical:
        # phase_t IS the clinical state; keeping it in a "genetics only" run
        # would smuggle the same signal back in through the front door.
        cols.remove("phase_t")
    return df[cols], cols


def enrichment_at_k(y: np.ndarray, s: np.ndarray, k: int) -> tuple[float, float]:
    """precision@k and its ratio to the base rate."""
    k = min(k, len(y))
    idx = np.argsort(-s)[:k]
    prec = y[idx].mean()
    base = y.mean()
    return prec, (prec / base if base > 0 else np.nan)


def evaluate(name: str, y: np.ndarray, s: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    row = {"model": name,
           "AP": average_precision_score(y, s),
           "AUC": roc_auc_score(y, s)}
    for k in KS:
        p, e = enrichment_at_k(y, s, k)
        row[f"P@{k}"] = p
        row[f"E@{k}"] = e
    return row


def run(drop_clinical: bool) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ot = OpenTargets(memory_limit="16GB").build(verbose=False)
    tr = frame(ot, *TRAIN)
    te = frame(ot, *TEST)

    # Score the same targets in both periods only if they exist in both; a
    # target absent at the test cutoff was not a live decision then.
    Xtr, cols = features(tr, drop_clinical)
    Xte, _ = features(te, drop_clinical)
    Xte = Xte[cols]
    ytr = tr["advanced"].to_numpy()
    yte = te["advanced"].to_numpy()

    regime = "genetics-only (no chembl, no phase_t)" if drop_clinical else "all sources"
    print(f"\n{'='*72}\nREGIME: {regime}\n{'='*72}")
    print(f"train  {TRAIN[0]}->{TRAIN[1]}  n={len(ytr):,}  pos={ytr.sum()} ({ytr.mean():.2%})")
    print(f"test   {TEST[0]}->{TEST[1]}  n={len(yte):,}  pos={yte.sum()} ({yte.mean():.2%})")
    print(f"features: {len(cols)}")

    rows = []

    # --- trivial reference points ---------------------------------------
    rng = np.random.default_rng(0)
    rows.append(evaluate("random", yte, rng.random(len(yte))))
    rows.append(evaluate("n_sources (heuristic)", yte,
                         te["n_sources"].to_numpy(dtype=float)))
    rows.append(evaluate("total_evidence (heuristic)", yte,
                         te["total_evidence"].to_numpy(dtype=float)))
    if not drop_clinical:
        rows.append(evaluate("phase_t (known clinical state)", yte,
                             te["phase_t"].to_numpy(dtype=float)))
    gen = "src_ot_genetics_portal"
    if gen in te.columns:
        rows.append(evaluate("genetics score alone", yte,
                             te[gen].fillna(0).to_numpy()))

    # --- learned models --------------------------------------------------
    lr = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                       StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    rows.append(evaluate("logistic regression", yte, lr.predict_proba(Xte)[:, 1]))

    # HistGradientBoosting handles NaN natively -- most source scores are
    # genuinely missing (the datasource has nothing for that target), which is
    # informative and should not be imputed to zero.
    gb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=0)
    gb.fit(Xtr, ytr)
    rows.append(evaluate("gradient boosting", yte, gb.predict_proba(Xte)[:, 1]))

    out = pd.DataFrame(rows)
    disp = out[["model", "AP", "AUC"] + [f"E@{k}" for k in KS]]
    print("\n" + disp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nbase rate = {yte.mean():.3%}   (E@k is precision@k / base rate)")
    return out


def main() -> int:
    all_src = run(drop_clinical=False)
    gen_only = run(drop_clinical=True)

    print(f"\n{'='*72}\nBAR FOR THE AGENT\n{'='*72}")
    for tag, df in (("all sources", all_src), ("genetics-only", gen_only)):
        best = df.loc[df["AP"].idxmax()]
        print(f"  {tag:<15} best AP {best['AP']:.3f} by {best['model']}"
              f"   (E@100 {best['E@100']:.1f}x)")
    print("\nAn agent must beat these on the SAME split to claim that reading"
          "\nevidence adds anything over tabulating it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
