"""TrialBench as a second, EXTERNAL evaluation axis.

WHY
---
Everything else measured in this project is either self-generated (the cis-MVMR
simulator) or a 10-problem smoke test (the GeneBench-Pro public package). That
makes "we built a benchmark and did well on it" the strongest available claim.
TrialBench is published, CC-BY-4.4.0 [sic: CC-BY-4.0], has its own baselines in
the literature, and asks a question adjacent to arm A: given a trial's design,
will it succeed?

WHAT MAKES IT A BETTER FIT THAN ARM A FOR THE PROJECT'S ACTUAL THESIS
---------------------------------------------------------------------
The arm-A decision set hands the agent a rendered feature vector -- per
datasource association scores -- so the agent and the classifier see exactly
the same information and "reads evidence" is not really under test.

TrialBench ships FREE TEXT: brief_summary, eligibility criteria, intervention
description, keywords, plus SMILES. A gradient-boosted tree over the numeric
columns cannot use any of it; an agent can read all of it. That gap is the
headroom the project's central question is about, and it exists here without
needing to build a literature-injection pipeline first.

So the tabular baseline below is deliberately NOT tuned to be strong. It is the
honest "tabulates it" arm: everything a tree can trivially use, nothing it
cannot. `--dossiers` emits the same test rows as agent-readable text, including
the fields the classifier ignored, so the two are scored on identical rows.

WHAT THIS IS NOT
----------------
The shipped train/test split is TrialBench's own. **It is not verifiably
temporal**: no date column survives into the released features, so we cannot
confirm train precedes test the way analysis/split.py can for arm A. Do not
describe results here as a temporal backtest. If a temporal claim is needed,
rebuild the split from AACT (which this project already downloads) and join on
NCT id.

Metrics come from the arm-A modules so the two are directly comparable: AP and
AUC, enrichment@k, and ECE. Note the base rate here is ~55%, not 0.7%, so AUC
is far less misleading than it is on arm A -- but AP and ECE still lead.

Run:
  venv/bin/python analysis/trialbench.py --phase Phase3
  venv/bin/python analysis/trialbench.py --phase Phase3 --dossiers 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.agent_backtest import ece  # noqa: E402
from analysis.baselines import KS, enrichment_at_k  # noqa: E402
from paths import data_root  # noqa: E402

TASKS = (
    "trial-approval-forecasting",
    "trial-failure-reason-identification",
)

# Rendered into the agent's dossier; invisible to the tabular baseline. This
# split is the experiment: it is exactly the material a tree cannot consume.
TEXT_COLS = (
    "brief_title",
    "brief_summary/textblock",
    "condition",
    "condition_browse/mesh_term",
    "intervention/intervention_name",
    "intervention/description",
    "intervention_browse/mesh_term",
    "eligibility/criteria/textblock",
    "keyword",
)

# Unlabelled first column of every TrialBench CSV; it is the NCT id, not an
# index. Excluded from features, but it is the join key -- and it is also the
# hook for a genuinely temporal split later, since AACT (already downloaded
# here) carries start dates per NCT id.
ID = "Unnamed: 0"
DROP = (ID,)


def root(task: str) -> Path:
    return data_root() / "trialbench" / task


def load(task: str, phase: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    base = root(task) / phase
    if not base.is_dir():
        raise FileNotFoundError(
            f"{base} missing. Fetch it with:\n"
            f"  scripts/fetch_trialbench.sh"
        )
    out = []
    for split in ("train", "test"):
        x = pd.read_csv(base / f"{split}_x.csv", low_memory=False)
        y = pd.read_csv(base / f"{split}_y.csv")
        label = [c for c in y.columns if c not in DROP][-1]

        # THE TRAP: `Unnamed: 0` is the NCT id, and _x.csv ships UNSORTED while
        # _y.csv ships sorted. Pairing them positionally silently attaches the
        # wrong label to every row. It does not error and it does not look
        # broken -- the model trains happily to train-AUC 0.89 and lands at
        # test-AUC 0.488, which reads as "trial outcomes are unpredictable"
        # rather than "the labels are scrambled". Join on the id.
        if ID not in x.columns or ID not in y.columns:
            raise KeyError(f"{base}/{split}: expected join column {ID!r}")
        if set(x[ID]) != set(y[ID]):
            raise ValueError(f"{base}/{split}: x and y id sets differ; cannot join")
        merged = x.merge(y[[ID, label]], on=ID, how="inner", validate="one_to_one")
        out += [merged.drop(columns=[label]), merged[label].values]
    return out[0], out[1], out[2], out[3]


def tabular_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Numeric + low-cardinality categorical only.

    Free text is EXCLUDED on purpose -- see the module docstring. Categories are
    fit on train and applied to test so an unseen category becomes NaN rather
    than silently shifting the encoding (which would leak test structure into
    the feature space).
    """
    cols, num, cat = [], [], []
    for c in train.columns:
        if c in DROP or c in TEXT_COLS:
            continue
        s = pd.to_numeric(train[c], errors="coerce")
        if s.notna().mean() > 0.5:
            num.append(c)
        elif train[c].nunique(dropna=True) <= 40:
            cat.append(c)

    def build(df: pd.DataFrame, cats: dict[str, pd.Index] | None) -> tuple[np.ndarray, dict]:
        parts, made = [], {}
        for c in num:
            parts.append(pd.to_numeric(df[c], errors="coerce").to_numpy(float).reshape(-1, 1))
        for c in cat:
            levels = cats[c] if cats else pd.Index(sorted(train[c].dropna().astype(str).unique()))
            made[c] = levels
            codes = pd.Categorical(df[c].astype(str), categories=levels).codes.astype(float)
            codes[codes < 0] = np.nan          # unseen level -> missing, not level 0
            parts.append(codes.reshape(-1, 1))
        return np.hstack(parts), made

    Xtr, levels = build(train, None)
    Xte, _ = build(test, levels)
    cols = num + cat
    return Xtr, Xte, cols


def baselines(task: str, phase: str) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    xtr, ytr, xte, yte = load(task, phase)
    Xtr, Xte, cols = tabular_features(xtr, xte)
    print(f"{task} / {phase}")
    print(f"  train {Xtr.shape}  test {Xte.shape}  "
          f"features used {len(cols)} of {xtr.shape[1]} "
          f"({len(TEXT_COLS)} text columns withheld from the tree)")
    print(f"  base rate: train {ytr.mean():.1%}  test {yte.mean():.1%}")

    rows = []

    def add(name: str, p: np.ndarray):
        row = {"model": name, "n": len(yte),
               "AP": average_precision_score(yte, p),
               "AUC": roc_auc_score(yte, p),
               "ECE": ece(np.asarray(yte), p)}
        for k in KS:
            if k < len(yte):
                row[f"E@{k}"] = enrichment_at_k(np.asarray(yte), p, k)[1]
        rows.append(row)

    add("prevalence (constant)", np.full(len(yte), ytr.mean()))
    rng = np.random.default_rng(0)
    add("random", rng.random(len(yte)))
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf.fit(Xtr, ytr)
    add("hist gradient boosting", clf.predict_proba(Xte)[:, 1])
    return pd.DataFrame(rows)


def dossiers(task: str, phase: str, n: int, seed: int, out: Path) -> None:
    """Agent-readable rendering of the SAME test rows the baseline is scored on.

    Includes the free-text fields the tabular model could not use. Labels are
    kept out of the rendered text and carried alongside for scoring only --
    same discipline as analysis/decision_set.py.
    """
    _, _, xte, yte = load(task, phase)
    idx = np.arange(len(xte))
    if n and n < len(idx):
        idx = np.random.default_rng(seed).choice(idx, n, replace=False)
        idx.sort()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for i in idx:
            r = xte.iloc[i]
            lines = [f"CLINICAL TRIAL ({phase})", ""]
            for c in TEXT_COLS:
                v = r.get(c)
                if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
                    lines.append(f"{c}:")
                    lines.append(f"  {' '.join(v.split())[:1500]}")
                    lines.append("")
            lines.append("STRUCTURED FIELDS:")
            for c in xte.columns:
                if c in DROP or c in TEXT_COLS:
                    continue
                v = r.get(c)
                if pd.notna(v) and str(v).strip() not in ("", "nan"):
                    lines.append(f"  {c}: {v}")
            fh.write(json.dumps({
                "row": int(i),
                "text": "\n".join(lines),
                "_label": int(yte[i]),          # scoring only, never rendered
            }) + "\n")
    print(f"wrote {len(idx)} dossiers -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=TASKS[0], choices=TASKS)
    ap.add_argument("--phase", default="Phase3",
                    choices=("Phase1", "Phase2", "Phase3", "Phase4"))
    ap.add_argument("--dossiers", type=int, default=0,
                    help="also emit N agent-readable test dossiers (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = baselines(args.task, args.phase)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("  AP/ECE lead; AUC is reported for comparability with the literature.")
    print("  The tree never saw the free-text columns -- that gap is the point.")
    print("  NOTE: this split is TrialBench's own and is NOT verifiably temporal.")

    if args.dossiers:
        out = data_root() / "trialbench" / f"dossiers_{args.task}_{args.phase}.jsonl"
        dossiers(args.task, args.phase, args.dossiers, args.seed, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
