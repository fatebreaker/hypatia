"""Score an agent on the temporal target-identification split.

The claim under test: does an agent that READS evidence beat a classifier that
TABULATES it, on the same frozen information?

Everything is scored on the identical 1,000-target decision set, against that
set's own base rate (21.5%), so the agent and the baselines face exactly the
same problem. Classifiers are fit on the full 22.06->24.06 cohort and then
predict on the decision set -- they never see a test label.

Metrics, in order of how much they should be trusted here:

  AP          average precision. The honest headline at this base rate.
  E@k         precision@k / base rate. What a triage list would actually buy.
  ECE         expected calibration error. An agent that says "0.8" should be
              right 80% of the time; a classifier's probability is meaningless
              if it is not. This is the axis where reading plausibly beats
              tabulating even when ranking does not.
  AUC         reported only because the literature does. At a 0.7% cohort base
              rate gradient boosting scores AUC 0.967 with AP 0.179 -- the AUC
              is dominated by ranking obvious negatives and flatters the model.

Abstention is tracked separately: an agent allowed to say "insufficient
evidence" trades coverage for precision, and a model that abstains on the hard
half while scoring well on the easy half has not solved the task.

Run:
  venv/bin/python analysis/agent_backtest.py --agent classifiers
  venv/bin/python analysis/agent_backtest.py --agent llm:Qwen3-8B:nothink
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.baselines import KS, enrichment_at_k, features  # noqa: E402
from analysis.normalize_ot import OpenTargets  # noqa: E402
from analysis.split import TemporalSplit  # noqa: E402
from paths import data_root  # noqa: E402

SYSTEM = """You are a drug-discovery target-assessment expert.

For one target you are given the evidence available AT A PAST CUTOFF DATE.
Judge whether a drug programme against this target subsequently ADVANCED to a
later clinical phase than it stood at that cutoff.

Weigh the evidence rather than counting it. Consider:
  - which datasources carry causal weight (human genetics, rare-variant burden,
    curated gene-disease validity) versus weak ones (text-mined co-occurrence
    is presence of a sentence, not evidence of causality)
  - whether a high score reflects real biology or an artefact of how the
    datasource is built
  - whether existing clinical activity already implies momentum
  - tractability and safety liabilities

Return ONE JSON object and nothing else:
{
  "score": <integer 0-100, your calibrated chance in percent that it advanced>,
  "key_evidence": [<datasource names you actually relied on>],
  "discounted": [<datasource names you judged misleading or weak here>],
  "reasoning": "<2-3 sentences>"
}
Be calibrated: 70 means you expect roughly 70 of every 100 such targets to have
advanced, so reserve scores above 70 for genuinely strong cases.

Use the WHOLE range and be deliberate about the exact number. Do not fall back
on round values like 10, 25, 50 or 80 -- if two targets are not equally likely,
they must not get the same score. Ties are not neutral here: this is scored by
ranking as well as calibration, and every tie discards information."""


def load_set(cutoff: str, final: str, root: Path):
    ds = pd.read_parquet(root / f"decision_set_{cutoff}_{final}.parquet")
    dos = [json.loads(l) for l in open(root / f"dossiers_{cutoff}_{final}.jsonl")]
    return ds, {d["targetId"]: d for d in dos}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if not m.any():
            continue
        e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def score(name: str, y: np.ndarray, s: np.ndarray, *,
          probs: np.ndarray | None = None, n_abstain: int = 0) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    row = {"model": name, "n": len(y), "AP": average_precision_score(y, s),
           "AUC": roc_auc_score(y, s)}
    for k in KS:
        p, e = enrichment_at_k(y, s, k)
        row[f"E@{k}"] = e
    row["ECE"] = ece(y, probs) if probs is not None else np.nan
    row["abstained"] = n_abstain
    return row


# --------------------------------------------------------------------------- #
def classifier_scores(ds: pd.DataFrame) -> list[dict]:
    """Fit on the full earlier cohort, predict on the decision set."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ot = OpenTargets(memory_limit="16GB").build(verbose=False)
    tr = TemporalSplit(ot, final="24.06").build("22.06").table().df()

    out = []
    for tag, drop in (("all sources", False), ("genetics-only", True)):
        Xtr, cols = features(tr, drop)
        Xte = ds[cols]
        ytr, yte = tr["advanced"].to_numpy(), ds["advanced"].to_numpy()

        lr = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                           StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(Xtr, ytr)
        p = lr.predict_proba(Xte)[:, 1]
        out.append(score(f"logistic ({tag})", yte, p, probs=p))

        gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                            random_state=0)
        gb.fit(Xtr, ytr)
        p = gb.predict_proba(Xte)[:, 1]
        out.append(score(f"grad boosting ({tag})", yte, p, probs=p))

    out.append(score("n_sources heuristic", ds["advanced"].to_numpy(),
                     ds["n_sources"].to_numpy(dtype=float)))
    return out


def ask(base_url: str, model: str, text: str, thinking: bool | None,
        timeout: int = 300) -> dict | None:
    body = {"model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}],
            "temperature": 0.2, "max_tokens": 1024}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = json.loads(r.read())["choices"][0]["message"].get("content") or ""
    except Exception:                                            # noqa: BLE001
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
    from agents import extract_answer, strip_reasoning            # noqa: E402
    obj = extract_answer(strip_reasoning(txt))
    if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    return obj if isinstance(obj, dict) else None


def agent_scores(ds: pd.DataFrame, dos: dict, spec: str, base_url: str,
                 workers: int, out_path: Path) -> list[dict]:
    import concurrent.futures as cf

    model = spec.split(":", 1)[1]
    thinking = None
    if model.endswith(":nothink"):
        model, thinking = model[: -len(":nothink")], False

    def one(tid: str):
        return tid, ask(base_url, model, dos[tid]["text"], thinking)

    preds: dict[str, dict | None] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (tid, obj) in enumerate(ex.map(one, ds.targetId.tolist()), 1):
            preds[tid] = obj
            if i % 100 == 0:
                print(f"  ... {i}/{len(ds)}", flush=True)

    with open(out_path, "w") as fh:
        for tid, obj in preds.items():
            fh.write(json.dumps({"targetId": tid, "pred": obj}) + "\n")

    p, y, abst = [], [], 0
    for tid, lab in zip(ds.targetId, ds.advanced):
        o = preds.get(tid)
        v = None
        if isinstance(o, dict):
            # `score` is 0-100 by design, not 0-1: asked as a percent because a
            # model asked for a probability answers in round decimals, and AP is
            # a RANKING metric that pays for every tie. Measured cost of that
            # (analysis/power.py): ~6 distinct values loses 0.06 AP, which is
            # the entire effect this decision set can resolve -- a competitive
            # agent would read as a clear loss for purely formatting reasons.
            # `probability` is still accepted so a model that ignores the
            # instruction is scored rather than counted as an abstention.
            try:
                v = float(o["score"]) / 100.0
            except (TypeError, ValueError, KeyError):
                try:
                    v = float(o.get("probability"))
                except (TypeError, ValueError):
                    v = None
        if v is None or not (0.0 <= v <= 1.0):
            abst += 1
            v = 0.0     # unusable output ranks last rather than being dropped,
                        # so an agent cannot improve its score by refusing
        p.append(v)
        y.append(lab)
    p, y = np.array(p), np.array(y)
    return [score(spec, y, p, probs=p, n_abstain=abst)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="classifiers")
    ap.add_argument("--cutoff", default="24.06")
    ap.add_argument("--final", default="26.06")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else data_root() / "backtest"
    ds, dos = load_set(args.cutoff, args.final, root)
    print(f"decision set: {len(ds)} targets, base rate {ds.advanced.mean():.1%}")

    rows = (classifier_scores(ds) if args.agent == "classifiers"
            else agent_scores(ds, dos, args.agent, args.base_url, args.workers,
                              root / f"preds_{args.agent.replace(':','_')}.jsonl"))

    df = pd.DataFrame(rows)
    cols = ["model", "AP", "AUC"] + [f"E@{k}" for k in KS] + ["ECE", "abstained"]
    print("\n" + df[cols].to_string(index=False,
                                    float_format=lambda v: f"{v:.3f}"))
    res = root / f"backtest_{args.agent.replace(':','_')}.json"
    df.to_json(res, orient="records", indent=2)
    print(f"\nwrote {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
