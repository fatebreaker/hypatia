"""causal_disc2_build.py (10-25, docs/RL_PLAN.md H-QR4-FG) -- causal_disc_build.py with NATURAL pairs and QRData's option formats:
  * the asked pair is drawn UNIFORMLY from all pairs with an unambiguous answer (a direct edge, or no directed path either way) of a
    SPARSE random DAG (edge probability 0.1-0.3), so "no causal relationship" appears at the rate the graph implies; causal_disc
    drew the true LETTER uniformly, which made "no relationship" 25% of its training questions (QRData-Verified: 49 of 87);
  * QRData's three formats: 45% three-option (A x causes y, B y causes x, C none), 45% four-option (C "the causal relation is
    double sided", never true in a DAG; D none), 10% two-option pairwise (edges only) -- QRData-Verified has 42 / 40 / 5.
Everything else (SCMs, sizes, the solvability filter, grader) is causal_disc_build.py's. Original docstring:
generated/causal_disc: synthetic causal-discovery questions in QRData's own format and grader (qrdata_official,
multiple_choice), on data from random causal graphs.

Each instance: a random DAG over 4-8 variables with neutral names (no domain knowledge helps), n = 600-2,500 observational
rows from a linear non-Gaussian SCM (uniform / Laplace / centred-exponential noise) or a nonlinear additive-noise SCM
(tanh, cubic, sin, softplus; Gaussian or uniform noise) -- both identifiable from observational data -- and ONE question,
QRData's wording: "Which cause-and-effect relationship is more likely? A. X causes Y B. Y causes X [C. No causal
relationship exists]". Asked only about pairs with an unambiguous answer: a DIRECT edge (A / B) or a pair with no directed
path either way (C; such pairs may still be correlated through a common cause). Pairs linked only indirectly are never
asked. The true letter is drawn uniformly (balanced in expectation). 70% three-option ('full graph') and 30% two-option
('pairwise', edges only) questions, as QRData has both.

Solvability filter (as causal_mix keeps only instances its reference estimator solves): a reference analysis on the
sampled data must recover the answer -- for C, X and Y are conditionally independent given some set of at most two other
variables (Fisher-z partial correlation, p > 0.05, after a rank transform for nonlinear data); for an edge, X and Y are
dependent given every such set, and the direction test (residual-independence for additive noise: the residual of the
regression in the causal direction is the less dependent one on the regressor, by a distance-correlation gap) points the
right way. Instances the reference cannot solve are redrawn (logged as gen_tries).

    python3 scripts/causal_disc2_build.py [--n-train 800] [--n-test 100]
Refuses to overwrite an existing pool. Writes generated/causal_disc/{train,test}/<id>/{data_files/<name>.csv,
eval_config.json} and generated/causal_disc/MANIFEST.json.
"""
import argparse
import itertools
import json
import os
import string
import sys

import numpy as np

G = "/scratch4/workspace/hieutran_umass_edu-deep-ai4s/generated"
OUT = f"{G}/causal_disc2"
PREFIX = ["marker", "signal", "index", "level", "rate", "score", "conc", "flux", "load", "count"]


def names(rng, p):
    out = set()
    while len(out) < p:
        out.add(f"{rng.choice(PREFIX)}_{rng.choice(list(string.ascii_uppercase))}{rng.integers(1, 10)}")
    return list(out)


def random_dag(rng, p):
    order = rng.permutation(p)
    adj = np.zeros((p, p), dtype=bool)            # adj[i, j]: i -> j
    prob = rng.uniform(0.1, 0.3)   # causal_disc2: sparse graphs
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < prob:
                adj[order[a], order[b]] = True
    return adj, order


def paths(adj):
    p = len(adj)
    reach = adj.copy()
    for k in range(p):                             # transitive closure
        reach = reach | (reach[:, [k]] & reach[[k], :])
    return reach


def simulate(rng, adj, order, n, kind):
    p = len(adj)
    X = np.zeros((n, p))
    funcs = [np.tanh, lambda v: v ** 3 / 3.0, np.sin, lambda v: np.log1p(np.exp(v)) - 0.7]
    for j in order:
        par = np.where(adj[:, j])[0]
        if kind == "linear":
            dist = rng.choice(["uniform", "laplace", "exp"])
            e = {"uniform": rng.uniform(-1.7, 1.7, n), "laplace": rng.laplace(0, 0.7, n),
                 "exp": rng.exponential(1.0, n) - 1.0}[dist]
            s = np.zeros(n)
            for i in par:
                s += rng.choice([-1, 1]) * rng.uniform(0.6, 1.4) * X[:, i]
        else:
            e = rng.normal(0, 0.5, n) if rng.random() < 0.5 else rng.uniform(-0.9, 0.9, n)
            s = np.zeros(n)
            for i in par:
                f = funcs[rng.integers(len(funcs))]
                z = (X[:, i] - X[:, i].mean()) / (X[:, i].std() + 1e-9)
                s += rng.choice([-1, 1]) * rng.uniform(0.8, 1.6) * f(z)
        X[:, j] = s + e
    return X


def rank_gauss(x):
    from scipy.stats import norm, rankdata
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def pcorr_p(D, i, j, S):
    from scipy.stats import norm
    idx = [i, j] + list(S)
    C = np.corrcoef(D[:, idx], rowvar=False)
    try:
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return 0.0
    r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    r = float(np.clip(r, -0.999999, 0.999999))
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(len(D) - len(S) - 3)
    return float(2 * (1 - norm.cdf(abs(z))))


def separable(D, i, j, p):
    others = [k for k in range(p) if k not in (i, j)]
    for size in (0, 1, 2):
        for S in itertools.combinations(others, size):
            if pcorr_p(D, i, j, S) > 0.05:
                return True
    return False


def dcor(x, y):
    n = len(x)
    if n > 700:                                    # subsample: distance correlation is O(n^2)
        idx = np.random.default_rng(0).choice(n, 700, replace=False)
        x, y = x[idx], y[idx]
    a = np.abs(x[:, None] - x[None, :]); b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean(); B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    v = (A * B).mean(); vx = (A * A).mean(); vy = (B * B).mean()
    return float(np.sqrt(max(v, 0) / np.sqrt(vx * vy))) if vx > 0 and vy > 0 else 0.0


def resid(x, y, kind):
    xs = (x - x.mean()) / (x.std() + 1e-9)
    deg = 1 if kind == "linear" else 5
    coef = np.polyfit(xs, y, deg)
    return y - np.polyval(coef, xs)


def direction_ok(X, i, j, kind, margin=0.02):
    """True if the residual-independence test prefers i -> j (the truth) by at least `margin`."""
    xi, xj = X[:, i], X[:, j]
    fwd = dcor(xi, resid(xi, xj, kind))            # i -> j: residual of j on i independent of i
    bwd = dcor(xj, resid(xj, xi, kind))
    return bwd - fwd >= margin


def build_one(rng, iid):
    for tries in range(1, 60):
        p = int(rng.integers(4, 9))
        adj, order = random_dag(rng, p)
        if adj.sum() == 0:
            continue
        reach = paths(adj)
        kind = "linear" if rng.random() < 0.5 else "nonlinear"
        n = int(rng.integers(600, 2501))
        X = simulate(rng, adj, order, n, kind)
        D = X if kind == "linear" else np.column_stack([rank_gauss(X[:, k]) for k in range(p)])
        r = rng.random()
        form = "pairwise" if r < 0.10 else ("three" if r < 0.55 else "four")
        three = form != "pairwise"
        edges = [(i, j) for i in range(p) for j in range(p) if adj[i, j]]
        unrel = [(i, j) for i in range(p) for j in range(i + 1, p) if not reach[i, j] and not reach[j, i]]
        cand = [("edge", i, j) for i, j in edges] + ([("none", i, j) for i, j in unrel] if three else [])
        what, i, j = cand[rng.integers(len(cand))]          # uniform over the unambiguous pairs (pairwise: edges only)
        if what == "none":
            if not separable(D, i, j, p):
                continue
            x, y = (i, j) if rng.random() < 0.5 else (j, i)
            want = "C" if form == "three" else "D"
        else:                                                # i -> j
            if separable(D, i, j, p) or not direction_ok(X, i, j, kind):
                continue
            x, y = (i, j) if rng.random() < 0.5 else (j, i)
            want = "A" if (x, y) == (i, j) else "B"       # option A is "x causes y"
        nm = names(rng, p)
        return dict(adj=adj, X=X, kind=kind, n=n, p=p, names=nm, x=x, y=y, answer=str(want), three=bool(three), form=form, tries=tries)
    return None


def write(inst, root, iid, seed):
    d = f"{root}/{iid}"
    os.makedirs(f"{d}/data_files")
    fn = f"{iid.split('_')[-1]}_data.csv"
    nm, X = inst["names"], inst["X"]
    with open(f"{d}/data_files/{fn}", "w") as fh:
        fh.write(",".join(nm) + "\n")
        for row in X:
            fh.write(",".join(f"{v:.5g}" for v in row) + "\n")
    a, b = nm[inst["x"]], nm[inst["y"]]
    opts = f"A. {a} causes {b} B. {b} causes {a}"
    if inst["form"] == "three":
        opts += " C. No causal relationship exists"; ask = "Please answer with A, B, or C."
    elif inst["form"] == "four":
        opts += f" C. The causal relation is double sided between {a} and {b} D. No causal relationship exists"
        ask = "Please answer with A, B, C, or D."
    else:
        ask = "Please answer with A or B."
    key = f"{iid}-q1"
    task = ("You are given the data files from a published statistical analysis.\nAnswer the question below by computing the result "
            f"from the data.\n\nThe CSV file {fn} contains {inst['n']} observational measurements of {inst['p']} variables "
            "from a system whose causal structure is unknown.\n\n"
            f"Files in data_files/:\n  {fn}\n\nQuestion: Which cause-and-effect relationship is more likely? {opts} {ask}\n\n"
            f"Return your answer as JSON: {{\"answer\": {{\"{key}\": <value>}}}}")
    cfg = {"id": iid, "task": task, "data_files": [fn], "ground_truth": {key: inst["answer"]},
           "grader": {"type": "qrdata_official", "config": {"keys": {key: {"question_type": "multiple_choice"}}}},
           "_provenance": {"family": "causal_disc2", "seed": seed, "gen_tries": inst["tries"], "scm": inst["kind"],
                           "n": inst["n"], "p": inst["p"], "form": "full graph, " + inst["form"] + " options" if inst["three"] else "pairwise",
                           "edges": [[nm[i], nm[j]] for i in range(inst["p"]) for j in range(inst["p"]) if inst["adj"][i, j]],
                           "asked": [a, b], "keywords": ["Causality", ("Full graph" if inst["three"] else "Pairwise") +
                                                         " causal discovery", "Observational data", "synthetic"]}}
    json.dump(cfg, open(f"{d}/eval_config.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261025)   # causal_disc2: fresh draws
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists -- not overwriting")
    counts = {}
    for split, n0, base in (("train", a.n_train, 0), ("test", a.n_test, 10_000_000)):
        os.makedirs(f"{a.out}/{split}")
        made = 0
        k = 0
        letters = {"A": 0, "B": 0, "C": 0, "D": 0}
        while made < n0:
            seed = a.seed + base + k
            k += 1
            inst = build_one(np.random.default_rng(seed), None)
            if inst is None:
                continue
            iid = f"causal_disc2_{split}_{made:05d}"
            write(inst, f"{a.out}/{split}", iid, seed)
            letters[inst["answer"]] += 1
            made += 1
        counts[split] = {"n": made, "answers": letters}
        print(split, counts[split], flush=True)
    json.dump({"built": "2026-10-25", "builder": "scripts/causal_disc2_build.py", "seed": a.seed, "splits": counts},
              open(f"{a.out}/MANIFEST.json", "w"), indent=1)


if __name__ == "__main__":
    main()
