"""Ecology: closed-population abundance from capture-recapture (a SAGE plug-in).

The data are a short trapping or netting session as a field crew records it. Every animal is
tagged at its first capture; one that later turns up without its tag is recognised by a second
mark (a PIT tag) and given a new tag, with the old number noted. The session is short, so the
population is closed (no births, deaths or movement). The DGP builds in the irregularities real
capture-recapture data carry:

  * individual heterogeneity in catchability: by body length in `eco_size` (nets select big
    fish), by sex in `eco_mb`; the animals never caught are mostly the hard-to-catch ones, so a
    model with one capture probability (M0) underestimates N;
  * a behavioural response in `eco_mb`: after its first capture an animal becomes trap-happy
    (bait) or trap-shy (handling), so recaptures misstate the chance of a first capture;
  * optional tag loss (half the draws): a re-tagged animal appears under a second tag, and left
    unmerged it becomes a phantom "new" animal first caught late.

Reference: Huggins' (1989) conditional likelihood for the animals caught, logit P(capture) linear
in the individual covariates (plus a behavioural coefficient after first capture in `eco_mb`),
and abundance as the Horvitz-Thompson sum of 1/P(caught at least once) (Huggins 1989; Alho 1990),
SE by the delta method. Tolerance: 2 SE of the reference, with a floor (2% of N; 5% of the odds
ratio).

Families, answer keys and traps (the step each skips; skills from the shared taxonomy):

  eco_size  fish in a lake, size-selective nets, 4-8 nights, a per-tag capture-history file; keys N
            (fish of at least the minimum size) and N_legal (fish at least the legal length)
    1 qc            tag_loss_unmerged      re-tagged fish counted twice (cond: tag loss drawn)
    2 estimate      distinct_count         the fish seen, no detection correction
    3 estimate      lincoln_petersen       Chapman's estimator on nights 1 and 2 only
    4 model-choice  m0_equal_catchability  one capture probability for every size
    5 model-choice  m0_by_length_class     M0 within length quintiles, summed (cond: live where the
                                           smallest class spans a steep part of the selectivity curve)
    -  variant      length_quadratic       logit p quadratic in length (nests the reference)
  eco_mb    crabs or lobsters in baited pots, 4-8 days, a long capture log; keys N_female, N_male
            and trap_response (odds ratio of capture after vs before an animal's first capture)
    1 qc            tag_loss_unmerged      (cond: tag loss drawn)
    2 estimate      distinct_count
    3 estimate      lincoln_petersen       per sex; recaptures on day 2 measure c, not p
    4 model-choice  m0_equal_catchability  no sex effect, no behaviour (trap_response 1)
    4 model-choice  ignore_sex             behaviour but no sex effect (cond: live on most draws)
    5 model-choice  ignore_trap_response   sex but no behaviour (trap_response 1)

Why not a latent two-class mixture (Pledger's Mh) as the gold: prototyped first, with 4-8
occasions it was identifiable (relative SE <= 5%) on only half the draws, landed within 2 SE of
the truth on 80-86% of those, and the capture frequencies of identifiable draws pin N/S to
1.15-1.4 -- so "1.25 x the animals seen" passed 50-60% of instances. A covariate carries the
heterogeneity instead: N/S then spans 1.3-2.5 and the gold is precise. For the same reason the
behavioural family is set in large populations (2,000-15,000 crabs) with low first-capture
probabilities, and asks for each sex's abundance: with a few hundred animals, "1.3 x the animals
seen" plus a trap response read off the raw recapture rate passed about a third of instances.

A conditional trap is recorded with step None on draws where it coincides with the reference.
The analysis functions are self-contained (numpy/scipy/pandas/statsmodels), so a demonstration
embeds their source verbatim: the trace and the ground truth are the same code.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logit
from statsmodels.tools.numdiff import approx_fprime, approx_hess

from sage.core import Cell, Family, Sample, Trap

CAPTURES, HISTORIES = "captures.csv", "capture_histories.csv"
NAN = float("nan")


# ----------------------------------------------------------------------------- analysis
def capture_histories(log, K, merge_retags=True):
    """Long capture log -> 0/1 histories, one row per animal (indexed by its first tag), one column
    per night. A re-tagged animal keeps its first tag's identity: follow previous_tag links back."""
    link = dict(log.dropna(subset=["previous_tag"])[["tag", "previous_tag"]].values) if merge_retags else {}

    def first_tag(t):
        while t in link:
            t = link[t]
        return t
    H = pd.crosstab(log["tag"].map(first_tag), log["occasion"]).reindex(columns=range(1, K + 1), fill_value=0)
    return (H > 0).astype(int)


def merge_retagged(fish, merge_retags=True):
    """Wide histories (one row per tag, columns night_1..night_K) -> one row per fish: the rows of a
    re-tagged fish are combined into its first tag's row (previous_tag links); length at first tagging."""
    link = dict(fish.dropna(subset=["previous_tag"])[["tag", "previous_tag"]].values) if merge_retags else {}

    def first_tag(t):
        while t in link:
            t = link[t]
        return t
    nights = [c for c in fish.columns if c.startswith("night_")]
    H = fish[nights].groupby(fish["tag"].map(first_tag).to_numpy()).max()
    return H.to_numpy(int), fish.set_index("tag")["length_mm"].reindex(H.index).to_numpy(float)


def huggins(H, X, behaviour=True):
    """Huggins' (1989) conditional likelihood of the animals caught: logit P(capture) = X b before an
    animal's first capture and X b + c after it (the behavioural response; no c if not
    `behaviour`). Returns the coefficients (b..., c), their covariance, and total(mask): the
    Horvitz-Thompson abundance sum 1/P(caught at least once) over the animals in `mask` (default
    all), with its delta-method SE (Huggins 1989)."""
    H, X = np.asarray(H, float), np.asarray(X, float)
    K, q = H.shape[1], X.shape[1]
    after = np.arange(K) > H.argmax(1)[:, None]                          # nights after the first capture
    log_p0 = lambda t: K * log_expit(-(X @ t[:q]))                      # log P(never caught)

    def nll(t):
        eta = (X @ t[:q])[:, None] + (t[q] * after if behaviour else 0.0)
        return -(np.sum(H * log_expit(eta) + (1 - H) * log_expit(-eta)) - np.sum(np.log1p(-np.exp(log_p0(t)))))
    t = minimize(nll, np.r_[logit(H.mean()), np.zeros(q - 1 + behaviour)], method="BFGS").x
    cov = np.linalg.pinv(approx_hess(t, nll))

    def total(mask=None):
        m = np.ones(len(H), bool) if mask is None else np.asarray(mask, bool)
        n_hat = lambda t: np.sum(1 / -np.expm1(log_p0(t)[m]))
        ps, g = -np.expm1(log_p0(t)[m]), approx_fprime(t, n_hat)
        return float(n_hat(t)), float(np.sqrt(np.sum((1 - ps) / ps ** 2) + g @ cov @ g))
    return t, cov, total


def lincoln_petersen(H):
    """TRAP: Chapman's two-sample estimator on the first two nights only."""
    H = np.asarray(H, int)
    n1, n2, m2 = H[:, 0].sum(), H[:, 1].sum(), (H[:, 0] & H[:, 1]).sum()
    return float((n1 + 1) * (n2 + 1) / (m2 + 1) - 1)


# ----------------------------------------------------------------------------- simulator
def simulate_session(rng, p_first, p_after, K, loss_rate, prefix, first_tag):
    """One session's capture log. Animal i is caught on a night with probability p_first[i] until its
    first capture and p_after[i] afterwards; a tagged animal loses its tag between nights with
    probability loss_rate, and at its next capture gets a new tag (previous_tag = the old one). Tags
    are issued in order of handling. Returns rows (occasion, tag, previous_tag, animal index)."""
    N = len(p_first)
    caught, tagged = np.zeros(N, bool), np.zeros(N, bool)
    tag = np.zeros(N, int)
    nxt, rows = first_tag, []
    for k in range(1, K + 1):
        y = np.flatnonzero(rng.random(N) < np.where(caught, p_after, p_first))
        rng.shuffle(y)
        for i in y:
            prev = ""
            if not tagged[i]:
                if caught[i]:
                    prev = f"{prefix}{tag[i]:04d}"
                tag[i], nxt = nxt, nxt + 1
            tagged[i] = True
            rows.append((k, f"{prefix}{tag[i]:04d}", prev, int(i)))
        caught[y] = True
        tagged &= ~(rng.random(N) < loss_rate)
    return rows


def _dates(rng, K, months=(5, 9)):
    start = dt.date(int(rng.integers(2008, 2024)), months[0], 1) + dt.timedelta(days=int(rng.integers(0, 30 * (months[1] - months[0]))))
    return [(start + dt.timedelta(days=k)).isoformat() for k in range(K)]


def _prep(rng, knobs, has_loss):
    return rng.uniform(0.03, 0.07) * knobs["loss"] if has_loss else 0.0


# ----------------------------------------------------------------------------- shared family base
class CaptureRecapture(Family):
    domain = "ecology"
    max_tries = 12
    rng_policy = "stream"
    demo_timeout = 240

    def knobs(self, rng):
        return {"n": 1.0, "loss": 1.0}

    def strengthen(self, k, dead):
        k = dict(k)
        if set(dead) - {"tag_loss_unmerged"}:
            k["n"] = min(k["n"] * 1.5, 4.0)          # more animals: a smaller SE, so every bias stands out
        if "tag_loss_unmerged" in dead:
            k["loss"] = min(k["loss"] * 1.6, 4.0)
        return k

    def extra_provenance(self, inst) -> dict:
        m = inst.sample.data["meta"]
        return {"knobs": inst.knobs, "design": {k: m[k] for k in ("K", "dates", "legal_mm") if k in m},
                "reference_diagnostics": {k: v for k, v in inst.a.gold.items() if k.startswith("_")},
                "skills": {str(s): v for s, v in self.skills.items()}}

    def _cached(self, data, key, fn):
        cache = data.setdefault("_cache", {})
        if key not in cache:
            cache[key] = fn()
        return cache[key]


# ----------------------------------------------------------------------------- eco_size
FISH = (("walleye", "Sander vitreus", 381, (330, 450), (60, 90)),
        ("largemouth bass", "Micropterus salmoides", 305, (260, 360), (45, 75)),
        ("smallmouth bass", "Micropterus dolomieu", 356, (280, 380), (45, 75)),
        ("brown trout", "Salmo trutta", 305, (250, 380), (50, 80)),
        ("northern pike", "Esox lucius", 610, (480, 640), (80, 120)))
GEAR = ("fyke nets", "trap nets", "hoop nets")


class SizeSelective(CaptureRecapture):
    name = "eco_size"
    keys = ("N", "N_legal")
    steps = {1: "combine the records of re-tagged fish (tag loss) into one history per fish",
             2: "estimate abundance, correcting for the fish never caught",
             3: "use every netting night, not a two-sample Lincoln-Petersen estimate",
             4: "model size-dependent catchability (capture probability by length, not M0)",
             5: "model catchability as a smooth function of length, not constant within coarse length classes"}
    skills = {1: "qc", 2: "estimate", 3: "estimate", 4: "model-choice", 5: "model-choice"}
    seed_blocks = {"train": 15_000_000, "test": 15_050_000}
    tol_rule = "each key: max(2% of the estimate, 2 SE), SE by the delta method for the Horvitz-Thompson total"
    note = ("ground truth is the reference analysis (re-tagged fish merged; Huggins conditional likelihood with "
            "logit capture probability linear in length; Horvitz-Thompson totals of all fish and of legal-size fish) "
            "on the sampled histories, not the latent abundance; deviation is in units of tolerance, maxed over the "
            "answer keys")

    def draw(self, rng, k) -> Sample:
        sp = FISH[int(rng.integers(len(FISH)))]
        K = int(rng.integers(4, 9))
        N = int(round(np.exp(rng.uniform(np.log(800), np.log(8000))) * k["n"]))
        mu, sd = rng.uniform(*sp[3]), rng.uniform(*sp[4])
        l_min = 10 * np.floor((mu - 1.6 * sd) / 10)                      # the population: fish >= l_min
        L = mu + sd * rng.standard_normal(3 * N)
        L = np.round(L[L >= l_min][:N])
        slope = rng.uniform(0.8, 2.0) * 100 / sd                         # per 100 mm, scaled to the spread
        p_mid = rng.uniform(0.07, 0.25)
        p = expit(logit(p_mid) + slope * (L - mu) / 100)
        rows = simulate_session(rng, p, p, K, _prep(rng, k, rng.random() < 0.5), "F", int(rng.integers(1, 9)) * 1000)
        log = pd.DataFrame(rows, columns=["occasion", "tag", "previous_tag", "_animal"])
        wide = pd.crosstab(log["tag"], log["occasion"]).reindex(columns=range(1, K + 1), fill_value=0).clip(upper=1)
        first = log.drop_duplicates("tag").set_index("tag")
        fish = pd.DataFrame({"tag": wide.index, "length_mm": L[first.loc[wide.index, "_animal"]].astype(int),
                             "previous_tag": first.loc[wide.index, "previous_tag"].replace("", np.nan).to_numpy()})
        for j in range(1, K + 1):
            fish[f"night_{j}"] = wide[j].to_numpy()
        fish = fish.sort_values("tag", kind="stable").reset_index(drop=True)
        legal = int(sp[2])
        meta = {"K": K, "species": sp[:2], "legal_mm": legal, "l_min": int(l_min), "gear": GEAR[int(rng.integers(3))],
                "lake": f"a {int(rng.integers(40, 400))}-ha lake", "dates": _dates(rng, K)}
        latent = {"N": N, "N_legal": int((L >= legal).sum()), "p_mid": float(p_mid), "slope_per_100mm": float(slope),
                  "length_mean": float(mu), "length_sd": float(sd), "retagged": int(fish["previous_tag"].notna().sum()),
                  "p_star": float(np.mean(1 - (1 - p) ** K))}
        return Sample({"fish": fish, "meta": meta}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["fish"].to_csv(d / HISTORIES, index=False)
        return [HISTORIES]

    def _fit(self, data, merge=True, model="linear"):
        def run():
            H, L = merge_retagged(data["fish"], merge)
            z = (L - 350) / 100
            X = {"linear": np.column_stack([np.ones(len(H)), z]), "m0": np.ones((len(H), 1)),
                 "quadratic": np.column_stack([np.ones(len(H)), z, z ** 2])}[model]
            t, cov, total = huggins(H, X, behaviour=False)
            (N, se), (Nl, sel) = total(), total(L >= data["meta"]["legal_mm"])
            return {"N": N, "N_legal": Nl, "_se": se, "_se_legal": sel, "_S": len(H),
                    "_S_legal": int((L >= data["meta"]["legal_mm"]).sum()), "_slope": float(t[1]) if len(t) > 1 else NAN}
        return self._cached(data, ("_fit", merge, model), run)

    def reference(self, data) -> dict:
        return dict(self._fit(data))

    def tolerance(self, gold) -> dict:
        return {"N": max(0.02 * gold["N"], 2 * gold["_se"]), "N_legal": max(0.02 * gold["N_legal"], 2 * gold["_se_legal"])}

    def sanity(self, gold, tol, latent) -> list[str]:
        ok = all(np.isfinite(gold[k]) and 0 < gold[k] for k in ("_se", "_se_legal")) and gold["_se"] < 0.08 * gold["N"]
        return [] if ok else ["N_not_identified"]

    def traps(self, data) -> dict:
        H, L = merge_retagged(data["fish"])
        legal = L >= data["meta"]["legal_mm"]
        m0 = self._fit(data, model="m0")
        pick = lambda r: {"N": r["N"], "N_legal": r["N_legal"]}
        return {"tag_loss_unmerged": Trap(pick(self._fit(data, merge=False)), 1, bool(data["fish"]["previous_tag"].notna().any())),
                "distinct_count": Trap({"N": float(len(H)), "N_legal": float(legal.sum())}, 2),
                "lincoln_petersen": Trap({"N": lincoln_petersen(H), "N_legal": lincoln_petersen(H[legal])}, 3),
                "m0_equal_catchability": Trap(pick(m0), 4),
                "m0_by_length_class": Trap(self._length_classes(data), 5, False),
                "length_quadratic": Trap(pick(self._fit(data, model="quadratic")), None)}

    def _length_classes(self, data) -> dict:
        """M0 within each quintile of the captured lengths, summed (the legal total from each class's
        P(caught at least once)): a common way to handle size selectivity, but the smallest class
        still spans a steep part of the selectivity curve."""
        def run():
            H, L = merge_retagged(data["fish"])
            legal = L >= data["meta"]["legal_mm"]
            cls = np.clip(np.searchsorted(np.quantile(L, np.linspace(0, 1, 6)), L, side="right") - 1, 0, 4)
            n = {c: huggins(H[cls == c], np.ones(((cls == c).sum(), 1)), behaviour=False)[2]()[0] for c in range(5)}
            return {"N": float(sum(n.values())),
                    "N_legal": float(sum(legal[cls == c].sum() * n[c] / (cls == c).sum() for c in range(5)))}
        return self._cached(data, ("_classes",), run)

    def shortcuts(self, data) -> dict:
        """Estimators a policy might reach for without the covariate model: the distinct counts scaled
        by fixed factors, and Chao's lower bound."""
        H, L = merge_retagged(data["fish"])
        legal = L >= data["meta"]["legal_mm"]
        out = {"count_x1.6_x1.15": {"N": 1.6 * len(H), "N_legal": 1.15 * legal.sum()}}
        chao = lambda h: float(len(h) + np.sum(h.sum(1) == 1) ** 2 / (2 * max(np.sum(h.sum(1) == 2), 1)))
        out["chao_lower_bound"] = {"N": chao(H), "N_legal": chao(H[legal])}
        return out

    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        sp, K = m["species"], m["K"]
        head = (f"You are a fisheries biologist. The {sp[0]} ({sp[1]}) population of {m['lake']} was sampled with "
                f"{m['gear']} set on {K} consecutive nights, {m['dates'][0]} to {m['dates'][-1]} (nights 1-{K}). Every "
                f"{sp[0]} caught (all were at least {m['l_min']} mm long) was measured, given a numbered external tag "
                "and a PIT tag, and released. The survey is short enough that the population can be treated as closed: "
                "no recruitment, deaths or movement in or out, and no growth that matters.\n\n"
                "Data files:\n"
                f"- data_files/{HISTORIES}: one row per external tag. Columns: tag, length_mm (total length when the "
                "tag was applied), previous_tag (usually empty; filled when a fish that had lost its external tag was "
                "identified by its PIT tag and given a new one: the tag it had before), night_1 .. "
                f"night_{K} (1 if the fish was caught under this tag on that night, else 0).\n\n")
        q = (f"Question: how many {sp[0]} of at least {m['l_min']} mm were in the lake during the survey, and how many "
             f"of those were of legal size (at least {m['legal_mm']} mm)? Report the abundance N of all such fish and "
             "N_legal, the number of legal-size fish (every fish present, caught or not).\n\n")
        method = ("" if blind else
                  "Approach: merge the rows of re-tagged fish into one capture history per fish (follow previous_tag "
                  "back to the first tag; keep the first length). The nets catch large fish more easily, so fit Huggins' "
                  "conditional-likelihood closed-population model to all nights, with logit(capture probability) linear "
                  "in length; N and N_legal are Horvitz-Thompson sums, over the fish caught (all, and those of legal "
                  "length), of 1 / P(caught at least once).\n\n")
        return head + q + method + 'Return your answer as JSON: {"answer": {"N": <fish>, "N_legal": <fish>}}'

    def demo(self, cfg) -> list[Cell]:
        legal = int(re.search(r"legal size \(at least (\d+) mm\)", cfg["task"]).group(1))
        inspect = (f'import numpy as np, pandas as pd\nfish = pd.read_csv("data_files/{HISTORIES}")\n'
                   'nights = [c for c in fish.columns if c.startswith("night_")]\n'
                   'print(fish.head().to_string(index=False))\n'
                   'k = fish[nights].sum(axis=1)\n'
                   'print(len(fish), "tags;", fish.previous_tag.notna().sum(), "re-tagged;", len(nights), "nights;",\n'
                   '      "tags caught on 1, 2, ... nights:", np.bincount(k, minlength=len(nights) + 1)[1:].tolist())\n'
                   'by_length = k.groupby(pd.qcut(fish.length_mm, 5)).mean()\n'
                   'print("mean nights caught by length quintile:",\n'
                   '      {f"{i.left:.0f}-{i.right:.0f} mm": round(v, 2) for i, v in by_length.items()})')

        def prose2(outs):
            o = outs[0]
            rt = re.search(r"tags; (\d+) re-tagged", o)
            n_rt = int(rt.group(1)) if rt else 0
            tag = (f"{n_rt} fish lost an external tag and were re-tagged; I combine each new tag's history with the "
                   "fish's first tag, or it would be counted twice. " if n_rt else
                   "No fish was re-tagged, so tags and fish coincide. ")
            q = re.findall(r"'(\d+)-(\d+) mm': ([\d.]+)", o)
            size = (f"Fish in the smallest length quintile ({q[0][0]}-{q[0][1]} mm) were caught on {float(q[0][2]):.2f} "
                    f"nights on average and those in the largest ({q[-1][0]}-{q[-1][1]} mm) on {float(q[-1][2]):.2f}"
                    if len(q) >= 2 else "Large fish are caught on more nights than small ones")
            return (tag + size + ", so catchability depends on length "
                    "and the fish never caught are mostly small. The count of fish seen, a two-night Lincoln-Petersen "
                    "estimate, or M0 (one capture probability for every fish) would all miss them. I fit Huggins' "
                    "conditional likelihood with logit(capture probability) linear in length to all nights, and sum "
                    "1/P(caught at least once) over the fish caught, and over those of legal length. M0 is shown for "
                    "comparison.")
        fit_code = (f'H, L = merge_retagged(fish)\nlegal = L >= {legal}\n'
                    'X = np.column_stack([np.ones(len(H)), (L - 350) / 100])       # length in 100 mm, centred\n'
                    't0, _, total0 = huggins(H, X[:, :1], behaviour=False)\n'
                    't, cov, total = huggins(H, X, behaviour=False)\n'
                    '(N, se), (N_legal, se_legal) = total(), total(legal)\n'
                    'print(f"{len(H)} fish caught, {legal.sum()} of legal size; M0 would give N = {total0()[0]:.0f}")\n'
                    'print(f"length slope {t[1]:.3f} per 100 mm (SE {np.sqrt(cov[1, 1]):.3f})")\n'
                    'print(f"N = {N:.1f} +/- {se:.1f}; N_legal = {N_legal:.1f} +/- {se_legal:.1f}")\n'
                    'print("FINAL N = %.4f N_legal = %.4f" % (N, N_legal))')
        return [Cell("I load the capture histories and look at the design: how many tags, how many fish were "
                     "re-tagged, how often fish were caught, and whether that depends on length.", inspect),
                Cell(prose2, fit_code, uses=(merge_retagged, huggins),
                     preamble="from scipy.optimize import minimize\nfrom scipy.special import log_expit, logit\n"
                              "from statsmodels.tools.numdiff import approx_fprime, approx_hess")]


CRABS = (("Dungeness crabs", "Metacarcinus magister"), ("blue crabs", "Callinectes sapidus"),
         ("European green crabs", "Carcinus maenas"), ("red rock crabs", "Cancer productus"),
         ("American lobsters", "Homarus americanus"))


class TrapResponse(CaptureRecapture):
    name = "eco_mb"
    keys = ("N_female", "N_male", "trap_response")
    steps = {1: "merge the capture records of re-tagged animals (tag loss) into one history per animal",
             2: "estimate abundance, correcting for the animals never caught",
             3: "use every trapping day, not a two-sample Lincoln-Petersen estimate",
             4: "model individual heterogeneity in catchability (capture probability by sex)",
             5: "model the behavioural response after first capture (trap-happy or trap-shy)"}
    skills = {1: "qc", 2: "estimate", 3: "estimate", 4: "model-choice", 5: "model-choice"}
    seed_blocks = {"train": 15_100_000, "test": 15_150_000}
    tol_rule = ("N_female, N_male: max(2% of the estimate, 2 SE) with the Huggins delta-method SE of each sex's "
                "Horvitz-Thompson total; trap_response: max(5%, 2 SE) of the odds ratio, from the SE of the "
                "behavioural coefficient")
    note = ("ground truth is the reference analysis (re-tagged animals merged; Huggins conditional likelihood with a "
            "sex effect and a behavioural response; Horvitz-Thompson totals per sex) on the sampled captures, not the "
            "latent abundance; deviation is in units of tolerance, maxed over the answer keys")

    def draw(self, rng, k) -> Sample:
        K = int(rng.integers(4, 9))
        sp = CRABS[int(rng.integers(len(CRABS)))]
        N = int(round(np.exp(rng.uniform(np.log(2000), np.log(15000))) * k["n"]))
        male = rng.random(N) < rng.uniform(0.35, 0.65)
        a, b_sex = logit(rng.uniform(0.025, 0.18)), rng.uniform(0.5, 1.5)
        happy = rng.random() < 0.6
        b_beh = rng.uniform(0.7, 1.8) * (1 if happy else -1)
        eta = a + b_sex * male
        rows = simulate_session(rng, expit(eta), expit(eta + b_beh), K, _prep(rng, k, rng.random() < 0.5), "T",
                                int(rng.integers(1, 9)) * 1000)
        dates = _dates(rng, K, months=(6, 10))
        log = pd.DataFrame(rows, columns=["occasion", "tag", "previous_tag", "_animal"])
        log["sex"] = np.where(male[log["_animal"]], "M", "F")
        log["date"] = [dates[o - 1] for o in log["occasion"]]
        log = log.sort_values(["occasion", "tag"], kind="stable").reset_index(drop=True)
        out = log[["occasion", "date", "tag", "sex", "previous_tag"]].replace({"previous_tag": {"": np.nan}})
        latent = {"N_female": int((~male).sum()), "N_male": int(male.sum()), "trap_response": float(np.exp(b_beh)),
                  "N": N, "p_first_female": float(expit(a)), "p_first_male": float(expit(a + b_sex)),
                  "sex_log_odds": float(b_sex), "trap_happy": bool(happy), "retagged": int(out["previous_tag"].notna().sum())}
        where = f"a bay of {int(rng.integers(3, 30))} km2"
        return Sample({"log": out, "meta": {"K": K, "species": sp, "dates": dates, "where": where}}, latent)

    def write(self, data, d: Path) -> list[str]:
        data["log"].to_csv(d / CAPTURES, index=False)
        return [CAPTURES]

    def _design(self, data, merge=True):
        H = capture_histories(data["log"], data["meta"]["K"], merge)
        male = (data["log"].groupby("tag")["sex"].first().reindex(H.index) == "M").to_numpy(float)
        return H.to_numpy(), np.column_stack([np.ones(len(H)), male])

    def _fit(self, data, merge=True, sex=True, behaviour=True):
        def run():
            H, X = self._design(data, merge)
            t, cov, total = huggins(H, X if sex else X[:, :1], behaviour)
            m = X[:, 1] > 0
            (Nf, sef), (Nm, sem) = total(~m), total(m)
            q = 2 if sex else 1
            return {"N_female": Nf, "N_male": Nm, "trap_response": float(np.exp(t[q])) if behaviour else 1.0,
                    "_se_female": sef, "_se_male": sem, "_se_beh": float(np.sqrt(cov[q, q])) if behaviour else NAN,
                    "_S": len(H), "_S_male": int(m.sum())}
        return self._cached(data, ("_fit", merge, sex, behaviour), run)

    def reference(self, data) -> dict:
        return dict(self._fit(data))

    def tolerance(self, gold) -> dict:
        tr = gold["trap_response"]
        return {"N_female": max(0.02 * gold["N_female"], 2 * gold["_se_female"]),
                "N_male": max(0.02 * gold["N_male"], 2 * gold["_se_male"]),
                "trap_response": max(0.05 * tr, 2 * tr * gold["_se_beh"])}

    def sanity(self, gold, tol, latent) -> list[str]:
        ok = all(np.isfinite(gold[f"_se_{s}"]) and 0 < gold[f"_se_{s}"] < 0.1 * gold[f"N_{s}"] for s in ("female", "male"))
        return [] if ok and np.isfinite(gold["_se_beh"]) else ["N_not_identified"]

    def traps(self, data) -> dict:
        ref = self._fit(data)
        H, X = self._design(data)
        m = X[:, 1] > 0
        pick = lambda r: {k: r[k] for k in self.keys}
        return {"tag_loss_unmerged": Trap(pick(self._fit(data, merge=False)), 1, bool(data["log"]["previous_tag"].notna().any())),
                "distinct_count": Trap({"N_female": float((~m).sum()), "N_male": float(m.sum()),
                                        "trap_response": ref["trap_response"]}, 2),
                "lincoln_petersen": Trap({"N_female": lincoln_petersen(H[~m]), "N_male": lincoln_petersen(H[m]),
                                          "trap_response": ref["trap_response"]}, 3),
                "m0_equal_catchability": Trap(pick(self._fit(data, sex=False, behaviour=False)), 4),
                "ignore_sex": Trap(pick(self._fit(data, sex=False)), 4, False),
                "ignore_trap_response": Trap(pick(self._fit(data, behaviour=False)), 5)}

    def shortcuts(self, data) -> dict:
        """Data-light answers: each sex's distinct count scaled by a fixed factor, with the trap
        response read off the raw rates (recapture rate of marked animals over the first-day capture
        rate); and -- an upper bound on what scaling the counts can earn -- with the reference's
        trap response."""
        H, X = self._design(data)
        m = X[:, 1] > 0
        after = np.arange(H.shape[1]) > H.argmax(1)[:, None]
        c = H[after].mean() if after.any() else NAN
        n_hat = 2.2 * (~m).sum() + 1.4 * m.sum()
        odds = lambda x: x / (1 - x)
        raw = float(odds(c) / odds(H[:, 0].sum() / n_hat))
        counts = {"N_female": 2.2 * float((~m).sum()), "N_male": 1.4 * float(m.sum())}
        return {"counts_x2.2_x1.4_raw_response": {**counts, "trap_response": raw},
                "counts_x2.2_x1.4_true_response": {**counts, "trap_response": self._fit(data)["trap_response"]}}

    def task_text(self, data, blind: bool = False) -> str:
        m = data["meta"]
        sp, K = m["species"], m["K"]
        head = (f"You are a fisheries ecologist. {sp[0].capitalize()} ({sp[1]}) in {m['where']} were sampled with "
                f"baited pots hauled once a day on {K} consecutive days, {m['dates'][0]} to {m['dates'][-1]} (days "
                f"1-{K}). At its first capture each animal was sexed, measured, given a numbered external tag and a PIT "
                "tag, and released where it was caught. The survey is short enough that the population can be treated "
                "as closed: no recruitment, deaths, moulting or movement in or out.\n\n"
                "Data files:\n"
                f"- data_files/{CAPTURES}: one row per capture. Columns: occasion (day, 1..{K}), date, tag (external "
                "tag number), sex (M/F), previous_tag (usually empty; filled when an animal caught without its "
                "external tag was identified by its PIT tag and given a new one: the tag it had before).\n\n")
        q = ("Question: how many females and how many males were in the bay during the survey, and how did being "
             "caught change an animal's chance of being caught again? Report N_female and N_male (every animal present, "
             "caught or not) and the trap response: the odds ratio of capture on a day for an animal that has been "
             "caught before versus one that has not (above 1: trap-happy; below 1: trap-shy).\n\n")
        method = ("" if blind else
                  "Approach: build one capture history per animal, merging the records of re-tagged animals (follow "
                  "previous_tag back to the first tag). Capture probability differs between the sexes and changes after "
                  "an animal's first capture, so fit Huggins' conditional-likelihood model to all days, with logit("
                  "capture probability) = intercept + sex + a behavioural coefficient that applies after first capture. "
                  "Each sex's abundance is the Horvitz-Thompson sum over the animals of that sex caught of 1 / P(caught "
                  "at least once); the trap response is exp(behavioural coefficient).\n\n")
        return head + q + method + ('Return your answer as JSON: {"answer": {"N_female": <animals>, "N_male": <animals>, '
                                    '"trap_response": <odds ratio>}}')

    def demo(self, cfg) -> list[Cell]:
        K = int(re.search(r"on (\d+) consecutive days", cfg["task"]).group(1))
        inspect = (f'import numpy as np, pandas as pd\nlog = pd.read_csv("data_files/{CAPTURES}")\nK = {K}\n'
                   'print(log.head().to_string(index=False))\n'
                   'print(len(log), "captures of", log.tag.nunique(), "tags;", log.previous_tag.notna().sum(), "re-tagged")\n'
                   'first = log.groupby("tag").occasion.transform("min")\n'
                   'print("captures per day:", log.groupby("occasion").size().tolist())\n'
                   'print("first captures per day:", log[log.occasion == first].groupby("occasion").size()'
                   '.reindex(range(1, K + 1), fill_value=0).tolist())\n'
                   'per = log.groupby("tag").agg(sex=("sex", "first"), days=("occasion", "size"))\n'
                   'print("tags by sex:", per.sex.value_counts().to_dict(), "| mean days caught by sex:",\n'
                   '      per.groupby("sex").days.mean().round(2).to_dict())')

        def prose2(outs):
            o = outs[0]
            rt = re.search(r"tags; (\d+) re-tagged", o)
            n_rt = int(rt.group(1)) if rt else 0
            tag = (f"{n_rt} animals lost an external tag and were re-tagged; I merge each new tag into the animal's "
                   "first tag, or they would count as extra animals first caught late. " if n_rt else
                   "No animal was re-tagged, so tags and animals coincide. ")
            sx = re.search(r"mean days caught by sex: \{'F': ([\d.]+), 'M': ([\d.]+)\}", o)
            fc = re.search(r"first captures per day: \[(\d+),.*?(\d+)\]", o)
            sexes = (f"A female tag was caught on {float(sx.group(1)):.2f} days on average and a male tag on "
                     f"{float(sx.group(2)):.2f}, so capture probability depends on sex. " if sx else
                     "The sexes are caught at different rates, so capture probability depends on sex. ")
            first = (f"First captures fall from {fc.group(1)} on day 1 to {fc.group(2)} on the last day as the "
                     "unmarked animals are used up; " if fc else "")
            return (tag + sexes + first + "but pots can change behaviour (bait makes animals trap-happy, handling can "
                    "make them trap-shy), in which case recaptures say little about the chance of a first capture. "
                    "M0, a model without the sex effect or "
                    "without the behavioural term, or Lincoln-Petersen on two days would then misstate the abundances. "
                    "I fit Huggins' conditional likelihood, logit p = intercept + sex + a behavioural coefficient after "
                    "first capture, to all days: each sex's abundance is the Horvitz-Thompson sum of 1/P(caught at "
                    "least once) over the animals of that sex caught, and the trap response is exp(behavioural "
                    "coefficient). The model without the behavioural term is shown for comparison.")
        fit_code = ('H = capture_histories(log, K)\n'
                    'male = (log.groupby("tag").sex.first().reindex(H.index) == "M").to_numpy()\n'
                    'X = np.column_stack([np.ones(len(H)), male])\n'
                    't0, _, total0 = huggins(H, X, behaviour=False)\n'
                    't, cov, total = huggins(H, X)\n'
                    '(Nf, sef), (Nm, sem) = total(~male), total(male)\n'
                    'print(f"{len(H)} animals seen ({male.sum()} male); without the behavioural term: "\n'
                    '      f"females {total0(~male)[0]:.0f}, males {total0(male)[0]:.0f}")\n'
                    'print(f"sex log-odds {t[1]:.3f}; behavioural coefficient {t[2]:.3f} +/- {np.sqrt(cov[2, 2]):.3f}")\n'
                    'print(f"females {Nf:.1f} +/- {sef:.1f}; males {Nm:.1f} +/- {sem:.1f}")\n'
                    'print("FINAL N_female = %.4f N_male = %.4f trap_response = %.6f" % (Nf, Nm, np.exp(t[2])))')
        return [Cell("I load the capture log and look at its structure: captures and first captures per day, "
                     "re-tagged animals, and how often each sex is caught.", inspect),
                Cell(prose2, fit_code, uses=(capture_histories, huggins),
                     preamble="from scipy.optimize import minimize\nfrom scipy.special import log_expit, logit\n"
                              "from statsmodels.tools.numdiff import approx_fprime, approx_hess")]


FAMILIES = [SizeSelective(), TrapResponse()]
