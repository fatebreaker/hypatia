"""Mixed and leave-one-domain-out RL pools, balanced by family or by domain.

`mix` (sage/engine.py) pools every instance of one domain. For training across sciences the
pool must be balanced -- otherwise the science with the most instances dominates the
gradient -- and a held-out science must be provably absent. Both are here:

  python -m sage pool --domains causal,astro,epidemiology --balance domain --split train \
      --gen $DEEP_DATA_ROOT/generated --out ROOT/tri_pool [--total 1200] [--seed 0]
  python -m sage lodo --domains causal,astro,epidemiology --heldout astro --split train \
      --gen $DEEP_DATA_ROOT/generated --out ROOT/lodo_astro.json [--pool ROOT/lodo_astro_pool]

A pool is OUT/<split>/: symlinks to instance directories (<gen>/<family>/<split>/<id>/),
plus OUT/<split>/_pool.json naming every linked instance, its family, domain and source.
Stale symlinks from an earlier build are removed first (real directories never are), so the
directory always equals its manifest. Instances are picked by a seeded permutation per
family, the rule `sage corpus` uses, so smaller shares take prefixes of the same order.

Balance: `family` gives every family the same count, `domain` every domain the same count
(split equally over its families), `none` links everything (as `mix`). Without --total the
pool is the largest exactly balanced one (the smallest family or domain sets the share).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from sage.corpus import _perm, equal_split, water_fill


def instances(gen: Path, family: str, split: str) -> list[Path]:
    return sorted(d for d in (Path(gen) / family / split).glob("*") if (d / "eval_config.json").exists())


def families_of(domains: list[str] | None = None, families: list[str] | None = None) -> dict[str, str]:
    """{family: domain} for the requested domains and/or families (registered plug-ins)."""
    from sage import registry
    fams = registry.families()
    out = {}
    for d in domains or []:
        hit = {n: f.domain for n, f in fams.items() if f.domain == d}
        if not hit:
            raise SystemExit(f"no registered family in domain {d!r}; domains: {sorted({f.domain for f in fams.values()})}")
        out.update(hit)
    for n in families or []:
        out[n] = registry.get(n).domain
    return dict(sorted(out.items()))


def plan(avail: dict[str, int], domain_of: dict[str, str], balance: str, total: int | None, seed: int) -> dict[str, int]:
    """{family: n}: equal per family or per domain (water-filled within a domain)."""
    fams = [f for f in avail if avail[f] > 0]
    if balance == "none":
        return {f: avail[f] for f in fams}
    if balance == "family":
        per = min(avail[f] for f in fams) if total is None else None
        want = {f: per for f in fams} if per is not None else equal_split(total, fams, seed, "pool-family")
        short = {f: (want[f], avail[f]) for f in fams if avail[f] < want[f]}
        if short:
            raise SystemExit(f"--total {total}: families short {short}; largest balanced total "
                             f"{len(fams) * min(avail[f] for f in fams)}")
        return want
    doms = sorted({domain_of[f] for f in fams})
    cap = {d: sum(avail[f] for f in fams if domain_of[f] == d) for d in doms}
    per_dom = {d: min(cap.values()) for d in doms} if total is None else equal_split(total, doms, seed, "pool-domain")
    short = {d: (per_dom[d], cap[d]) for d in doms if cap[d] < per_dom[d]}
    if short:
        raise SystemExit(f"--total {total}: domains short {short}; largest balanced total {len(doms) * min(cap.values())}")
    out = {}
    for d in doms:
        out.update(water_fill(per_dom[d], {f: avail[f] for f in fams if domain_of[f] == d}, seed, f"pool:{d}"))
    return out


def pool(gen: Path, out: Path, split: str, domain_of: dict[str, str], balance: str = "domain",
         total: int | None = None, seed: int = 0, dry: bool = False) -> dict:
    """Build OUT/<split>/ as symlinks and write OUT/<split>/_pool.json."""
    gen, out = Path(gen), Path(out)
    src = {f: instances(gen, f, split) for f in domain_of}
    missing = [f for f, v in src.items() if not v]
    avail = {f: len(v) for f, v in src.items() if v}
    if not avail:
        raise SystemExit(f"no instances under {gen}/<family>/{split} for {sorted(domain_of)}")
    want = plan(avail, domain_of, balance, total, seed)
    picked = {f: [src[f][i] for i in sorted(_perm(len(src[f]), seed, "pool", f)[:n])] for f, n in want.items() if n}
    names = Counter(d.name for ds in picked.values() for d in ds)
    rows = [{"link": (d.name if names[d.name] == 1 else f"{f}__{d.name}"), "family": f, "domain": domain_of[f],
             "source": str(d.resolve())} for f, ds in picked.items() for d in ds]
    dest = out / split
    man = {"sage_pool": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "gen": str(gen), "split": split,
           "balance": balance, "total": total, "seed": seed, "n": len(rows),
           "per_family": {f: len(ds) for f, ds in picked.items()},
           "per_domain": dict(Counter(domain_of[f] for f, ds in picked.items() for _ in ds)),
           "available": avail, "missing_families": missing, "instances": rows}
    if dry:
        return man
    dest.mkdir(parents=True, exist_ok=True)
    for p in dest.iterdir():
        if p.is_symlink():
            p.unlink()
    for r in rows:
        (dest / r["link"]).symlink_to(r["source"], target_is_directory=True)
    (dest / "_pool.json").write_text(json.dumps(man, indent=1))
    return man


def lodo(domains: list[str], heldout: str, gen: Path | None = None, split: str = "train") -> dict:
    """Leave-one-domain-out: which families train and which are held out (never trained on)."""
    if heldout not in domains:
        domains = list(domains) + [heldout]
    dom = families_of(domains)
    train = sorted(f for f, d in dom.items() if d != heldout)
    held = sorted(f for f, d in dom.items() if d == heldout)
    man = {"sage_lodo": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "heldout_domain": heldout,
           "train_domains": sorted({dom[f] for f in train}), "train_families": train, "heldout_families": held,
           "split": split, "domain_of": dom,
           "rule": "held-out families are never trained on and no hyperparameter is tuned on them"}
    if gen is not None:
        man["gen"] = str(gen)
        man["available"] = {f: len(instances(gen, f, split)) for f in train + held}
        man["available_test"] = {f: len(instances(gen, f, "test")) for f in held}
    return man
