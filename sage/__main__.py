"""python -m sage {list,emit,mix,pool,lodo,real,traces,corpus,reword,check,skills,author}

  list     every registered family: domain, answer keys, steps, skills, loop policy, seed blocks
  emit     mint a split:        --family causal_iv|domain:astro|all --split train --n 400 --out ROOT
                                [--variants K] [--blind] [--skills] [--variant-cache C.jsonl]
  mix      pool a domain:       --domain causal --split train --out ROOT     (ROOT/causal_mix/train)
  pool     balanced mixed pool: --domains causal,astro --balance domain --split train --out ROOT/NAME [--gen SRC]
  lodo     leave-one-domain-out manifest:  --domains causal,astro,genomics --heldout astro --out FILE.json
  real     real test set:       --source kepler --families transit,eb,rot --out ROOT [--keep-controls] [--partial]
  traces   demonstrations:      --family enzyme_mm --gen ROOT --n 150 --out PREFIX
  corpus   SFT corpus:          --from PREFIX ... --total N --seed S [--variants K] [--blind] --out PREFIX
  reword   LLM rewordings (offline, a served model): --gen ROOT --family F | --corpus PREFIX  --k 8 --out C.jsonl
  check    invariants + shortcut audit:  --family F --n 50   |   --corpus DIR (read-only)
  skills   skill-coverage report per family and domain  [--family SPEC] [--corpus DIR ...]
  author   scaffold a plug-in:  --name N --spec spec.yaml --out sage/domains/N.py   |   --check N [--n 40]

`--out` is always explicit: the engine never writes into the shared data root by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sage  # noqa: F401  (BLAS pinning, repo root on sys.path)
from sage import registry


def _kv(items) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _csv(s):
    return [x for x in s.split(",") if x] if s else None


def _scratch(args, path) -> None:
    if getattr(args, "scratch", False):
        from sage.engine import refuse_data_root
        refuse_data_root(Path(path))


def _list(args) -> int:
    fams = registry.families()
    w = max([len("domain")] + [len(f.domain) for f in fams.values()]) + 2
    print(f"{'family':<20}{'domain':<{w}}{'keys':<32}{'steps':>5}  {'skills':<7}{'blind':<6}{'policy':<7}{'tries':>5}  "
          f"enforce       seed blocks")
    from sage.skills import skills_of
    from sage.variants import blind_supported
    for n in sorted(fams, key=lambda n: (fams[n].domain, n)):
        f = fams[n]
        tag = "  [demonstration]" if f.demonstration_only else ""
        sk = "yes" if skills_of(f) else "-"
        bl = "yes" if blind_supported(f) else "-"
        print(f"{n:<20}{f.domain:<{w}}{','.join(f.keys):<32}{len(f.steps):>5}  {sk:<7}{bl:<6}{f.rng_policy:<7}"
              f"{f.max_tries:>5}  {','.join(sorted(f.enforce)):<14}{f.seed_blocks}{tag}")
    for m, e in registry.errors().items():
        print(f"\nIMPORT ERROR {m}: {e.splitlines()[0]}")
    return 0


def _emit(args) -> int:
    from sage.engine import EmitOptions, emit
    _scratch(args, args.out)
    opts = EmitOptions(variants=args.variants, blind=args.blind, skills=args.skills,
                       variant_cache=tuple(args.variant_cache or ()))
    for name in registry.resolve(args.family):
        man = emit(registry.get(name), args.split, args.n, Path(args.out), args.seed0, args.workers,
                   args.prefix, args.allow_seed_overlap, opts if opts.active else None)
        print(json.dumps({k: v for k, v in man.items() if k not in ("weak_ids", "unenforced")}
                         | {"unenforced": {k: v["n"] for k, v in man["unenforced"].items()}}), flush=True)
    return 0


def _mix(args) -> int:
    from sage.engine import mix
    names = registry.resolve(f"domain:{args.domain}")
    pool = args.pool or f"{args.domain}_mix"
    print(json.dumps({pool: args.split, "families": names, "n": mix(names, args.split, Path(args.out), pool)}))
    return 0


def _pool(args) -> int:
    from sage.pools import families_of, pool
    _scratch(args, args.out)
    dom = families_of(_csv(args.domains), _csv(args.families))
    gen = Path(args.gen) if args.gen else Path(args.out).parent
    man = pool(gen, Path(args.out), args.split, dom, args.balance, args.total, args.seed, args.dry)
    print(json.dumps({k: v for k, v in man.items() if k != "instances"}, indent=1))
    if not args.dry:
        print(f"pool -> {Path(args.out) / args.split}  ({man['n']} links; manifest _pool.json)")
    return 0


def _lodo(args) -> int:
    from sage.pools import lodo, pool
    _scratch(args, args.out)
    man = lodo(_csv(args.domains), args.heldout, Path(args.gen) if args.gen else None, args.split)
    if args.pool:
        _scratch(args, args.pool)
        dom = {f: man["domain_of"][f] for f in man["train_families"]}
        p = pool(Path(args.gen), Path(args.pool), args.split, dom, args.balance, args.total, args.seed)
        man["train_pool"] = {"path": str(Path(args.pool) / args.split), "n": p["n"], "per_family": p["per_family"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return 0


def _real(args) -> int:
    from sage.real import Policy, build
    factory = registry.real_source(args.source)
    ctl = True if args.keep_controls else (False if args.no_controls else None)
    pol = Policy(keep_controls=ctl, require_variants_inside=args.variants_inside,
                 allow_partial=True if args.partial else None)
    for f in args.families.split(","):
        rep = build(factory, (f, Path(args.raw) if args.raw else None, args.limit), Path(args.out), pol,
                    args.workers, args.dry)
        print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1, default=str), flush=True)
    return 0


def _traces(args) -> int:
    from sage.traces import build
    rep = build(registry.get(args.family), Path(args.gen), args.n, args.val, args.out, args.split, args.workers)
    print(json.dumps({**rep, "discarded": len(rep["discarded"])}))
    return 0


def _corpus(args) -> int:
    from sage.corpus import build, summary
    _scratch(args, args.out)
    man = build(args.from_, args.out, args.total, args.val_total, args.balance, args.within, args.short, args.seed,
                args.variants, args.blind, args.variant_cache, _csv(args.sciences), _csv(args.families),
                _kv(args.science), args.dry, argv=sys.argv)
    print(summary(man), flush=True)
    if not args.dry:
        print(f"  manifest -> {args.out}_manifest.json")
    return 0


def _reword(args) -> int:
    from sage.reword import run
    _scratch(args, args.out)
    rep = run(args)
    print(json.dumps(rep, indent=1))
    return 0


def _check(args) -> int:
    from sage.audit import check_corpus, check_family, render
    reps = []
    if args.corpus:
        for c in args.corpus:
            reps += check_corpus(Path(c), args.limit)
    else:
        for name in registry.resolve(args.family):
            reps.append(check_family(registry.get(name), args.n, args.seed0, args.workers))
    for r in reps:
        print(render(r), flush=True)
    if args.json:
        from sage.engine import refuse_data_root
        refuse_data_root(Path(args.json))
        Path(args.json).write_text(json.dumps(reps, indent=1, default=str))
    return 0


def _skills(args) -> int:
    from sage.audit import skill_report
    print(skill_report(registry.resolve(args.family), args.corpus, args.limit))
    return 0


def _author(args) -> int:
    from sage import author as A
    if args.example:
        print(A.SPEC_EXAMPLE)
        return 0
    if args.check:
        ok, text, _ = A.check(args.check, args.n, args.seed0, args.workers)
        print(text)
        return 0 if ok else 1
    if not (args.name and args.spec):
        raise SystemExit("author needs --name and --spec (or --check NAME, or --example)")
    spec = A.load_spec(Path(args.spec))
    spec["name"] = args.name
    out = Path(args.out or f"sage/domains/{args.name}.py")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (pass --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(A.scaffold(spec))
    in_domains = out.parent.resolve() == (sage.ROOT / "sage" / "domains").resolve()
    target = f"sage.domains.{out.stem}" if in_domains else str(out)
    print(f"wrote {out}\n  next: fill in the TODOs (draw, write, analyse), then\n"
          f"    venv/bin/python -m sage author --check {target} --n 40\n"
          + (f"  and, when it passes, add \"sage.domains.{out.stem}\" to MODULES in sage/registry.py" if in_domains else
             f"  and, when it passes, move it to sage/domains/{out.stem}.py and add \"sage.domains.{out.stem}\" to "
             "MODULES in sage/registry.py"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=_list)

    e = sub.add_parser("emit")
    e.add_argument("--family", required=True, help="name, comma list, domain:<d>, or all")
    e.add_argument("--split", default="train")
    e.add_argument("--n", type=int, default=400)
    e.add_argument("--seed0", type=int, default=None, help="default: the family's block for --split")
    e.add_argument("--out", required=True, help="root; instances go to OUT/<family>/<split>/")
    e.add_argument("--workers", type=int, default=1, help="processes; the output does not depend on it")
    e.add_argument("--prefix", default=None, help="instance id prefix (default: the family name)")
    e.add_argument("--allow-seed-overlap", action="store_true",
                   help="reproduce a legacy corpus whose split seeds overlap another split's (leakage)")
    e.add_argument("--variants", type=int, default=0, help="K task texts per instance (task = one chosen by the seed; "
                   "the others in _provenance.task_variants)")
    e.add_argument("--blind", action="store_true", help="task = task_text(data, blind=True) where supported")
    e.add_argument("--skills", action="store_true", help="record _provenance.skills and trap_skills")
    e.add_argument("--variant-cache", nargs="*", default=None, help="LLM rewording caches from `sage reword`")
    e.add_argument("--scratch", action="store_true", help="refuse --out under the shared data root")
    e.set_defaults(fn=_emit)

    m = sub.add_parser("mix")
    m.add_argument("--domain", required=True)
    m.add_argument("--split", default="train")
    m.add_argument("--out", required=True)
    m.add_argument("--pool", default=None, help="default: <domain>_mix")
    m.set_defaults(fn=_mix)

    p = sub.add_parser("pool", help="a balanced mixed RL pool of symlinks")
    p.add_argument("--domains", default=None, help="comma list of domains")
    p.add_argument("--families", default=None, help="comma list of extra families")
    p.add_argument("--balance", choices=("family", "domain", "none"), default="domain")
    p.add_argument("--split", default="train")
    p.add_argument("--total", type=int, default=None, help="default: the largest exactly balanced pool")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen", default=None, help="root holding <family>/<split>/ (default: the parent of --out)")
    p.add_argument("--out", required=True, help="the pool: OUT/<split>/ symlinks + _pool.json")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--scratch", action="store_true")
    p.set_defaults(fn=_pool)

    lo = sub.add_parser("lodo", help="leave-one-domain-out manifest (and optionally its train pool)")
    lo.add_argument("--domains", required=True)
    lo.add_argument("--heldout", required=True)
    lo.add_argument("--split", default="train")
    lo.add_argument("--gen", default=None, help="count available instances under GEN/<family>/<split>")
    lo.add_argument("--out", required=True, help="the manifest (.json)")
    lo.add_argument("--pool", default=None, help="also build the train pool here (needs --gen)")
    lo.add_argument("--balance", choices=("family", "domain", "none"), default="domain")
    lo.add_argument("--total", type=int, default=None)
    lo.add_argument("--seed", type=int, default=0)
    lo.add_argument("--scratch", action="store_true")
    lo.set_defaults(fn=_lodo)

    r = sub.add_parser("real")
    r.add_argument("--source", required=True)
    r.add_argument("--families", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--raw", default=None, help="raw data root (default: the source's)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--keep-controls", action="store_true", help="keep reproduced targets with no live trap "
                   "(default: the source's policy; Kepler keeps them for rot only)")
    r.add_argument("--no-controls", action="store_true", help="strict only: drop targets with no live trap")
    r.add_argument("--variants-inside", action="store_true", help="drop targets where an accepted variant misses")
    r.add_argument("--partial", action="store_true", help="grade a target on the keys it publishes (default: the "
                   "source's partial_keys)")
    r.add_argument("--dry", action="store_true")
    r.set_defaults(fn=_real)

    t = sub.add_parser("traces")
    t.add_argument("--family", required=True)
    t.add_argument("--gen", required=True, help="generated root holding <family>/<split>/")
    t.add_argument("--split", default="train")
    t.add_argument("--n", type=int, default=150)
    t.add_argument("--val", type=int, default=10)
    t.add_argument("--out", required=True, help="writes OUT_train.jsonl and OUT_val.jsonl")
    t.add_argument("--workers", type=int, default=1)
    t.set_defaults(fn=_traces)

    co = sub.add_parser("corpus", help="assemble an SFT corpus from demonstration corpora, balanced by science")
    co.add_argument("--from", dest="from_", nargs="+", required=True,
                    help="corpus prefixes (PREFIX_train.jsonl / PREFIX_val.jsonl) or .jsonl files; PREFIX=SCIENCE "
                         "names the science of rows whose family is not a registered plug-in")
    co.add_argument("--out", required=True, help="writes OUT_train.jsonl, OUT_val.jsonl, OUT_manifest.json")
    co.add_argument("--total", type=int, default=None, help="train rows (default: every row)")
    co.add_argument("--val-total", type=int, default=None, help="val rows (default: every val row)")
    co.add_argument("--balance", choices=("domain", "family", "none"), default="domain",
                    help="equal shares per science (the plug-in's domain), per family, or none (uniform pool)")
    co.add_argument("--within", choices=("family", "pool"), default="family",
                    help="with --balance domain: equal per family inside a science, or uniform over its rows")
    co.add_argument("--short", choices=("error", "fill", "repeat"), default="error",
                    help="a science with fewer rows than its share: stop, give the deficit to the others, or re-use rows")
    co.add_argument("--seed", type=int, default=0)
    co.add_argument("--sciences", default=None, help="comma list: keep only these sciences")
    co.add_argument("--families", default=None, help="comma list: keep only these families")
    co.add_argument("--science", nargs="*", default=None, metavar="FAMILY=SCIENCE",
                    help="science of an unregistered family")
    co.add_argument("--variants", type=int, default=0, help="K task texts per row, one chosen by the row's seed")
    co.add_argument("--variant-cache", nargs="*", default=None, help="LLM rewording caches from `sage reword`")
    co.add_argument("--blind", action="store_true", help="substitute the blind task text where the family supports it")
    co.add_argument("--dry", action="store_true", help="plan and print only")
    co.add_argument("--scratch", action="store_true", help="refuse --out under the shared data root")
    co.set_defaults(fn=_corpus)

    rw = sub.add_parser("reword", help="offline LLM rewordings of task texts into a validated cache")
    rw.add_argument("--gen", default=None, help="emitted root: texts from GEN/<family>/<split>/*/eval_config.json")
    rw.add_argument("--family", default=None, help="with --gen: name, comma list, domain:<d>, or all")
    rw.add_argument("--split", default="train")
    rw.add_argument("--corpus", nargs="*", default=None, help="or: texts from the user turns of these corpora")
    rw.add_argument("--limit", type=int, default=0, help="at most this many distinct texts (0: all)")
    rw.add_argument("--blind", action="store_true", help="reword the blind version of each text")
    rw.add_argument("--k", type=int, default=8, help="verified rewordings wanted per text")
    rw.add_argument("--attempts", type=int, default=0, help="requests per text (default 2k)")
    rw.add_argument("--model", default=None, help="default: GBP_MODEL, else the server's first model")
    rw.add_argument("--base-url", default=None, help="default: GBP_BASE_URL (http://127.0.0.1:PORT/v1)")
    rw.add_argument("--temperature", type=float, default=0.9)
    rw.add_argument("--seed", type=int, default=0)
    rw.add_argument("--workers", type=int, default=4, help="concurrent requests")
    rw.add_argument("--out", required=True, help="the cache (.jsonl); appended to, texts already there are skipped")
    rw.add_argument("--scratch", action="store_true")
    rw.set_defaults(fn=_reword)

    c = sub.add_parser("check")
    c.add_argument("--family", default=None)
    c.add_argument("--n", type=int, default=30)
    c.add_argument("--seed0", type=int, default=None)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--corpus", nargs="*", default=None, help="audit emitted instances instead (read-only)")
    c.add_argument("--limit", type=int, default=0, help="per split, with --corpus")
    c.add_argument("--json", default=None, help="also write the reports as JSON")
    c.set_defaults(fn=_check)

    s = sub.add_parser("skills", help="skill coverage per family and domain")
    s.add_argument("--family", default="all")
    s.add_argument("--corpus", nargs="*", default=None, help="also count credit-bearing trap rows per skill here")
    s.add_argument("--limit", type=int, default=200)
    s.set_defaults(fn=_skills)

    a = sub.add_parser("author", help="scaffold a plug-in from a spec, or check one")
    a.add_argument("--name", default=None)
    a.add_argument("--spec", default=None, help="YAML or JSON (see --example)")
    a.add_argument("--out", default=None, help="default: sage/domains/NAME.py")
    a.add_argument("--force", action="store_true")
    a.add_argument("--check", default=None, metavar="NAME", help="a family, sage.domains.<module>, or a .py path")
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--seed0", type=int, default=None)
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--example", action="store_true", help="print an example spec")
    a.set_defaults(fn=_author)

    args = ap.parse_args(argv)
    if args.cmd == "check" and not (args.family or args.corpus):
        ap.error("check needs --family or --corpus")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
