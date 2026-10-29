"""`sage reword`: offline LLM rewordings of task texts, validated, into a cache.

Run on a pod with a served model (the OpenAI-compatible vLLM server harness/agents.py talks
to; GBP_BASE_URL=http://127.0.0.1:PORT/v1):

  python -m sage reword --gen ROOT --family domain:causal --split train --k 8 --out causal_rw.jsonl
  python -m sage reword --corpus $D/sft/epi_rt $D/sft/pk_oral --blind --k 8 --out blind_rw.jsonl

Each distinct task text (the plug-in's own text; with --blind its blind text) is sent until
k rewordings pass sage.variants.check -- every number, file name, identifier, requirement
and the answer format kept -- or 2k requests are spent. One cache row per text:
{"sha1", "text", "family", "variants", "rejected": [{"why", "text"}], "model", "requests"}.
`sage emit --variant-cache` and `sage corpus --variant-cache` read it (keyed by the sha1 of the
text they are rewording), so emission and corpus assembly stay deterministic and never
call a model themselves. The cache is appended to, and texts already in it are skipped, so
an interrupted run resumes.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sage import variants as V


def canonical_task(cfg: dict) -> str:
    """The plug-in's own text of an emitted instance, even if it shipped a variant as `task`."""
    prov = cfg.get("_provenance") or {}
    chosen = prov.get("task_variant_chosen")
    if chosen is None or chosen in ("canonical", "blind"):
        return cfg["task"]
    for s, t in zip(prov.get("task_variant_sources") or [], prov.get("task_variants") or []):
        if s in ("canonical", "blind"):
            return t
    return cfg["task"]


def texts_from_gen(gen: Path, names: list[str], split: str) -> list[tuple[str, str]]:
    out = []
    for n in names:
        for p in sorted((Path(gen) / n / split).glob("*/eval_config.json")):
            out.append((n, canonical_task(json.loads(p.read_text()))))
    return out


def texts_from_corpus(prefixes: list[str]) -> list[tuple[str, str]]:
    from sage.corpus import load
    rows, _, _ = load(prefixes)
    out = []
    for r in rows:
        msgs = json.loads(r.raw).get("messages") or []
        u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        if V.PROMPT_SEP in u:
            out.append((r.family, u.split(V.PROMPT_SEP, 1)[0]))
    return out


def run(args) -> dict:
    from sage import registry
    fams = registry.families()
    if args.gen:
        items = texts_from_gen(Path(args.gen), registry.resolve(args.family or "all"), args.split)
    elif args.corpus:
        items = texts_from_corpus(args.corpus)
    else:
        raise SystemExit("reword needs --gen ROOT --family F, or --corpus PREFIX ...")
    todo, skipped = {}, 0
    for fam_name, text in items:
        fam = fams.get(fam_name)
        if args.blind:
            b = V.blind_from_text(fam, text) if fam is not None else None
            if b is None:
                skipped += 1
                continue
            text = b
        todo.setdefault(V.sha1(text), (fam_name, text))
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["sha1"] for l in out.read_text().splitlines() if l.strip()}
    todo = {h: v for h, v in todo.items() if h not in done}
    if args.limit:
        todo = dict(list(todo.items())[: args.limit])
    rw = V.LLMRewriter(model=args.model, base_url=args.base_url, temperature=args.temperature)
    lock, kept, rejected = threading.Lock(), 0, 0
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(item):
        h, (fam_name, text) = item
        fam = fams.get(fam_name)
        return {**V.llm_variants(rw, text, args.k, seed=V.stable_int(args.seed, h) % (2 ** 31),
                                 attempts=args.attempts or None, answer_keys=fam.keys if fam is not None else ()),
                "family": fam_name, "blind": bool(args.blind)}
    with ThreadPoolExecutor(max(1, args.workers)) as ex, out.open("a") as fh:
        for fut in as_completed([ex.submit(one, it) for it in todo.items()]):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                kept += len(row["variants"])
                rejected += len(row["rejected"])
    return {"texts": len(todo), "already_cached": len(done), "skipped_blind_unsupported": skipped,
            "variants_kept": kept, "rejected": rejected, "model": rw.model, "cache": str(out)}
