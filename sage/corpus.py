"""SFT corpus assembly: executed demonstrations from several corpora, balanced by science.

Built for diversity-scaling experiments (SFT on k = 1, 3, 5, 7 sciences at a FIXED total
number of demonstrations, tested on held-out sciences), where the one thing that must not
vary with k is the budget, and every row must be traceable to its source.

  python -m sage corpus --from $D/sft/epi_rt $D/sft/epi_r0 $D/sft/causal5 $D/sft/astro3 \
      --total 600 --seed 0 [--variants 4] [--blind] --out OUT/k3_s0

Inputs. Each `--from` is a corpus prefix (PREFIX_train.jsonl and PREFIX_val.jsonl, either
optional) or a .jsonl file. Rows are the trace builders' format: {"id", "messages",
"_family", ...}, the user turn being harness/agents.build_prompt(task, listing). Append
"=SCIENCE" to a prefix to name the science of rows whose family is not a registered
plug-in (the legacy genomics corpora); `--science FAMILY=SCIENCE` does it per family.

Science = the registered plug-in's `domain`. A row's family is its `_family`, else the
longest registered family name its id starts with, else the prefix's own name.

Allocation (`--balance domain`, the default): `--total N` is split into equal shares per
science (the N mod k remainder goes to sciences picked by the seed); within a science,
equal shares per family (water-filling: a family with too few rows gives what it has and
the rest is spread over its siblings). If a science has fewer rows than its share,
`--short error` (default) stops and prints the largest feasible total; `fill` gives the
deficit to the other sciences (the total is kept, equality is not); `repeat` re-uses the
short science's rows (a fresh seeded pass each time; the manifest counts repeats).
`--balance family` splits equally over families, `none` samples the pool uniformly.

Sampling is a seeded permutation per family, independent of k and N: a smaller share takes
a PREFIX of the same permutation, so at one seed the rows a science contributes to a
k = 5 corpus are a subset of the rows it contributes to the k = 3 corpus. Rows are
de-duplicated by (family, id); an instance in both a train and a val input is kept in train
only. Output rows are shuffled by the seed.

Substitution (optional; the rest of the trace is never touched):
  --blind       the user turn's task text becomes the blind text (sage.variants.blind_from_text,
                the plug-in's rule; tested equal to task_text(data, blind=True)); families
                without blind support keep their text and are counted
  --variants K  K candidate texts per row (the original or blind text, then verified
                rewordings: an LLM cache from `sage reword`, then the generic rewriter), one
                chosen by the row's seed; every candidate passes sage.variants.check

Outputs: OUT_train.jsonl, OUT_val.jsonl and OUT_manifest.json, which lists for EVERY output
row its source file, source line, sha1 of the source line, id, family, science, repeat
index and the task text it carries (source label and sha1); plus the parameters, input
file hashes, per-science and per-family counts and warnings. Rows are copied byte for byte
unless their task text was substituted.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sage import variants as V

METHOD_REFS = ("Approach:", "the approach above", "as the approach", "the suggested approach", "the method menu",
               "example methods")


@dataclass
class Row:
    src: str
    line: int
    split: str
    raw: str
    id: str
    family: str | None
    science: str | None
    rep: int = 0
    info: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.family or "?", self.id)


# ----------------------------------------------------------------------------- inputs
def _files(spec: str) -> tuple[list[tuple[str, Path]], str | None]:
    """[(split, path)] for one --from entry, and its explicit science (after '=')."""
    sci = None
    if "=" in spec:
        spec, sci = spec.rsplit("=", 1)
    p = Path(spec)
    if p.suffix == ".jsonl":
        if not p.exists():
            raise SystemExit(f"--from {spec}: no such file")
        return [("val" if p.stem.endswith("_val") else "train", p)], sci
    out = [(s, Path(f"{spec}_{s}.jsonl")) for s in ("train", "val") if Path(f"{spec}_{s}.jsonl").exists()]
    if not out:
        raise SystemExit(f"--from {spec}: neither {spec}_train.jsonl nor {spec}_val.jsonl exists")
    return out, sci


def _registry():
    from sage import registry
    return registry.families()


def resolve_family(row: dict, fams: dict, prefix_name: str) -> str:
    """`_family`, else the longest registered family name the id starts with, else the
    input's own name (e.g. "hic" for hic_train.jsonl; then it is not a registered plug-in)."""
    f = row.get("_family")
    if f:
        return f
    rid = str(row.get("id", ""))
    best = max((n for n in fams if rid == n or rid.startswith(n + "_")), key=len, default=None)
    return best or re.sub(r"_(train|val)$", "", prefix_name)


def load(specs: list[str], science_of: dict[str, str] | None = None) -> tuple[list[Row], list[dict], list[str]]:
    """All rows of all inputs (train then val per input, file order), with family and
    science resolved; the input table for the manifest; warnings."""
    fams = _registry()
    science_of = dict(science_of or {})
    rows, inputs, warn = [], [], []
    for spec in specs:
        files, sci = _files(spec)
        for split, p in files:
            data = p.read_bytes()
            n = 0
            for i, line in enumerate(data.decode().splitlines()):
                if not line.strip():
                    continue
                r = json.loads(line)
                fam = resolve_family(r, fams, p.stem)
                science = sci or science_of.get(fam or "") or (fams[fam].domain if fam in fams else None)
                rows.append(Row(str(p), i, split, line, str(r.get("id", f"{p.stem}:{i}")), fam, science))
                n += 1
            inputs.append({"path": str(p), "split": split, "rows": n, "sha256": hashlib.sha256(data).hexdigest(),
                           "science_override": sci})
    return rows, inputs, warn


def dedupe(rows: list[Row]) -> tuple[list[Row], dict]:
    """First occurrence of each (family, id) per split; val copies of train instances dropped."""
    seen = {}
    out, dup, cross = [], Counter(), 0
    train_keys = {r.key for r in rows if r.split == "train"}
    for r in rows:
        k = (r.split, r.key)
        if k in seen:
            dup[r.split] += 1
            continue
        if r.split == "val" and r.key in train_keys:
            cross += 1
            continue
        seen[k] = r
        out.append(r)
    return out, {"duplicates_dropped": dict(dup), "val_rows_also_in_train_dropped": cross}


# ----------------------------------------------------------------------------- allocation
def _perm(n: int, *seed_parts) -> np.ndarray:
    return np.random.default_rng(V.stable_int(*seed_parts)).permutation(n)


def equal_split(total: int, names: list[str], seed, tag: str) -> dict[str, int]:
    """total split into equal integer shares; the remainder goes to names picked by the seed."""
    names = sorted(names)
    k = len(names)
    if not k:
        return {}
    base, rem = divmod(total, k)
    extra = {names[i] for i in _perm(k, seed, "remainder", tag)[:rem]}
    return {n: base + (n in extra) for n in names}


def water_fill(total: int, cap: dict[str, int], seed, tag: str) -> dict[str, int]:
    """Equal shares, capped by capacity; the deficit is spread over the uncapped names."""
    alloc = {n: 0 for n in cap}
    left, open_ = total, sorted(n for n in cap if cap[n] > 0)
    while left > 0 and open_:
        share = equal_split(left, open_, seed, f"{tag}:{left}")
        given = 0
        for n in open_:
            give = min(share[n], cap[n] - alloc[n])
            alloc[n] += give
            given += give
        left -= given
        open_ = [n for n in open_ if alloc[n] < cap[n]]
        if not given:
            break
    return alloc


def allocate(rows: list[Row], total: int | None, balance: str, within: str, short: str, seed) -> tuple[dict, dict]:
    """{(science, family): n} and a plan record. total None = every row."""
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[(r.science, r.family)].append(r)
    cap_f = {k: len(v) for k, v in by_fam.items()}
    sciences = sorted({s for s, _ in cap_f})
    cap_s = {s: sum(n for (ss, _), n in cap_f.items() if ss == s) for s in sciences}
    plan = {"available": {s: {"rows": cap_s[s], "families": {f: cap_f[(s, f)] for (ss, f) in sorted(cap_f,
            key=lambda x: str(x[1])) if ss == s}} for s in sciences}}
    if total is None:
        return {k: n for k, n in cap_f.items()}, {**plan, "rule": "all rows"}
    if balance == "none":
        pooled = sorted(cap_f)
        flat = [k for k in pooled for _ in range(cap_f[k])]
        if total > len(flat) and short == "error":
            raise SystemExit(f"--total {total} exceeds the {len(flat)} rows available")
        pick = Counter(flat[i] for i in _perm(len(flat), seed, "pool")[:total])
        return dict(pick), {**plan, "rule": "uniform over the pooled rows"}
    if balance == "family":
        groups = {k: cap_f[k] for k in cap_f}
        return _fit(groups, total, short, seed, "family", plan, label=lambda k: f"{k[0]}/{k[1]}")
    # balance == "domain": equal per science, then per family within it
    want = equal_split(total, sciences, seed, "science")
    shortfall = {s: (want[s], cap_s[s]) for s in sciences if cap_s[s] < want[s]}
    if shortfall and short == "error":
        k = len(sciences)
        feas = k * min(cap_s.values())
        lines = "\n".join(f"    {s:<18} needs {w:>6}, has {c:>6}" for s, (w, c) in sorted(shortfall.items()))
        raise SystemExit(f"--total {total} over {k} sciences needs {total // k}-{-(-total // k)} rows per science; "
                         f"short:\n{lines}\n  largest total with exactly equal shares: {feas} "
                         f"(or pass --short fill|repeat)")
    if short == "fill":
        want = water_fill(total, cap_s, seed, "science")
    alloc = {}
    for s in sciences:
        fams = {f: cap_f[(s, f)] for (ss, f) in cap_f if ss == s}
        n = want[s]
        if within == "pool":
            flat = [f for f in sorted(fams, key=str) for _ in range(fams[f])]
            reps = -(-n // len(flat)) if flat else 0
            order = np.concatenate([_perm(len(flat), seed, "pool", s, j) for j in range(max(reps, 1))])
            cnt = Counter(flat[i] for i in order[:n])
        else:
            cnt = water_fill(min(n, cap_s[s]), fams, seed, f"family:{s}")
            if n > cap_s[s]:                              # repeat: equal re-use across families
                extra = equal_split(n - cap_s[s], sorted(fams, key=str), seed, f"repeat:{s}")
                cnt = {f: cnt[f] + extra[f] for f in fams}
        for f, m in cnt.items():
            if m:
                alloc[(s, f)] = m
    plan.update({"rule": f"equal per science ({short} when short), then {within} within", "science_share": want,
                 "shortfall": {s: {"wanted": w, "available": c} for s, (w, c) in shortfall.items()}})
    return alloc, plan


def _fit(groups, total, short, seed, tag, plan, label):
    names = {label(k): k for k in groups}
    caps = {label(k): groups[k] for k in groups}
    want = equal_split(total, list(caps), seed, tag)
    short_ = {n: (want[n], caps[n]) for n in caps if caps[n] < want[n]}
    if short_ and short == "error":
        raise SystemExit(f"--total {total}: short groups {short_}; largest equal total {len(caps) * min(caps.values())}")
    if short == "fill":
        want = water_fill(total, caps, seed, tag)
    plan.update({"rule": f"equal per {tag} ({short} when short)", "share": want})
    return {names[n]: m for n, m in want.items() if m}, plan


def take(rows_of_family: list[Row], n: int, seed, science, family) -> list[Row]:
    """The first n rows of the family's seeded permutation; beyond its size, fresh passes
    (repeat index 1, 2, ...)."""
    out, m = [], len(rows_of_family)
    rep = 0
    while len(out) < n and m:
        order = _perm(m, seed, "rows", science, family, rep) if rep else _perm(m, seed, "rows", science, family)
        for i in order[: n - len(out)]:
            r = rows_of_family[i]
            out.append(Row(r.src, r.line, r.split, r.raw, r.id, r.family, r.science, rep))
        rep += 1
    return out


# ----------------------------------------------------------------------------- substitution
def substitute(row: Row, blind: bool, k: int, seed, cache: dict | None) -> str:
    """The output line for a row: the raw line, or the row with its task text replaced."""
    if not blind and k <= 1:
        row.info = {"task": "original", "variant": "none"}
        return row.raw
    obj = json.loads(row.raw)
    msgs = obj.get("messages") or []
    ui = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
    content = msgs[ui]["content"] if ui is not None else ""
    if V.PROMPT_SEP not in content:
        row.info = {"task": "original", "variant": "none", "note": "no task/listing boundary in the user turn"}
        return row.raw
    task, rest = content.split(V.PROMPT_SEP, 1)
    fams = _registry()
    fam = fams.get(row.family or "")
    base, label = task, "original"
    if blind:
        try:
            b = V.blind_from_text(fam, task) if fam is not None else None
        except ValueError as exc:                    # the plug-in's edit does not apply to this text
            b, row.info["warning"] = None, f"blind failed: {exc}"
        if b is None:
            label = "blind-unsupported" if "warning" not in row.info else "blind-failed"
        elif b == task:
            label = "blind-same"                       # the text names no method (e.g. enzyme_mm)
        else:
            base, label = b, "blind"
            refs = [m for m in METHOD_REFS for x in msgs[ui + 1:]
                    if x.get("role") == "assistant" and m.lower() in (x.get("content") or "").lower()]
            if refs:
                row.info["warning"] = f"assistant turns mention {sorted(set(refs))}"
    chosen, n_c = ("none", base), 1
    if k > 1:
        rs = V.stable_int(seed, "variant", row.family, row.id, row.split, row.rep)
        cands = V.candidates(base, k, rs, label="none", cache=cache,
                             answer_keys=fam.keys if fam is not None else ())
        chosen, n_c = cands[V.choose(len(cands), rs)], len(cands)
    row.info = {**row.info, "task": label, "variant": chosen[0], "task_sha1": V.sha1(chosen[1]),
                "candidates": n_c, "original_task_sha1": V.sha1(task)}
    if chosen[1] == task:
        return row.raw
    msgs[ui] = {**msgs[ui], "content": chosen[1] + V.PROMPT_SEP + rest}
    return json.dumps(obj)


# ----------------------------------------------------------------------------- build
def build(specs: list[str], out: str, total: int | None = None, val_total: int | None = None,
          balance: str = "domain", within: str = "family", short: str = "error", seed: int = 0,
          variants: int = 0, blind: bool = False, variant_cache=None, sciences: list[str] | None = None,
          families: list[str] | None = None, science_of: dict | None = None, dry: bool = False,
          argv: list[str] | None = None) -> dict:
    t0 = time.time()
    rows, inputs, warn = load(specs, science_of)
    unknown = Counter(Path(r.src).name for r in rows if r.science is None)
    if unknown:
        raise SystemExit("rows with no science (family not a registered plug-in): "
                         + ", ".join(f"{k} ({v})" for k, v in unknown.items())
                         + "\n  name it: --from PREFIX=SCIENCE, or --science FAMILY=SCIENCE")
    if sciences:
        rows = [r for r in rows if r.science in sciences]
        miss = sorted(set(sciences) - {r.science for r in rows})
        if miss:
            raise SystemExit(f"--sciences {miss}: no rows in the inputs")
    if families:
        rows = [r for r in rows if r.family in families]
    rows, dd = dedupe(rows)
    cache = V.load_cache(variant_cache) if variant_cache else None
    splits = {}
    for split, n in (("train", total), ("val", val_total)):
        pool = [r for r in rows if r.split == split]
        if not pool:
            splits[split] = ([], {"available": {}, "rule": "no input rows"})
            continue
        alloc, plan = allocate(pool, n, balance, within, short, [seed, split])
        by_fam = defaultdict(list)
        for r in pool:
            by_fam[(r.science, r.family)].append(r)
        picked = []
        for key in sorted(alloc, key=lambda x: (str(x[0]), str(x[1]))):
            picked += take(by_fam[key], alloc[key], [seed, split], *key)
        order = _perm(len(picked), seed, "order", split)
        splits[split] = ([picked[i] for i in order], plan)
    man = {"sage_corpus": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": argv or sys.argv,
           "params": {"from": specs, "total": total, "val_total": val_total, "balance": balance, "within": within,
                      "short": short, "seed": seed, "variants": variants, "blind": blind,
                      "variant_cache": [str(p) for p in (variant_cache or [])], "sciences": sciences,
                      "families": families, "science_of": science_of or {}},
           "inputs": inputs, "dedupe": dd, "warnings": list(warn), "splits": {}, "rows": []}
    for split, (picked, plan) in splits.items():
        lines = [substitute(r, blind, variants, [seed, split], cache) for r in picked] if not dry else []
        path = Path(f"{out}_{split}.jsonl")
        if not dry:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = "".join(l + "\n" for l in lines)
            path.write_text(data)
        per_s, per_f = Counter(r.science for r in picked), Counter(f"{r.science}/{r.family}" for r in picked)
        tasks = Counter(f"{r.info.get('task', '?')}+{r.info.get('variant', 'none').split(':')[0]}" for r in picked)
        man["splits"][split] = {"path": str(path), "rows": len(picked),
                                "sha256": (hashlib.sha256(data.encode()).hexdigest() if not dry else None),
                                "per_science": dict(sorted(per_s.items())), "per_family": dict(sorted(per_f.items())),
                                "repeated_rows": sum(r.rep > 0 for r in picked),
                                "task_text": dict(tasks), "plan": plan}
        for j, r in enumerate(picked):
            rec = {"split": split, "out_line": j, "src": r.src, "src_line": r.line,
                   "src_sha1": hashlib.sha1(r.raw.encode()).hexdigest(), "id": r.id, "family": r.family,
                   "science": r.science, "rep": r.rep}
            rec.update({k: v for k, v in r.info.items()})
            man["rows"].append(rec)
            if r.info.get("warning"):
                man["warnings"].append(f"{split}:{j} {r.id}: {r.info['warning']}")
    man["seconds"] = round(time.time() - t0, 2)
    if not dry:
        Path(f"{out}_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def summary(man: dict) -> str:
    L = [f"sage corpus  seed {man['params']['seed']}  total {man['params']['total']}  balance "
         f"{man['params']['balance']}/{man['params']['within']}  variants {man['params']['variants']}  "
         f"blind {man['params']['blind']}"]
    for split, s in man["splits"].items():
        L.append(f"  {split:<5} {s['rows']:>6} rows -> {s['path']}" + (f"  ({s['repeated_rows']} repeats)"
                                                                        if s["repeated_rows"] else ""))
        for sci, n in s["per_science"].items():
            fams = {f.split('/', 1)[1]: m for f, m in s["per_family"].items() if f.split('/', 1)[0] == str(sci)}
            L.append(f"        {str(sci):<18} {n:>6}   {fams}")
        if s["task_text"] and set(s["task_text"]) != {"original+none"}:
            L.append(f"        task texts (base+rewording): {dict(sorted(s['task_text'].items()))}")
    if man["dedupe"]["duplicates_dropped"] or man["dedupe"]["val_rows_also_in_train_dropped"]:
        L.append(f"  dedupe: {man['dedupe']}")
    for w in man["warnings"][:5]:
        L.append(f"  WARNING {w}")
    if len(man["warnings"]) > 5:
        L.append(f"  ... {len(man['warnings']) - 5} more warnings in the manifest")
    return "\n".join(L)
