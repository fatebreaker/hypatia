#!/usr/bin/env python
"""Tests for the generality features: variants, blind prompts, skills, corpus, pool/lodo, author.

Fast and self-contained: synthetic corpora built from fresh draws in a temporary directory
(never the shared data root), a stub in place of the LLM server, tiny emissions. Run:

  venv/bin/python -m sage.tests.test_generality            # all
  venv/bin/python -m sage.tests.test_generality corpus     # tests whose name contains "corpus"
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import threading
import traceback
import warnings
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

import sage  # noqa: F401
from sage import registry
from sage import variants as V

sys.path.insert(0, str(sage.ROOT / "harness"))                                # harness/agents.py
warnings.filterwarnings("ignore")
TMP = Path(tempfile.mkdtemp(prefix="sage_gen_test_"))
CHEAP = ("causal_did", "causal_iv", "epi_rt", "epi_r0", "pk_oral", "surv_cox", "eco_mb", "enzyme_mm")


def _draw(name: str, seed: int = 0):
    fam = registry.get(name)
    rng = np.random.default_rng(seed)
    return fam, fam.draw(rng, fam.knobs(rng)).data


def _listing(fam) -> str:
    return "  data_files/x.csv  (1 bytes)\n"


def _row(fam, data, i: int, extra_turns: int = 2) -> dict:
    from agents import SYSTEM, build_prompt                                  # harness/agents.py
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(fam.task_text(data), _listing(fam))}]
    for t in range(extra_turns):
        msgs += [{"role": "assistant", "content": f"step {t} of {fam.name} #{i}\n```python\nprint({t})\n```"},
                 {"role": "user", "content": f"[stdout]\n{t}\n"}]
    msgs.append({"role": "assistant", "content": '```json\n{"answer": {}}\n```'})
    return {"id": f"{fam.name}_{i:05d}", "messages": msgs, "reward": 1.0, "n_cells": extra_turns, "_family": fam.name}


def _corpus(prefix: Path, name: str, n_train: int, n_val: int, seed0: int = 0) -> None:
    """A synthetic trace corpus for a registered family: real task texts from fresh draws
    (a few distinct draws, reused), fake cells."""
    fam = registry.get(name)
    datas = [_draw(name, seed0 + j)[1] for j in range(3)]
    rows = [_row(fam, datas[i % 3], i) for i in range(n_train + n_val)]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:n_train]))
    Path(f"{prefix}_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[n_train:]))


_CORPORA: dict = {}


def corpora() -> dict:
    """{family: prefix} of the synthetic corpora (built once)."""
    if not _CORPORA:
        sizes = {"causal_did": 30, "causal_iv": 30, "epi_rt": 40, "epi_r0": 40, "pk_oral": 50, "surv_cox": 45,
                 "eco_mb": 12}
        for name, n in sizes.items():
            p = TMP / "sft" / name
            _corpus(p, name, n, 6)
            _CORPORA[name] = p
    return _CORPORA


# ============================================================================= variants
def test_validator_negatives():
    fam, data = _draw("pk_oral")
    t = fam.task_text(data)
    assert V.check(t, V.generic_variants(t, 1, 0)[0]) == []
    m = V._NUM.search(t)
    bad = {"number": t[:m.start(2)] + "999" + t[m.end(2):],
           "file": t.replace("data_files/concentrations.csv", "the file"),
           "column": t.replace("time_h", "time"),
           "answer": t.replace('"t_half"', '"half_life"'),
           "requirement": t.replace("Report the geometric mean across subjects of each subject's CL/F and of each "
                                    "subject's terminal half-life.", ""),
           "identical": t}
    for what, v in bad.items():
        assert V.check(t, v), f"validator accepted a {what} change"


def test_generic_variants_valid_everywhere():
    """Every generic rewording of every cheap family's text passes the validator."""
    for name in CHEAP:
        fam, data = _draw(name)
        for blind in (False, True):
            t, _ = V.task_text_of(fam, data, blind)
            vs = V.generic_variants(t, 6, 7)
            assert len(vs) >= 3, f"{name}: only {len(vs)} generic variants"
            for v in vs:
                assert V.check(t, v) == [], (name, V.check(t, v))
            assert vs == V.generic_variants(t, 6, 7), f"{name}: generic variants not deterministic"


def test_blind_from_text_matches_plugin():
    """For EVERY registered family with blind support, the text-level blind rule the corpus
    builder uses equals task_text(data, blind=True) on fresh draws."""
    heavy = {"rare_burden", "cis_mvmr", "scrna_ambient", "gwas_loci", "hic_loop_sage"}
    checked = []
    listed = {m for m in registry.MODULES}
    for name, fam in sorted(registry.families().items()):
        if type(fam).__module__ not in listed:
            continue                                  # registered in-process by another test (author --check)
        if not V.blind_supported(fam):
            assert V.blind_from_text(fam, "anything") is None
            continue
        if name in heavy:
            continue
        for seed in (0, 1):
            _, data = _draw(name, seed)
            full, blind = fam.task_text(data), fam.task_text(data, blind=True)
            assert V.blind_from_text(fam, full) == blind, f"{name}: text-level blind differs from the plug-in's"
        checked.append(name)
    assert {"causal_did", "astro_eb", "astro_rot", "astro_transit", "epi_rt"} <= set(checked), checked


def test_blind_drops_method_names():
    for name, words in {"causal_iv": ("Instrumental variable", "Difference-in-differences", "Matching"),
                        "astro_rot": ("more than one longitude",), "astro_eb": ("may be eccentric",),
                        "astro_transit": ("instrumental residuals",), "epi_rt": ("Approach:",)}.items():
        fam, data = _draw(name)
        b = fam.task_text(data, blind=True)
        assert not any(w in b for w in words), (name, [w for w in words if w in b])
        assert V.answer_template(b) == V.answer_template(fam.task_text(data)), name
        assert V.numbers(b) <= V.numbers(fam.task_text(data)) | {"1.0"}, name


def test_task_text_of_unsupported():
    class Old:
        def task_text(self, data):
            return "Question: x?\n\nReturn your answer as JSON: {\"answer\": {\"x\": <v>}}"
    t, kind = V.task_text_of(Old(), None, blind=True)
    assert kind == "blind-unsupported" and t.startswith("Question")


def test_plugin_phrasings_validated():
    fam, data = _draw("causal_att")
    for blind in (False, True):
        base, _ = V.task_text_of(fam, data, blind)
        c = V.candidates(base, 4, 3, fam=fam, data=data, blind=blind, answer_keys=fam.keys)
        assert len(c) == 4 and c[0][1] == base
        assert any(s.startswith("plugin") for s, _ in c), c
        for s, t in c[1:]:
            assert V.check(base, t, answer_keys=fam.keys) == [], (s, V.check(base, t))


class _Stub:
    """Transport stub: /models lists one model; /chat/completions returns canned rewordings."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, method, url, payload, headers, timeout):
        self.calls.append((method, url, payload))
        if url.endswith("/models"):
            return {"data": [{"id": "stub-model"}]}
        return {"choices": [{"message": {"content": self.replies.pop(0) if self.replies else ""}}]}


def test_llm_client_stub_and_validation():
    fam, data = _draw("epi_rt")
    t = fam.task_text(data)
    good = V.generic_variants(t, 2, 1)
    bad = t.replace("Question:", "Q:").replace(sorted(V.file_names(t))[0], "the cases file")
    stub = _Stub([f"<think>hm</think><task>\n{good[0]}\n</task>", f"<task>{bad}</task>", f"sure: <task>{good[1]}</task>"])
    rw = V.LLMRewriter(base_url="http://127.0.0.1:9/v1", transport=stub)
    assert rw.model == "stub-model"
    res = V.llm_variants(rw, t, k=2, seed=5, answer_keys=fam.keys)
    assert res["variants"] == good and len(res["rejected"]) == 1 and "files lost" in res["rejected"][0]["why"][0]
    posts = [p for m, u, p in stub.calls if m == "POST"]
    assert [p["seed"] for p in posts] == [5, 6, 7] and posts[0]["messages"][0]["content"] == V.REWORD_SYSTEM
    assert posts[0]["chat_template_kwargs"] == {"enable_thinking": False}
    cache = TMP / "cache.jsonl"
    cache.write_text(json.dumps({**res, "family": "epi_rt"}) + "\n")
    c = V.candidates(t, 3, 0, cache=V.load_cache(cache), answer_keys=fam.keys)
    assert [s for s, _ in c] == ["canonical", "llm:0", "llm:1"]


def test_llm_client_http_roundtrip():
    """The real urllib path against a local OpenAI-compatible stub server."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self._send({"data": [{"id": "local"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            self._send({"choices": [{"message": {"content": "<task>A reworded task.</task>"}}]})

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        rw = V.LLMRewriter(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k", timeout=10)
        assert rw.model == "local"
        assert rw.reword("Some task.", 1, seed=3) == ["A reworded task."]
        assert seen[0][0] == "/v1/chat/completions" and seen[0][1] == "Bearer k" and seen[0][2]["seed"] == 3
    finally:
        srv.shutdown()


def test_llm_client_refuses_openai_without_key():
    try:
        V.LLMRewriter(base_url="https://api.openai.com/v1", api_key="EMPTY", transport=_Stub([]))
    except RuntimeError:
        return
    raise AssertionError("called api.openai.com without a key")


def test_prompt_sep_matches_harness():
    from agents import build_prompt
    assert build_prompt("TASK", "LIST").startswith("TASK" + V.PROMPT_SEP)


# ============================================================================= corpus
def _build(tag: str, specs, **kw):
    from sage.corpus import build
    out = TMP / "out" / tag
    return build([str(s) for s in specs], str(out), argv=["test"], **kw), out


def _lines(p):
    return Path(p).read_text().splitlines()


def test_corpus_fixed_total_equal_shares():
    c = corpora()
    specs = [c[f] for f in ("causal_did", "causal_iv", "epi_rt", "epi_r0", "pk_oral", "surv_cox")]
    for k, sciences in ((1, ["survival"]), (2, ["causal", "survival"]), (4, ["causal", "epidemiology",
                                                                            "pharmacokinetics", "survival"])):
        man, out = _build(f"k{k}", specs, total=40, seed=3, sciences=sciences)
        tr = man["splits"]["train"]
        assert tr["rows"] == 40 == len(_lines(f"{out}_train.jsonl"))
        per = tr["per_science"]
        assert set(per) == set(sciences) and max(per.values()) - min(per.values()) <= 1, per
        # within a science, families split equally
        for s in sciences:
            fams = [n for f, n in tr["per_family"].items() if f.startswith(s + "/")]
            assert max(fams) - min(fams) <= 1, (s, fams)


def test_corpus_nested_across_k():
    """At one seed, the rows a science contributes shrink to a subset as k grows."""
    c = corpora()
    specs = [c[f] for f in ("causal_did", "causal_iv", "epi_rt", "epi_r0", "pk_oral", "surv_cox")]
    rows_at = {}
    for k, sciences in ((2, ["epidemiology", "survival"]), (4, ["causal", "epidemiology", "pharmacokinetics",
                                                               "survival"])):
        man, _ = _build(f"nest{k}", specs, total=40, seed=11, sciences=sciences)
        rows_at[k] = {(r["science"], r["src"], r["src_line"]) for r in man["rows"] if r["split"] == "train"}
    for s in ("epidemiology", "survival"):
        a = {x for x in rows_at[4] if x[0] == s}
        b = {x for x in rows_at[2] if x[0] == s}
        assert a and a < b, (s, len(a), len(b))


def test_corpus_manifest_is_exact_and_deterministic():
    c = corpora()
    specs = [c["epi_rt"], c["pk_oral"], c["causal_did"]]
    man, out = _build("m1", specs, total=30, val_total=6, seed=5)
    man2, out2 = _build("m2", specs, total=30, val_total=6, seed=5)
    for sp in ("train", "val"):
        assert Path(f"{out}_{sp}.jsonl").read_bytes() == Path(f"{out2}_{sp}.jsonl").read_bytes()
    assert [r for r in man["rows"]] == [r for r in man2["rows"]]
    srcs = {}
    for r in man["rows"]:
        src = srcs.setdefault(r["src"], _lines(r["src"]))
        line = src[r["src_line"]]
        assert hashlib.sha1(line.encode()).hexdigest() == r["src_sha1"]
        assert _lines(f"{out}_{r['split']}.jsonl")[r["out_line"]] == line        # no substitution: verbatim
    man3, _ = _build("m3", specs, total=30, val_total=6, seed=6)
    assert [r["src_line"] for r in man3["rows"]] != [r["src_line"] for r in man["rows"]], "seed ignored"


def test_corpus_short_policies():
    c = corpora()
    specs = [c["eco_mb"], c["epi_rt"], c["epi_r0"]]              # ecology has 12 train rows
    try:
        _build("short_err", specs, total=40, seed=0)
    except SystemExit as e:
        assert "largest total with exactly equal shares: 24" in str(e), str(e)
    else:
        raise AssertionError("--short error did not stop")
    man, _ = _build("short_fill", specs, total=40, seed=0, short="fill")
    assert man["splits"]["train"]["rows"] == 40 and man["splits"]["train"]["per_science"] == {"ecology": 12,
                                                                                               "epidemiology": 28}
    man, out = _build("short_rep", specs, total=40, seed=0, short="repeat")
    tr = man["splits"]["train"]
    assert tr["per_science"] == {"ecology": 20, "epidemiology": 20} and tr["repeated_rows"] == 8
    eco = [r for r in man["rows"] if r["science"] == "ecology" and r["split"] == "train"]
    assert len({(r["src"], r["src_line"]) for r in eco}) == 12


def test_corpus_dedupe_and_unknown_science():
    c = corpora()
    both = TMP / "sft" / "epi_both"
    Path(f"{both}_train.jsonl").write_text(Path(f"{c['epi_rt']}_train.jsonl").read_text()
                                           + Path(f"{c['epi_r0']}_train.jsonl").read_text())
    Path(f"{both}_val.jsonl").write_text(Path(f"{c['epi_rt']}_train.jsonl").read_text().splitlines()[0] + "\n")
    man, _ = _build("dedupe", [c["epi_rt"], both], seed=0)
    assert man["dedupe"]["duplicates_dropped"] == {"train": 40} and man["dedupe"]["val_rows_also_in_train_dropped"] == 1
    legacy = TMP / "sft" / "oldgen"
    rows = [json.loads(l) for l in _lines(f"{c['pk_oral']}_train.jsonl")[:5]]
    for r in rows:
        r.pop("_family")
        r["id"] = "old_" + r["id"]
    Path(f"{legacy}_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        _build("unk", [legacy], seed=0)
    except SystemExit as e:
        assert "PREFIX=SCIENCE" in str(e)
    else:
        raise AssertionError("an unknown science was not refused")
    man, _ = _build("unk_ok", [f"{legacy}=genomics"], seed=0, blind=True)
    assert man["splits"]["train"]["per_family"] == {"genomics/oldgen": 5}
    assert man["splits"]["train"]["task_text"] == {"blind-unsupported+none": 5}


def test_corpus_blind_and_variants_substitute_only_the_task():
    from agents import build_prompt
    c = corpora()
    specs = [c["causal_did"], c["epi_rt"], c["pk_oral"]]
    man, out = _build("bv", specs, total=30, seed=2, blind=True, variants=4)
    reg = registry.families()
    variant_kinds = Counter()
    for r in man["rows"]:
        src = json.loads(_lines(r["src"])[r["src_line"]])
        new = json.loads(_lines(f"{out}_{r['split']}.jsonl")[r["out_line"]])
        assert {k: v for k, v in src.items() if k != "messages"} == {k: v for k, v in new.items() if k != "messages"}
        assert len(src["messages"]) == len(new["messages"])
        for i, (a, b) in enumerate(zip(src["messages"], new["messages"])):
            if i != 1:
                assert a == b, "a non-task turn changed"
        old_task, listing = src["messages"][1]["content"].split(V.PROMPT_SEP, 1)
        new_task, listing2 = new["messages"][1]["content"].split(V.PROMPT_SEP, 1)
        assert listing == listing2
        fam = reg[r["family"]]
        base = V.blind_from_text(fam, old_task)
        assert r["task"] == "blind" and base != old_task
        assert new_task == base or V.check(base, new_task, answer_keys=fam.keys) == []
        assert V.sha1(new_task) == r["task_sha1"]
        assert new["messages"][1]["content"] == new_task + V.PROMPT_SEP + listing
        assert src["messages"][1]["content"] == build_prompt(old_task, listing[:-1])      # the harness's shape
        variant_kinds[r["variant"].split(":")[0]] += 1
    assert variant_kinds["none"] and variant_kinds["plugin"] + variant_kinds["generic"], variant_kinds


def test_corpus_production_smoke():
    """Read-only: the production per-family corpora, when present (skipped otherwise)."""
    try:
        from paths import data_root
        d = Path(data_root()) / "sft"
    except Exception:                                             # noqa: BLE001
        return
    names = [n for n in ("epi_rt", "pk_oral", "surv_cox", "astro_rot") if Path(f"{d / n}_train.jsonl").exists()]
    if len(names) < 2:
        return
    man, out = _build("prod", [d / n for n in names], total=8 * len(names), val_total=0, seed=0, blind=True,
                      variants=3)
    assert man["splits"]["train"]["rows"] == 8 * len(names)
    assert all(v == 8 for v in man["splits"]["train"]["per_science"].values())


# ============================================================================= emit options
def _tree(root: Path) -> dict:
    """Every instance file's bytes (the manifest records wall-clock timings, so it is left out)."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "_manifest.json"}


def test_emit_default_unchanged_and_options_keep_invariants():
    from sage.engine import EmitOptions, emit
    for name in ("causal_did", "enzyme_mm"):
        fam = registry.get(name)
        a, b, c = TMP / "e_none", TMP / "e_off", TMP / "e_on"
        emit(fam, "train", 2, a)
        emit(fam, "train", 2, b, opts=EmitOptions())                          # all off == legacy path
        assert _tree(a / name) == _tree(b / name), f"{name}: inactive options changed the output"
        emit(fam, "train", 2, c, workers=2, opts=EmitOptions(variants=4, blind=True, skills=True))
        for d in sorted((a / name / "train").glob(f"{name}_*")):
            x = json.loads((d / "eval_config.json").read_text())
            y = json.loads((c / name / "train" / d.name / "eval_config.json").read_text())
            for k in ("ground_truth", "grader", "data_files"):
                assert x[k] == y[k], (name, k)
            assert x["_provenance"]["ablations"] == y["_provenance"]["ablations"], name
            assert (d / "data_files").exists() and _tree(d / "data_files") == _tree(
                c / name / "train" / d.name / "data_files")
            p = y["_provenance"]
            assert p["task_mode"] == "blind" and len(p["task_variants"]) == 3 == len(p["task_variant_sources"])
            texts = [y["task"]] + p["task_variants"]
            blind_text = next(t for s, t in zip([p["task_variant_chosen"]] + p["task_variant_sources"], texts)
                              if s == "blind")
            assert "Approach:" not in blind_text and len(set(texts)) == 4
            for t in texts:
                assert t == blind_text or V.check(blind_text, t, answer_keys=fam.keys) == []
            assert p["skills"] == {str(s): v for s, v in fam.skills.items()}
            assert set(p["trap_skills"]) == set(x["_provenance"]["ablations"]) - {"reference"}
        man = json.loads((c / name / "train" / "_manifest.json").read_text())
        assert man["generality"]["task_mode"] == {"blind": 2}
        for _d in (a, b, c):                      # NFS leaves .nfs* files behind; the checks above are what matter
            shutil.rmtree(_d, ignore_errors=True)


def test_emit_variants_deterministic_across_workers():
    from sage.engine import EmitOptions, emit
    fam = registry.get("causal_iv")
    o = EmitOptions(variants=3, skills=True)
    emit(fam, "test", 3, TMP / "w1", workers=1, opts=o)
    emit(fam, "test", 3, TMP / "w3", workers=3, opts=o)
    assert _tree(TMP / "w1" / fam.name) == _tree(TMP / "w3" / fam.name)


def test_skills_record_idempotent_and_valid():
    from sage import skills as SK
    for name, fam in registry.families().items():
        assert SK.problems(fam) == [], (name, SK.problems(fam))
    fam = registry.get("epi_rt")
    cfg = {"_provenance": {"skills": {str(s): v for s, v in fam.skills.items()}, "ablations": {
        "reference": {"fails_at_step": None}, "no_truncation_correction": {"fails_at_step": 1},
        "no_weekday_effects": {"fails_at_step": None}}}}
    before = json.dumps(cfg["_provenance"]["skills"])
    SK.record(fam, cfg)
    SK.record(fam, cfg)
    assert json.dumps(cfg["_provenance"]["skills"]) == before and list(cfg["_provenance"]) == ["skills", "ablations",
                                                                                               "trap_skills"]
    assert cfg["_provenance"]["trap_skills"] == {"no_truncation_correction": "qc", "no_weekday_effects": None}
    cov = SK.coverage(registry.families().values())
    assert cov["families"]["causal_did"]["tagged"]


# ============================================================================= pools
def test_pool_balanced_and_lodo():
    from sage.engine import emit
    from sage.pools import families_of, lodo, pool
    gen = TMP / "gen"
    for name, n in (("causal_did", 4), ("causal_iv", 2), ("epi_rt", 3), ("enzyme_mm", 5)):
        emit(registry.get(name), "train", n, gen)
    dom = families_of(["causal", "epidemiology", "biochemistry"])
    man = pool(gen, gen / "p_dom", "train", dom, "domain")
    assert man["per_domain"] == {"causal": 3, "epidemiology": 3, "biochemistry": 3}
    links = sorted(p.name for p in (gen / "p_dom" / "train").iterdir() if p.is_symlink())
    assert links == sorted(r["link"] for r in man["instances"]) and len(links) == 9
    man = pool(gen, gen / "p_dom", "train", dom, "family", total=4)          # rebuild: stale links removed
    assert man["per_family"] == {"causal_did": 1, "causal_iv": 1, "enzyme_mm": 1, "epi_rt": 1}
    assert len([p for p in (gen / "p_dom" / "train").iterdir() if p.is_symlink()]) == 4
    try:
        pool(gen, gen / "p_x", "train", dom, "domain", total=30)
    except SystemExit as e:
        assert "largest balanced total 9" in str(e)
    else:
        raise AssertionError("an infeasible pool was built")
    lm = lodo(["causal", "epidemiology"], "biochemistry", gen)
    assert lm["heldout_families"] == ["enzyme_mm"] and "enzyme_mm" not in lm["train_families"]
    assert lm["available"]["causal_did"] == 4


# ============================================================================= author
def _load_path(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_author_scaffold_from_example():
    from sage import author as A
    from sage import skills as SK
    p = TMP / "decay.yaml"
    p.write_text(A.SPEC_EXAMPLE)
    spec = A.load_spec(p)
    assert A.spec_problems(spec) == []
    src = A.scaffold(spec)
    out = TMP / "decay_halflife.py"
    out.write_text(src)
    fam = _load_path(out).FAMILIES[0]
    assert fam.name == "decay_halflife" and fam.domain == "nuclear-physics" and fam.keys == ("half_life", "background")
    assert SK.problems(fam) == [] and fam.skills == {1: "normalise", 2: "model-choice"}
    data = {"meta": {"duration": 600}}
    t, b = fam.task_text(data), fam.task_text(data, blind=True)
    assert "Approach:" in t and "Approach:" not in b and "600" in t
    assert V.answer_keys_of(t) == ["half_life", "background"] and V.blind_from_text(fam, t) == b
    assert fam.tolerance({"half_life": 100.0, "background": 2.0}) == {"half_life": 5.0, "background": 0.5}
    ok, text, _ = A.check(str(out), n=2)
    assert not ok and "TODO draw" in text and "VERDICT: FAIL" in text


def test_author_spec_problems():
    from sage import author as A
    bad = {"name": "Bad-Name", "science": "x", "keys": ["a", "b"], "task": "t",
           "steps": {1: {"does": "d", "skill": "cleaning"}},
           "traps": [{"name": "t1", "step": 2}], "tolerance": {"a": "banana"}}
    probs = "\n".join(A.spec_problems(bad))
    for frag in ("lower_snake_case", "cleaning", "step 2 is not a declared step", "no rule for key 'b'",
                 "'banana' is not"):
        assert frag in probs, (frag, probs)


TOY = '''
import numpy as np, pandas as pd
from sage.core import Family, Sample, Trap

class ToyMean(Family):
    """Mean of a sample that sometimes carries gross outliers; the trap keeps them."""
    name = "toy_mean_check"
    domain = "toy"
    keys = ("mean",)
    steps = {1: "drop the gross outliers"}
    skills = {1: "qc"}
    seed_blocks = {"train": 99_000_000}
    max_tries = 1                       # no redraws: failures ship, so the check must report them
    def task_text(self, data, blind=False):
        return 'Question: the mean?\\n\\nReturn your answer as JSON: {"answer": {"mean": <value>}}'
    def draw(self, rng, k):
        x = rng.normal(10, 1, 200)
        if rng.random() < 0.5:
            x[:5] = 60.0                # outliers on half the draws
        return Sample({"x": x}, {"mean": 10.0})
    def write(self, data, d):
        pd.DataFrame({"x": data["x"]}).to_csv(d / "x.csv", index=False)
        return ["x.csv"]
    def reference(self, data):
        x = data["x"]
        return {"mean": float(x[np.abs(x - np.median(x)) < 10].mean())}
    def traps(self, data):
        return {"keep_outliers": Trap({"mean": float(data["x"].mean())}, 1)}
    def tolerance(self, gold):
        return {"mean": 0.5}

FAMILIES = [ToyMean()]
'''


def test_author_check_reports_failing_seeds():
    from sage import author as A
    p = TMP / "toy_mean_check.py"
    p.write_text(TOY)
    ok, text, det = A.check(str(p), n=10)
    assert not ok and "FAIL I2 fails on" in text and "keep_outliers (shipped)" in text and " tol" in text, text
    seeds = [s for s, _ in det["toy_mean_check"]["I2"]["keep_outliers"]]
    assert seeds and all(99_000_000 <= s < 99_000_010 for s in seeds)
    assert "strengthen()" in text                                    # the actionable hint


def test_author_check_pass_on_finished_family():
    from sage import author as A
    ok, text, _ = A.check("enzyme_mm", n=4, workers=2)
    assert ok and "VERDICT: PASS" in text, text


# ============================================================================= real, partial targets
def test_real_partial_targets():
    from sage.audit import reward_check
    from sage.core import RealSource, RealTarget
    from sage.real import Policy, assess_real, decide, write_real
    fam = registry.get("enzyme_mm")

    class Src(RealSource):
        label = "toy_real"
        note = "test"

        def __init__(self):
            self.family = fam

        def keys(self):
            return [0, 1]

        def load(self, key):
            rng = np.random.default_rng(key)
            data = fam.draw(rng, 1.0).data
            ref = fam.reference(data)
            d = TMP / f"raw{key}"
            d.mkdir(exist_ok=True)
            files = fam.write(data, d)
            return RealTarget(f"toy_{key}", data, {"vmax": ref["vmax"]}, {}, {f: d / f for f in files})
    src = Src()
    t = src.load(0)
    assert assess_real(src, t)["status"] == "no_published_value"             # default: all keys required
    r = assess_real(src, t, allow_partial=True)
    assert r["status"] == "assessed" and r["reproduced"] and r["graded_keys"] == ["vmax"]
    assert all(set(row) == {"vmax", "fails_at_step", "deviation"} for row in r["_table"].values())
    r["decision"] = decide(r, Policy(keep_controls=True), False)
    d = TMP / "real_out" / "toy_0"
    write_real(src, t, r, Policy(allow_partial=True), d)
    cfg = json.loads((d / "eval_config.json").read_text())
    assert cfg["ground_truth"] == {"vmax": r["_gold"]["vmax"]} and list(cfg["grader"]["config"]["keys"]) == ["vmax"]
    assert cfg["_provenance"]["ungraded_keys"] == ["km"] and cfg["_provenance"]["policy"]["allow_partial"] is True
    assert reward_check(cfg)["oracle_ok"]
    assert "allow_partial" not in Policy().as_json()                      # default files unchanged
    Src.partial_keys = True
    assert assess_real(src, t)["status"] == "assessed"


# ============================================================================= runner
def main(argv=None) -> int:
    sel = (argv if argv is not None else sys.argv[1:])
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)
             and (not sel or any(s in n for s in sel))]
    failed = []
    for n, f in tests:
        try:
            f()
            print(f"  ok    {n}", flush=True)
        except Exception:                                         # noqa: BLE001
            failed.append(n)
            print(f"  FAIL  {n}\n{traceback.format_exc()}", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed" + (f"; FAILED {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
