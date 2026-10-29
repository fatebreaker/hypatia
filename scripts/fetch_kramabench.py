"""Adapt KramaBench into harness-compatible problems.

KramaBench (Lai et al., 2025, arXiv:2506.06541; github.com/mitdbg/KramaBench) is 104 end-to-end
data-science pipelines over real data lakes in six domains. Each task is a natural-language question
over raw public files (CSV, XLSX, HTML, JSON, TXT) -- Massachusetts beach-water testing, FTC
Consumer Sentinel fraud reports, NOAA/NIFC wildfire statistics, archaeological radiocarbon and
climate records -- with a gold answer and a runnable reference solution. The best published system
reaches ~22 % end to end (o3, DS-GURU self-correcting).

WHY THIS BENCHMARK
------------------
It is the external test the project did not have: a different science and a different FORMAT from
everything the corpora train on. The data are messy multi-file lakes, not one tidy table per
question (headers inside the data, thousands separators, one CSV per state, HTML tables); answers
are strings and lists as well as numbers; and no step of the pipeline is named in the prompt.

WHICH TASKS (70 of 104)
-----------------------
Four of the six domains: archaeology, environment, legal, wildfire. Astronomy and biomedical are
left out because they overlap sciences the corpora train on (the astro families; genomics), and an
external test is only informative on a science the model has not seen. Within the four:
  * 9 tasks whose answer_type is string_approximate / list_approximate are dropped: upstream grades
    them with an LLM paraphrase judge, which is not exact grading;
  * 2 wildfire tasks read GeoPackage layers (.gpkg) and need geopandas, absent from the sandbox;
  * 2 wildfire tasks need files neither the GitHub repo nor its HF mirror ships
    (WeatherEvents_Jan2016-Dec2022.csv, ZHVI.csv).
Subtasks (upstream's step-level questions) are not included; the headline metric is end to end.

SETTING
-------
Upstream's "truth subset" (oracle-retrieval) setting: data_files/ holds the task's data_sources, at
their paths inside the lake. That hand-written list is sometimes short of what the question needs
(environment-easy-6 asks about 2002-2023 and omits 2023), so any lake file the reference solution
opens is added -- see trace_inputs(); the additions are listed per problem. The full-lake setting
adds a retrieval step this harness does not model. Reference solutions are NOT shipped -- they
would hand over the answer; verify() asserts no code reaches data_files/.

GOLD CHECK
----------
--reproduce replays every reference solution on the shipped data_files/ and stamps the result in
_provenance.gold_check (strict | in_output | inspected | not_in_output | solution_error). Some golds
are not what the upstream solution prints (archeology-hard-1: 8577.53 vs 8665.52); those stay in,
because the benchmark is scored on them, and benchmarks_ext/kramabench_verified links the rest so a
result can also be stated on golds this project has reproduced.

GRADING
-------
Ported from benchmark/metrics.py and fixtures/answer_type_fixtures.json at the pinned commit, per
answer_type: numeric_exact / string_exact by upstream `success`, list_exact by upstream `f1`,
numeric_approximate by upstream `rae_score` = 1/(1+RAE). `score` is the upstream per-task score
(the leaderboard averages it); `passed` is strict (success, F1 == 1, or RAE <= 1 % -- upstream
sets no pass mark for numeric_approximate). See GRADER below for the one deliberate deviation.

DISPERSION
----------
rl/dispersion.py scores with rl/reward.py, not this grader. StepReward handles scalar keys only, so
$DEEP_DATA_ROOT/generated/kramabench/test links the 63 scalar-answer problems (the 7 list answers
are left out) and each numeric key declares an `absolute_tolerance` approximating upstream's
success band. A reference_grader.py link beside them lets harness/runner.py run on that path too.

Run (a few minutes on a login node; upstream solutions execute in harness/sandbox.py):
  venv/bin/python scripts/fetch_kramabench.py                # fetch + build + replay golds + verify
  venv/bin/python scripts/fetch_kramabench.py --verify-only
  venv/bin/python scripts/fetch_kramabench.py --reproduce    # replay the reference solutions only
"""
from __future__ import annotations

import argparse
import ast
import collections
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, data_root  # noqa: E402

REPO = "https://github.com/mitdbg/KramaBench.git"
COMMIT = "b2e0d77540263f8b6119f977fe79a8c4386b5a02"      # main, 2026-07-24
HF_MIRROR = "https://huggingface.co/datasets/eugenie-y/KramaBench"
LICENCE = ("HF mirror card: 'cc' (Creative Commons, variant unspecified); the GitHub repo has no "
           "LICENSE file. Data are public-sector releases (MA DPH beach testing, FTC Consumer "
           "Sentinel, NOAA, NIFC, EPA AQI, US Census) and published archaeological datasets.")
DOMAINS = ("archeology", "environment", "legal", "wildfire")   # upstream spelling
DOMAIN_NAME = {"archeology": "archaeology", "environment": "environment",
               "legal": "legal", "wildfire": "wildfire"}
EXACT_TYPES = ("numeric_exact", "numeric_approximate", "string_exact", "list_exact")
SCALAR_TYPES = ("numeric_exact", "numeric_approximate", "string_exact")
APPROX_PASS_REL_TOL = 0.01
CACHE = data_root() / "cache" / "kramabench_repo"
DEFAULT_ROOT = data_root() / "benchmarks_ext" / "kramabench"
GENERATED = data_root() / "generated" / "kramabench" / "test"
BANNED_SUFFIXES = (".py", ".ipynb", ".r", ".rmd", ".sh")

GRADER = r'''"""KramaBench grading, ported in behaviour from the official benchmark/metrics.py and
benchmark/fixtures/answer_type_fixtures.json (github.com/mitdbg/KramaBench @ b2e0d77).

Per answer_type, as upstream:
  numeric_exact, string_exact  `success`: numbers equal; or, when either is a float, |p-t| < 1e-6,
                               relative error < 1e-6, or |p-t| < 1e-4 * max(|p|,|t|); two ints must
                               be equal; two strings equal after strip().lower()
  list_exact                   `f1` over elements (strings strip().lower(), numbers within 1e-6
                               relative); a string prediction is parsed as JSON, then as a Python
                               literal, then split on commas
  numeric_approximate          `rae_score` = 1 / (1 + |p-t|/|t|)

`score` is the upstream per-task score -- the leaderboard averages it, so MEAN SCORE is the number
comparable to published results. `passed` is strict: success == 1, F1 == 1, or, for
numeric_approximate (upstream sets no pass mark), relative error <= the key's `pass_rel_tol`.

Deliberately NOT "improved": a JSON boolean answer to a string gold such as "True" scores 0, as it
does upstream. The one deviation: upstream parses a string list prediction with eval(); this uses
ast.literal_eval, which gives the same result on every literal and never executes model output.

`absolute_tolerance` in a key's config is NOT read here. It exists for rl/reward.py (dispersion),
which grades numeric keys by an absolute band only.
"""
from __future__ import annotations

import ast
import json


def str_to_float(s: str) -> float:
    if s.endswith("%"):
        return float(s.strip("%")) / 100
    return float(s)


def _to_number(v):
    """Upstream try_convert_to_number (note: '50%' -> 50.0 here, unlike str_to_float)."""
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        try:
            c = v.strip()
            if c.endswith("%"):
                c = c[:-1].strip()
            return str_to_float(c)
        except (ValueError, TypeError):
            return v
    return v


def success(pred, target) -> float:
    if pred is None:
        return 0.0
    try:
        p, t = _to_number(pred), _to_number(target)
        if isinstance(p, (int, float)) and isinstance(t, (int, float)):
            if p == t:
                return 1.0
            if isinstance(t, float) or isinstance(p, float):
                if t == 0:
                    return float(abs(p) < 1e-6)
                rea = abs(p - t) / abs(t)
                diff = abs(p - t)
                big = max(abs(p), abs(t))
                return float(rea < 1e-6 or diff < 1e-6 or diff < big * 1e-4)
            return 0.0
        if isinstance(target, str) and isinstance(pred, str):
            return float(pred.strip().lower() == target.strip().lower())
        if isinstance(target, str) and isinstance(pred, (int, float)):
            return 0.0                      # upstream raises here, and scores 0
        return float(pred == target)
    except Exception:
        return 0.0


def f1(pred, target) -> float:
    try:
        if isinstance(pred, list) and isinstance(target, str):
            target = json.loads(target)
        if isinstance(pred, str) and isinstance(target, list):
            try:
                pred = json.loads(pred)
            except json.JSONDecodeError:
                try:
                    pred = ast.literal_eval(pred)
                except Exception:
                    pred = [x.strip() for x in pred.split(",") if x.strip()]
        if isinstance(pred, str) and isinstance(target, str):
            return float(pred.strip().lower() == target.strip().lower())
        if len(target) == 0:
            return float(len(pred) == 0)
        matched, recall_cnt = set(), 0
        for t in target:
            found = False
            for j, p in enumerate(pred):
                if isinstance(t, str):
                    if t.strip().lower() == str(p).strip().lower():
                        found = True
                        matched.add(j)
                        break
                elif isinstance(t, (float, int)):
                    if isinstance(p, (float, int)):
                        pp = p
                    elif isinstance(p, str):
                        try:
                            pp = str_to_float(p)
                        except Exception:
                            continue
                    else:
                        continue
                    if abs(pp - t) / max(abs(t), 1e-12) < 1e-6:
                        found = True
                        matched.add(j)
                        break
                elif p == t:
                    found = True
                    matched.add(j)
                    break
            if found:
                recall_cnt += 1
        recall = recall_cnt / len(target)
        precision = (len(matched) / len(pred)) if pred else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    except Exception:
        return 0.0


def rae_score(pred, target):
    """(upstream score, relative error or None when unscorable)."""
    try:
        if isinstance(pred, str):
            pred = str_to_float(pred)
        if isinstance(target, str):
            target = str_to_float(target)
        rae = abs(pred - target) / abs(target)
        return 1 / (1 + rae), rae
    except Exception:
        return 0.0, None


def grade_key(pred, target, spec: dict) -> tuple[float, bool]:
    at = spec["answer_type"]
    if at in ("numeric_exact", "string_exact"):
        s = success(pred, target)
        return s, s == 1.0
    if at == "list_exact":
        s = f1(pred, target)
        return s, s == 1.0
    if at == "numeric_approximate":
        s, rae = rae_score(pred, target)
        return s, rae is not None and rae <= spec.get("pass_rel_tol", 0.01)
    raise ValueError(f"answer_type {at!r} is not exactly gradable")


def evaluate(config: dict, submission: dict) -> dict:
    """Harness entry point -- same contract as the other vendored graders."""
    grader = config.get("grader") or {}
    if grader.get("type") != "kramabench_official":
        raise ValueError(f"unsupported grader type: {grader.get('type')}")
    gt = config.get("ground_truth") or {}
    keys = (grader.get("config") or {}).get("keys") or {}
    obj = submission or {}
    if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    if not isinstance(obj, dict):
        obj = {}
    details, total, npass = [], 0.0, 0
    for k, spec in keys.items():
        s, ok = grade_key(obj.get(k), gt.get(k), spec)
        total += s
        npass += bool(ok)
        details.append({"key": k, "answer_type": spec["answer_type"], "correct": bool(ok),
                        "score": s, "expected": gt.get(k), "got": obj.get(k)})
    n = max(len(keys), 1)
    return {"eval_id": config.get("id"), "grader_type": "kramabench_official",
            "score": total / n, "passed": npass == n, "correct": npass, "total": n,
            "details": details}


def grade(submission: dict, cfg: dict) -> dict:
    return evaluate(cfg, submission)
'''

TASK = """You are given raw data files from a real-world data lake in the {domain} domain.
Build the data pipeline the question needs -- load the files, clean and transform them, and
compute the final answer from the data.

Files in data_files/ ({n} file{s}):
  {files}

Question: {query}

The answer is {kind}. Return your answer as JSON: {{"answer": {{"{pid}": {slot}}}}}"""

KIND = {"numeric_exact": ("a number", "<number>"),
        "numeric_approximate": ("a number", "<number>"),
        "string_exact": ("a string", '"<string>"'),
        "list_exact": ("a list", "[<item>, ...]")}


# --------------------------------------------------------------------------- fetch
def fetch_repo() -> Path:
    """Sparse, blob-filtered clone of the four domains at the pinned commit (~70 MB)."""
    head = CACHE / ".git" / "HEAD"
    if head.exists():
        rev = subprocess.run(["git", "-C", str(CACHE), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        if rev == COMMIT and all((CACHE / "workload" / f"{d}.json").exists() for d in DOMAINS):
            return CACHE
    shutil.rmtree(CACHE, ignore_errors=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO, str(CACHE)],
                   check=True)
    subprocess.run(["git", "-C", str(CACHE), "sparse-checkout", "init", "--no-cone"], check=True)
    pats = [f"/data/{d}/" for d in DOMAINS] + [f"/solutions/{d}/" for d in DOMAINS] + \
           [f"/workload/{d}.json" for d in DOMAINS] + ["/benchmark/metrics.py", "/README.md"]
    (CACHE / ".git" / "info" / "sparse-checkout").write_text("\n".join(pats) + "\n")
    subprocess.run(["git", "-C", str(CACHE), "checkout", "-q", COMMIT], check=True)
    return CACHE


def lake_files(dom: str) -> list[str]:
    base = CACHE / "data" / dom / "input"
    return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
                  and p.name != ".DS_Store")


def resolve(dom: str, sources: list[str]) -> list[str] | None:
    """Upstream data_sources to lake-relative paths.

    They are written loosely: bare file names for files two directories deep, globs, and
    'State MSA Identity Theft Data/*' for a directory spelled '... data'. Match exactly, then by
    case-insensitive path suffix, then as a case-insensitive glob. None if any source is missing.
    """
    files = lake_files(dom)
    out: list[str] = []
    for s in sources:
        lo = s.lower()
        hit = [f for f in files if f == s] \
            or [f for f in files if f.lower() == lo or f.lower().endswith("/" + lo)] \
            or [f for f in files if fnmatch.fnmatch(f.lower(), lo) or fnmatch.fnmatch(f.lower(), "*/" + lo)]
        if not hit:
            return None
        out.extend(h for h in hit if h not in out)
    return out


def drop_reason(dom: str, t: dict, traced: list[str] | None = None) -> str | None:
    if t["answer_type"] not in EXACT_TYPES:
        return f"answer_type {t['answer_type']} is graded upstream by an LLM judge"
    files = resolve(dom, t["data_sources"])
    if files is None:
        return "a data source is not distributed with the benchmark"
    if any(f.lower().endswith(".gpkg") for f in files + (traced or [])):
        return "reads GeoPackage layers; needs geopandas, which the sandbox lacks"
    return None


def trace_inputs(root: Path) -> dict[str, list[str]]:
    """Lake files each reference solution actually opens, with the whole domain lake available.

    Upstream's data_sources is hand-written and sometimes short: environment-easy-6 asks about
    2002-2023 and omits the 2023 file; environment-hard-17 must compare every summer's rainfall but
    lists one year. Shipping data_sources alone would make such a task unanswerable. So every
    candidate's reference solution is run once in the sandbox against the full lake, a Python audit
    hook records the files it opens, and data_files = data_sources + those files. Cached, per commit,
    in solution_inputs.json; the additions are recorded per problem in _provenance.
    """
    cache = root / "solution_inputs.json"
    if cache.exists():
        got = json.loads(cache.read_text())
        if got.get("commit") == COMMIT:
            return got["inputs"]
    sys.path.insert(0, str(PROJECT_ROOT / "harness"))
    from sandbox import Sandbox  # noqa: E402
    out: dict[str, list[str]] = {}
    for dom in DOMAINS:
        lake = CACHE / "data" / dom / "input"
        for t in json.loads((CACHE / "workload" / f"{dom}.json").read_text()):
            if drop_reason(dom, t):
                continue
            src = (CACHE / "solutions" / dom / f"{t['id']}.py").read_text()
            sb = Sandbox(data_dir=lake, timeout=300, link_data=True)
            try:
                res = sb.run(replay_source(dom, src, trace=True))
            finally:
                sb.cleanup()
            line = next((ln for ln in res.stdout.splitlines()[::-1] if ln.startswith("__OPENED__")), None)
            opened = ast.literal_eval(line[len("__OPENED__"):]) if line else []
            # An open that FAILED is audited too (wildfire-hard-17 tries an undistributed
            # noaa_wildfires_sylvia.csv), so keep only files the lake really has.
            out[t["id"]] = sorted({os.path.normpath(p) for p in opened
                                   if not p.lower().endswith(BANNED_SUFFIXES)
                                   and (lake / os.path.normpath(p)).is_file()})
            print(f"  traced {t['id']:<22} opens {len(out[t['id']])} file(s)", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"commit": COMMIT, "inputs": out}, indent=1))
    return out


def key_spec(t: dict) -> dict:
    at, a = t["answer_type"], t["answer"]
    spec: dict = {"answer_type": at,
                  "metric": {"numeric_exact": "success", "string_exact": "success",
                             "list_exact": "f1", "numeric_approximate": "rae_score"}[at]}
    if at == "numeric_exact":
        # rl/reward.py only: an absolute band approximating upstream `success`.
        spec["absolute_tolerance"] = max(1e-4 * abs(float(a)), 1e-6)
    elif at == "numeric_approximate":
        spec["pass_rel_tol"] = APPROX_PASS_REL_TOL
        spec["absolute_tolerance"] = max(APPROX_PASS_REL_TOL * abs(float(a)), 1e-6)
    return spec


def gold(t: dict):
    a = t["answer"]
    if t["answer_type"].startswith("numeric") and isinstance(a, str):
        f = float(a)
        return int(f) if f.is_integer() and "." not in a else f
    return a


def link_readonly(src: Path, dst: Path) -> None:
    """Hard-link (else copy) a lake file into data_files/, read-only.

    The 18 environment problems share ~30 MB of yearly CSVs; links cost nothing. Read-only because
    a shared inode is exactly what scripts/protect_benchmark_data.sh guards: under
    Sandbox(link_data=True) a cell writing to data_files/ would otherwise corrupt every problem.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode & ~0o222)


def build(root: Path) -> dict:
    fetch_repo()
    traced = trace_inputs(root)
    (root / "problems").mkdir(parents=True, exist_ok=True)
    for p in (root / "problems").iterdir():     # stale builds only -- never another benchmark's
        c = p / "eval_config.json"
        if c.exists() and json.loads(c.read_text())["_provenance"].get("benchmark") == "KramaBench":
            shutil.rmtree(p)
    (root / "reference_grader.py").write_text(GRADER)
    kept, dropped = [], []
    for dom in DOMAINS:
        for t in json.loads((CACHE / "workload" / f"{dom}.json").read_text()):
            why = drop_reason(dom, t, traced.get(t["id"]))
            if why:
                dropped.append({"id": t["id"], "answer_type": t["answer_type"], "reason": why})
                continue
            pid = t["id"]
            listed = resolve(dom, t["data_sources"])
            added = [f for f in traced.get(pid, []) if f not in listed
                     and (CACHE / "data" / dom / "input" / f).is_file()]
            files = sorted(listed + added)
            pdir = root / "problems" / pid
            ddir = pdir / "data_files"
            for f in files:
                link_readonly(CACHE / "data" / dom / "input" / f, ddir / f)
            kind, slot = KIND[t["answer_type"]]
            task = TASK.format(domain=DOMAIN_NAME[dom], n=len(files), s="" if len(files) == 1 else "s",
                               files="\n  ".join(files), query=t["query"].strip(), kind=kind,
                               pid=pid, slot=slot)
            cfg = {
                "id": pid,
                "task": task,
                "data_files": files,
                "ground_truth": {pid: gold(t)},
                "grader": {"type": "kramabench_official", "config": {"keys": {pid: key_spec(t)}}},
                "_provenance": {
                    "benchmark": "KramaBench",
                    "paper": "arXiv:2506.06541",
                    "source": REPO, "version": COMMIT, "mirror": HF_MIRROR, "licence": LICENCE,
                    "domain": DOMAIN_NAME[dom],
                    "difficulty": "hard" if "-hard-" in pid else "easy",
                    "answer_type": t["answer_type"],
                    "upstream_data_sources": t["data_sources"],
                    "data_files_added": added,
                    "setting": ("truth subset (oracle retrieval): data_files = the task's data_sources "
                                "plus any lake file its reference solution opens"),
                    "reference_solution": f"solutions/{dom}/{pid}.py (upstream; not shipped)",
                    "n_upstream_subtasks": len(t.get("subtasks") or []),
                },
            }
            (pdir / "eval_config.json").write_text(json.dumps(cfg, indent=1))
            kept.append(pid)
    manifest = {"benchmark": "KramaBench", "source": REPO, "version": COMMIT,
                "domains": [DOMAIN_NAME[d] for d in DOMAINS], "n_kept": len(kept),
                "kept": kept, "dropped": dropped}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    link_generated(root)
    return manifest


def link_generated(root: Path) -> int:
    """generated/kramabench/test: symlinks to the scalar-answer problems, for rl/dispersion.py."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    for p in GENERATED.iterdir():
        if p.is_symlink():
            p.unlink()
    n = 0
    for p in sorted((root / "problems").iterdir()):
        cfg = json.loads((p / "eval_config.json").read_text())
        if cfg["_provenance"]["answer_type"] in SCALAR_TYPES:
            (GENERATED / p.name).symlink_to(p)
            n += 1
    (GENERATED / "reference_grader.py").symlink_to(root / "reference_grader.py")
    return n


# --------------------------------------------------------------------------- verify
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selftest(g) -> None:
    """The ported grader against upstream behaviour, case by case."""
    S, F, R = g.success, g.f1, g.rae_score
    assert S(268, 268) == 1 and S(267, 268) == 0                    # two ints: exact
    assert S(268.0, 268) == 1 and S("268", 268) == 1                # float / string forms
    assert S(12964.87, 12964.8727) == 1                             # within 1e-4 relative
    assert S(12960.0, 12964.8727) == 0
    assert S(7805400, 7805421) == 0 and S(7805400.0, 7805421) == 1  # upstream int/float asymmetry
    assert S("  São Paulo ", "são paulo") == 1 and S("Sao Paulo", "São Paulo") == 0
    assert S(True, "True") == 0 and S("true", "True") == 1          # JSON bool vs string gold
    assert S(None, 3) == 0 and S("4.796%", 4.796) == 1              # '%' stripped, not /100
    assert S(0.0000005, 0) == 1 and S(0.001, 0) == 0
    assert F(["Nevada", "california"], ["California", "Nevada"]) == 1.0
    assert abs(F(["California"], ["California", "Nevada"]) - 2 / 3) < 1e-12
    assert F("[2003, 2011]", [2003, 2011]) == 1.0 and F("California, Nevada", ["California", "Nevada"]) == 1.0
    assert F("__import__('os').getcwd()", ["x"]) == 0.0              # no eval of model output
    assert F(["6.326", 0.787], [6.326, 0.787]) == 1.0 and F([6.33, 0.787], [6.326, 0.787]) < 1
    assert F(None, [1]) == 0.0 and F([], []) == 1.0
    s, rae = R(0.524, 0.519)
    assert abs(s - 1 / (1 + 0.005 / 0.519)) < 1e-12 and rae < 0.01
    assert R("x", 1.0) == (0.0, None) and R(None, 1.0) == (0.0, None)
    cfg = {"id": "t", "ground_truth": {"t": 0.519},
           "grader": {"type": "kramabench_official",
                      "config": {"keys": {"t": {"answer_type": "numeric_approximate", "pass_rel_tol": 0.01}}}}}
    assert g.evaluate(cfg, {"answer": {"t": 0.52}})["passed"]
    assert not g.evaluate(cfg, {"answer": {"t": 0.53}})["passed"]
    assert not g.evaluate(cfg, {"answer": {}})["passed"] and not g.evaluate(cfg, None)["passed"]


def verify(root: Path) -> dict:
    probs = sorted(p for p in (root / "problems").iterdir() if (p / "eval_config.json").exists())
    bad = [f for p in probs for f in (p / "data_files").rglob("*")
           if f.is_file() and (f.suffix.lower() in BANNED_SUFFIXES or "solution" in f.name.lower())]
    assert not bad, f"ANSWER-LEAKAGE RISK: code reached data_files/: {bad[:5]}"

    # Load every problem exactly as harness/runner.py does.
    sys.path.insert(0, str(PROJECT_ROOT / "harness"))
    import runner  # noqa: E402
    assert [p.name for p in runner.problems(root)] == [p.name for p in probs]
    g = runner.load_grader(probs[0])
    assert g.__file__ == str(root / "reference_grader.py"), g.__file__
    selftest(g)

    from rl.dispersion import instance_tol  # noqa: E402
    from rl.reward import StepReward  # noqa: E402

    cfgs = [json.loads((p / "eval_config.json").read_text()) for p in probs]
    for p, cfg in zip(probs, cfgs):
        assert cfg["id"] == p.name and set(cfg["ground_truth"]) == set(cfg["grader"]["config"]["keys"])
        for f in cfg["data_files"]:
            assert (p / "data_files" / f).is_file(), (p.name, f)
        assert g.evaluate(cfg, {"answer": cfg["ground_truth"]})["passed"], p.name
        null = {k: "" if isinstance(v, str) else 0 for k, v in cfg["ground_truth"].items()}
        assert not g.evaluate(cfg, {"answer": null})["passed"], p.name

    # The dispersion view: every linked problem, scored by rl/reward.py as dispersion.py would.
    linked = sorted(p for p in GENERATED.iterdir() if (p / "eval_config.json").exists())
    assert runner.load_grader(linked[0]).__file__.endswith("reference_grader.py")
    for p in linked:
        cfg = json.loads((p / "eval_config.json").read_text())
        sr = StepReward(tolerance=instance_tol(cfg, 0.025))
        r = sr.score(cfg, {"answer": cfg["ground_truth"]})
        assert r.passed and r.reward == 1.0, (p.name, r)
        null = {k: "" if isinstance(v, str) else 0 for k, v in cfg["ground_truth"].items()}
        assert not sr.score(cfg, {"answer": null}).passed, p.name

    verified = root.parent / f"{root.name}_verified" / "problems"
    report = {"n_problems": len(probs), "n_dispersion_view": len(linked),
              "n_gold_verified_view": len(list(verified.iterdir())) if verified.is_dir() else 0,
              "gold_check": dict(collections.Counter(c["_provenance"].get("gold_check", "unchecked")
                                                     for c in cfgs)),
              "by_domain": dict(collections.Counter(c["_provenance"]["domain"] for c in cfgs)),
              "by_answer_type": dict(collections.Counter(c["_provenance"]["answer_type"] for c in cfgs)),
              "by_difficulty": dict(collections.Counter(c["_provenance"]["difficulty"] for c in cfgs)),
              "baselines": baselines(g, cfgs)}
    print(json.dumps(report, indent=1))
    print(f"verify OK: {len(probs)} problems load through harness/runner.py, oracle passes and null "
          f"fails every one, grader self-test passes, {len(linked)} scalar problems score 1.0 "
          f"under rl/reward.py in {GENERATED}")
    return report


def baselines(g, cfgs: list[dict]) -> dict:
    """Pass rate and upstream mean score of answers that ignore the data.

    `modal` answers every question with the most frequent gold answer of its answer_type, chosen
    IN-SAMPLE -- an optimistic floor, which is the conservative way to state one.
    """
    by_type = collections.defaultdict(list)
    for c in cfgs:
        (k, v), = c["ground_truth"].items()
        by_type[c["_provenance"]["answer_type"]].append(json.dumps(v))
    modal = {t: json.loads(collections.Counter(vs).most_common(1)[0][0]) for t, vs in by_type.items()}
    out = {}
    for name, answer in (("null", lambda c, v: "" if isinstance(v, str) else 0),
                         ("modal_by_type", lambda c, v: modal[c["_provenance"]["answer_type"]]),
                         ("always_No", lambda c, v: "No"),
                         ("always_True", lambda c, v: "True")):
        res = [g.evaluate(c, {"answer": {k: answer(c, v) for k, v in c["ground_truth"].items()}})
               for c in cfgs]
        out[name] = {"pass_rate": round(sum(r["passed"] for r in res) / len(res), 4),
                     "mean_score": round(sum(r["score"] for r in res) / len(res), 4)}
    out["modal_answers"] = modal
    return out


# --------------------------------------------------------------------------- reproduce
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")

# Solutions whose output does not SPELL the gold but decides it -- checked by reading the replay
# output (the notebook idiom: the answer is read off a table). A human judgement, recorded as one.
INSPECTED = {
    "wildfire-easy-8": "cause counts: L (lightning) 3593 > H (human) 1691 -> 'Lightning'",
    "wildfire-easy-15": "prints r = -0.010 -> not strongly correlated -> 'No'",
    "legal-hard-30": "prints True for each area with more identity-theft than fraud reports; none of 401 -> 'No'",
    "environment-hard-9": ("the beaches with no exceedance are the pleasure_bay_and_castle_island and "
                           "city_point datasheets -> Pleasure Bay, Castle Island, City Point"),
}


TRACE = """import sys as _s, os as _o, atexit as _ae
_opened = set()
def _hook(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, _o.PathLike)):
        _p = _o.path.abspath(_o.fsdecode(args[0]))
        if "/data_files/" in _p:
            _opened.add(_p.split("/data_files/", 1)[1])
_s.addaudithook(_hook)
_ae.register(lambda: print("__OPENED__" + repr(sorted(_opened))))
"""


def replay_source(dom: str, src: str, trace: bool = False) -> str:
    """The upstream solution, runnable in a sandbox whose data sit in data_files/. Replay only --
    none of this reaches an agent.

    What changes: `data_path` points at data_files/; directory names written in the wrong case
    (the authors' filesystem was case-insensitive: 'State MSA Identity Theft Data/' for '... data/')
    are corrected; BeautifulSoup's 'lxml' parser, absent from the sandbox, becomes 'html.parser';
    an UNUSED `import geopandas` (three wildfire solutions carry one) is satisfied by a stub; and a
    trailing bare expression -- the notebook idiom several solutions end on -- is printed.
    """
    src = re.sub(r"^data_path\s*=.*$", 'data_path = "data_files/"', src, flags=re.M)
    src = src.replace(f"./data/{dom}/input", "data_files")
    dirs = {part for f in lake_files(dom) for part in Path(f).parent.parts}
    for d in sorted(dirs, key=len, reverse=True):      # only as a path component
        src = re.sub(re.escape(d) + r"(?=[/'\"])", lambda _m, d=d: d, src, flags=re.I)
    src = src.replace("'lxml'", "'html.parser'").replace('"lxml"', '"html.parser"')
    stub = ("import sys as _s, types as _t, functools as _f\n"
            "for _m in ('geopandas',):\n"
            "    _s.modules.setdefault(_m, _t.ModuleType(_m))\n"
            "import pandas as _pd\n"
            "_pd.read_html = _f.partial(_pd.read_html, flavor='bs4')   # lxml is not installed\n")
    if trace:
        stub = TRACE + stub
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        call = last.value
        if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "print"):
            lines = src.splitlines()
            seg = "\n".join(lines[last.lineno - 1:last.end_lineno])
            src = "\n".join(lines[:last.lineno - 1]) + f"\nprint(repr({seg.strip()}))\n"
    return stub + src


def found_in(stdout: str, v, spec: dict, g, pct: bool = False) -> bool:
    """Lenient: does the gold answer appear anywhere in the solution's output -- allowing the
    rounding to the gold's precision, or (when the question asks for a percentage) the
    fraction-to-percent step, that a notebook-style solution leaves to its reader?"""
    if isinstance(v, list):
        return all(found_in(stdout, x, {"answer_type": "numeric_exact" if isinstance(x, (int, float))
                                        else "string_exact"}, g, pct) for x in v)
    if isinstance(v, str):      # a whole token, so a gold "No" is not found in "Northwest"
        pat = r"(?<![a-z0-9])" + re.escape(v.strip().lower()) + r"(?![a-z0-9])"
        return re.search(pat, stdout.lower()) is not None
    dp = len(repr(float(v)).split(".")[1].rstrip("0")) if "." in repr(float(v)) else 0
    for m in NUM_RE.findall(stdout):
        try:
            x = float(m.replace(",", ""))
        except ValueError:
            continue
        for y in ((x, 100 * x) if pct else (x,)):
            if any(g.grade_key(c, v, spec)[1] for c in (y, round(y, dp))):
                return True
    return False


def strict_value(stdout: str):
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    try:
        return ast.literal_eval(last)
    except Exception:
        nums = NUM_RE.findall(last)
        if nums and len(nums) == 1:
            try:
                return float(nums[0].replace(",", ""))
            except ValueError:
                pass
        return last


def reproduce(root: Path, only: str | None) -> dict:
    """Run every upstream reference solution in harness/sandbox.py on the SHIPPED data_files/.

    Each problem's _provenance.gold_check records the outcome:
      strict          the solution's last printed value passes the grader against the gold
      in_output       the gold appears in its output (after the rounding / percent step the
                      question asks for)
      inspected       not spelled out, but decided by what it prints (INSPECTED, read by hand)
      not_in_output   it runs, and nothing it prints gives the gold
      solution_error  it does not run to completion (an upstream bug, or an undistributed file)
    and benchmarks_ext/kramabench_verified links the strict + in_output + inspected problems.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "harness"))
    from sandbox import Sandbox  # noqa: E402
    g = load_module(root / "reference_grader.py", "kb_grader")
    rows = []
    for p in sorted((root / "problems").iterdir()):
        if only and only not in p.name:
            continue
        cfg = json.loads((p / "eval_config.json").read_text())
        dom = next(d for d in DOMAINS if p.name.startswith(d))
        (k, v), = cfg["ground_truth"].items()
        spec = cfg["grader"]["config"]["keys"][k]
        question = cfg["task"].split("Question:", 1)[1]
        pct = "percent" in question.lower() or "%" in question
        src = (CACHE / "solutions" / dom / f"{p.name}.py").read_text()
        sb = Sandbox(data_dir=p / "data_files", timeout=300)
        try:
            res = sb.run(replay_source(dom, src))
        finally:
            sb.cleanup()
        sv = strict_value(res.stdout)
        strict = g.grade_key(sv, v, spec)[1] if sv is not None else False
        lenient = strict or found_in(res.stdout, v, spec, g, pct)
        err = (res.stderr.strip().splitlines() or [""])[-1][:160] if res.returncode else ""
        check = ("strict" if strict else "in_output" if lenient
                 else "not_in_output" if res.returncode == 0 else "solution_error")
        if check == "not_in_output" and p.name in INSPECTED:
            check = "inspected"
        rows.append({"id": p.name, "gold_check": check, "gold": v,
                     "last_line": (res.stdout.strip().splitlines() or [""])[-1][:120], "error": err})
        cfg["_provenance"]["gold_check"] = check
        (p / "eval_config.json").write_text(json.dumps(cfg, indent=1))
        print(f"  [{check:<14}] {p.name:<22} gold={json.dumps(v)[:40]:<40} "
              f"last={rows[-1]['last_line'][:60]!r}" + (f"  err={err}" if err else ""), flush=True)
    counts = collections.Counter(r["gold_check"] for r in rows)
    summary = {"n": len(rows), "counts": dict(counts), "rows": rows}
    print(f"\nreference solutions on the shipped data: {dict(counts)} of {len(rows)}")
    if not only:
        (root / "reproduce.json").write_text(json.dumps(summary, indent=1, default=str))
        verified = root.parent / f"{root.name}_verified"
        (verified / "problems").mkdir(parents=True, exist_ok=True)
        for q in (verified / "problems").iterdir():
            if q.is_symlink():
                q.unlink()
        for r in rows:
            if r["gold_check"] in ("strict", "in_output", "inspected"):
                (verified / "problems" / r["id"]).symlink_to(root / "problems" / r["id"])
        grader_link = verified / "reference_grader.py"
        if grader_link.is_symlink():
            grader_link.unlink()
        grader_link.symlink_to(root / "reference_grader.py")
        n_ok = counts["strict"] + counts["in_output"] + counts["inspected"]
        print(f"{n_ok} gold-verified problems linked in {verified}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="benchmark root (default $DEEP_DATA_ROOT/benchmarks_ext/kramabench)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--reproduce", action="store_true", help="replay the reference solutions")
    ap.add_argument("--only", default=None, help="substring filter for --reproduce")
    a = ap.parse_args()
    if a.reproduce:
        reproduce(a.root, a.only)
        raise SystemExit(0)
    if not a.verify_only:
        m = build(a.root)
        print(f"built {m['n_kept']} problems in {a.root} ({len(m['dropped'])} dropped, see manifest.json)")
        reproduce(a.root, None)       # stamps gold_check, which a rebuild would otherwise lose
    verify(a.root)
