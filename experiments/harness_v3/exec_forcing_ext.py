"""EF2 / EFX -- THE EXECUTION-FORCING GATE ON THE EXTERNAL BENCHMARKS (docs/RL_PLAN.md 2026-09-28 23:00 UTC, "EF2
REGISTERED (before any build or run)": EFX = the untrained 8B + the gate on DABench / QRData / BixBench vs the 8B plain v3
base externals). The gate itself is experiments/harness_v3/exec_forcing.py, UNCHANGED (this module never edits it, and
exec_forcing.py / ef_site/ are byte-identical to the EF_1 build); this module is loaded only by
experiments/harness_v3/ef2_site/sitecustomize.py, i.e. only under screen_v3ef.sh with EF_EXT=1.

WHY AN EXTENSION IS NEEDED. The registered gate is process_signals' 'grounded' signal, which grounds the NUMERIC answer
keys -- rl.reward.StepReward._fields(cfg): the keys whose ground truth is not a string. Every external benchmark stores
its labels as strings (DABench "34.65", QRData "0.20" / "treatment group", BixBench "C"), so _fields is empty on all 727
external problems and the unextended gate reads "rule n/a -> accept" on EVERY answer: EF on the externals would be the
plain harness. This module maps each benchmark's answer format onto the SAME rule and then CALLS it
(ProcessScorer().signals(...).fired["grounded"] through exec_forcing.scorer / as_messages -- the call exec_forcing.grounded
makes): which cells count as executed, typed literals excluded, the rounding-aware, sign-aware match and the all-keys
semantics are the registered rule's own, not reimplemented. The rule is shown a VIEW of the problem whose ground truth
holds exactly the numeric keys (placeholder 0.0 values: the rule reads only which keys are numeric) and an answer holding
exactly the values below.

FORMAT RULES (which keys are grounded, with which value):
  DABench  (grader "dabench_official")  a key is NUMERIC iff its label parses as a finite number -- StepReward._fields'
           own criterion (a numeric ground truth) with the benchmark's string-encoded numbers read as numbers (207 of 257
           problems have >= 1 such key). The submitted value goes to the rule AS SUBMITTED (the rule's own float():
           missing, a list, a phrase -> "missing or not numeric", 'nan' -> "not finite" -> refused, exactly as on the
           synthetic families). Text labels ('no', 'linear', 'Nigeria', composite '1, 2018, 88.32', '', and da-554's
           non-finite 'nan') -> that key is not grounded (n/a).
  QRData   (grader "qrdata_official")   question_type "numerical" (the benchmark's own declaration; 163 of 411) ->
           NUMERIC. Value = the submission if it is a number, or a string that is one (thousands separators allowed);
           otherwise the grader's own reading (benchmarks/qrdata/reference_grader.extract_first_number: the FIRST number
           in the response). A percent "x%" is grounded if x OR x/100 was printed (the grader scores x/100). No number
           -> refused ("missing or not numeric", as above). "multiple_choice" (248: text options -- causal-discovery
           A/B/C/D statements, yes/no, group names) -> n/a.
  BixBench (grader "multi_choice_exact") the chosen letter (benchmarks/bixbench/reference_grader.normalise, the grader's
           own reading) -> that option's text in the task. An option that IS a single number (optionally "~", "%",
           scientific notation, thousands separators; 415 of 820 options) -> NUMERIC with that value (x% as above). An
           interval "(a,b)", a range "a-b", an inequality ">100", a gene name or a phrase, a letter that is not an option,
           or no letter -> n/a for that key. An interval is NOT grounded by "some printed number falls inside it": with
           adjacent intervals and a printed table that accepts almost any choice, so it would not discriminate.
  A problem's answer is GATED iff >= 1 key is NUMERIC; it is accepted iff the rule fires on every numeric key (for a
  percent, either reading); no numeric key -> the rule does not apply: accept, logged "na" (exec_forcing's own n/a path).
SYNTHETIC FAMILIES UNCHANGED: a cfg whose grader type is not one of the three above goes to exec_forcing.grounded itself
(the same function object, called with the same arguments); experiments/harness_v3/tests/test_ef2.py replays every gate
decision of the stored V3EF rollouts through both and finds them byte-identical.

LOGGING. Per offered answer on an external problem: rollout["ef"]["ext"] = [{turn, phase, accepted, kind, numeric (keys
grounded), na {key: why}, values {key: [candidates]}, fired, why}], beside exec_forcing's own log (rejected, offered, na,
gate_errors, accepted, events). harness/runner.py never calls sandbox.stats(), so its per-problem JSON would not carry
the log: install_runner wraps runner.run_one (a module global, looked up by runner.main at call time) to add rec["ef"]
and rewrite that problem's JSON atomically after the original wrote it. Nothing else of runner.py changes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE2 = HERE / "ef2_site" / "sitecustomize.py"
EXT_GRADERS = {"dabench_official": "dabench", "qrdata_official": "qrdata", "multi_choice_exact": "bixbench"}
RULE_EXT = ("externals (experiments/harness_v3/exec_forcing_ext.py): the SAME 'grounded' rule on a view whose numeric keys "
            "are -- DABench: labels that parse as numbers (submitted value as submitted); QRData: question_type "
            "'numerical' (value = the number, else the grader's first number; x% grounded as x or x/100); BixBench: the "
            "chosen option when it is a single number (x% as above). Text labels / multiple-choice text / intervals, "
            "ranges, names -> n/a (accept, logged).")
_TLS = threading.local()
_GRADERS: dict = {}
_GLOCK = threading.Lock()
NUM_FULL = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
THOUSANDS = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _sha16(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def ext_record() -> dict:
    return {"rule": RULE_EXT, "graders": dict(EXT_GRADERS),
            "sha256": {"experiments/harness_v3/exec_forcing_ext.py": _sha16(Path(__file__).resolve()),
                       "experiments/harness_v3/ef2_site/sitecustomize.py": _sha16(SITE2)}}


def ext_kind(cfg) -> str | None:
    """'dabench' / 'qrdata' / 'bixbench' by the problem's own grader type, else None (every synthetic family)."""
    g = (cfg or {}).get("grader") if isinstance(cfg, dict) else None
    return EXT_GRADERS.get(g.get("type")) if isinstance(g, dict) else None


def grader(kind: str):
    """The benchmark's vendored reference_grader.py (benchmarks/<kind>/), loaded by path once per process."""
    with _GLOCK:
        mod = _GRADERS.get(kind)
        if mod is None:
            spec = importlib.util.spec_from_file_location(f"_gbp_ef_grader_{kind}",
                                                          ROOT / "benchmarks" / kind / "reference_grader.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _GRADERS[kind] = mod
        return mod


def _answer_obj(submission) -> dict:
    obj = submission if isinstance(submission, dict) else {}
    if "answer" in obj and isinstance(obj["answer"], dict):
        obj = obj["answer"]
    return obj


def number(x):
    """A finite float if x is a number (not a bool) or a string that is exactly one (thousands separators allowed)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        s = x.strip()
        if THOUSANDS.fullmatch(s):
            s = s.replace(",", "")
        if NUM_FULL.fullmatch(s):
            v = float(s)
            return v if math.isfinite(v) else None
    return None


def numeric_text(s: str):
    """Candidates of a text that IS one number: [v] or, for 'x%', [x, x/100]; None otherwise. Allows a leading '~' /
    '≈' / 'approximately' / 'about', scientific notation and thousands separators."""
    if not isinstance(s, str):
        return None
    t = s.strip().replace("−", "-")
    t = re.sub(r"^(?:~|≈|approx\.?|approximately|about)\s*", "", t, flags=re.I)
    pct = t.endswith("%")
    if pct:
        t = t[:-1].strip()
    v = number(t)
    if v is None:
        return None
    return [v, v / 100.0] if pct else [v]


def option_kind(s: str) -> str:
    """For the log only: what kind of non-numeric option this is."""
    t = (s or "").strip()
    if re.fullmatch(r"[\(\[]\s*[-+]?[\d.eE+-]+\s*,\s*[-+]?[\d.eE+-]+\s*[\)\]]", t):
        return "interval"
    if re.fullmatch(r"[-+]?[\d.,]+\s*(?:-|to|–)\s*[-+]?[\d.,]+%?", t):
        return "range"
    if re.match(r"^\s*(?:[<>]=?|≤|≥)", t):
        return "inequality"
    return "text"


def bix_options(task: str, key: str) -> dict:
    """{letter: option text} of question `key` in a BixBench task ('Q1 [bix-1-q1]: ...' then '    A) ...' lines)."""
    m = re.search(r"\[" + re.escape(key) + r"\]:(.*?)(?=\n\s*\nQ\d+ \[|\n\s*\nReturn ONE|\Z)", task or "", re.S)
    if not m:
        return {}
    return {a: b.strip() for a, b in re.findall(r"^[ \t]+([A-Z])\)[ \t]*(.*)$", m.group(1), re.M)}


def plan(cfg: dict, submission, kind: str | None = None) -> dict:
    """Which keys the rule grounds, and with which candidate values:
    {"kind", "numeric": {key: [candidates] | [raw]}, "na": {key: why}}. For DABench the raw submitted value is passed
    (the rule's own float() decides "missing or not numeric"); a numeric key with no usable value carries [None]."""
    kind = kind or ext_kind(cfg)
    obj = _answer_obj(submission)
    num, na = {}, {}
    if kind == "dabench":
        for k, lab in (cfg.get("ground_truth") or {}).items():
            if number(lab) is None:
                na[k] = "text label"
            else:
                num[k] = [obj.get(k)]
    elif kind == "qrdata":
        keys = (((cfg.get("grader") or {}).get("config") or {}).get("keys")) or {}
        for k, spec in keys.items():
            if (spec or {}).get("question_type", "numerical") != "numerical":
                na[k] = "multiple choice (text answer)"
                continue
            raw = obj.get(k)
            v = number(raw)
            if v is not None:
                num[k] = [v]
                continue
            p = grader("qrdata").extract_first_number(str(raw).lower().strip()) if raw is not None else None
            if p is None:
                num[k] = [None]
            elif p.endswith("%"):
                x = float(p[:-1])
                num[k] = [x, x / 100.0]
            else:
                num[k] = [float(p)]
    elif kind == "bixbench":
        keys = list((((cfg.get("grader") or {}).get("config") or {}).get("keys")) or {}) or list(cfg.get("ground_truth") or {})
        norm = grader("bixbench").normalise
        for k in keys:
            letter = norm(obj.get(k)) if k in obj else None
            opts = bix_options(cfg.get("task") or "", k)
            if letter is None:
                na[k] = "no option letter"
            elif letter not in opts:
                na[k] = f"letter {letter} is not an option"
            else:
                c = numeric_text(opts[letter])
                if c is None:
                    na[k] = f"option {letter} is {option_kind(opts[letter])}: {opts[letter][:60]}"
                else:
                    num[k] = c
    return {"kind": kind, "numeric": num, "na": na}


def _view(cfg: dict, keys) -> dict:
    """The problem as the rule reads it: ground truth = exactly `keys`, all numeric (values never read by the rule)."""
    v = dict(cfg)
    v["ground_truth"] = {k: 0.0 for k in keys}
    v["grader"] = {"type": "ef_ext_view", "config": {}}
    return v


def _rule(EF, transcript, cfg, data_dir, vals: dict):
    """The registered rule (exec_forcing.scorer().signals(...).fired['grounded']) on the view with answer `vals`."""
    sig = EF.scorer().signals(EF.as_messages(transcript), _view(cfg, list(vals)), data_dir, answer={"answer": dict(vals)})
    return sig.fired.get("grounded"), sig.detail.get("grounded")


def _safe(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)[:120]


def grounded_ext(EF, transcript: list, cfg: dict, data_dir, submission, kind: str | None = None):
    """(fired, reason) for an external problem: 1.0 / 0.0 from the registered rule, None when no key is numeric."""
    pl = plan(cfg, submission, kind)
    num = pl["numeric"]
    info = {"kind": pl["kind"], "numeric": sorted(num), "na": dict(pl["na"]),
            "values": {k: [_safe(x) for x in v] for k, v in num.items()}}
    _TLS.last = info
    if not num:
        why = f"ext:{pl['kind']} rule n/a (no numeric key: " + "; ".join(f"{k}: {w}" for k, w in sorted(pl["na"].items())) + ")"
        info.update(fired=None, why=why[:300])
        return None, why
    if all(len(v) == 1 for v in num.values()):
        fired, why = _rule(EF, transcript, cfg, data_dir, {k: v[0] for k, v in num.items()})
    else:
        bad = []
        for k, cands in sorted(num.items()):
            if not any(_rule(EF, transcript, cfg, data_dir, {k: c})[0] == 1.0 for c in cands):
                bad.append(k)
        fired = 0.0 if bad else 1.0
        why = (f"not in any computed output (any reading): {bad}" if bad
               else "every numeric answer value was printed by an executed cell (any reading)")
    why = f"ext:{pl['kind']} {why}"
    info.update(fired=fired, why=str(why)[:300])
    return fired, why


def install(EF) -> None:
    """In the loaded exec_forcing module (idempotent): grounded := dispatch (externals -> the format view, everything
    else -> the original function, unchanged); Gate := a subclass that logs each external offer and registers the
    episode's gate for the runner hook; record() gains the "ext" record (-> config.json / harness_v2.json)."""
    if getattr(EF.grounded, "_ef_ext", False):
        return
    orig_grounded, orig_record, Gate0 = EF.grounded, EF.record, EF.Gate

    def grounded(transcript, cfg, data_dir, submission):
        kind = ext_kind(cfg)
        if kind is None:
            return orig_grounded(transcript, cfg, data_dir, submission)
        return grounded_ext(EF, transcript, cfg, data_dir, submission, kind)

    grounded._ef_ext = True
    grounded._orig = orig_grounded

    class Gate(Gate0):
        def __init__(self, cfg, sandbox):
            super().__init__(cfg, sandbox)
            self._ext = ext_kind(cfg)
            if self._ext:
                self.log["ext"] = []
            _TLS.gate = self

        def accept(self, submission, transcript, turn, phase):
            if not self._ext:
                return super().accept(submission, transcript, turn, phase)
            _TLS.last = None
            ok = super().accept(submission, transcript, turn, phase)
            info = _TLS.last or {"kind": self._ext, "error": True}
            self.log["ext"].append({"turn": turn, "phase": phase, "accepted": bool(ok), **info})
            return ok

    Gate._ef_ext = True

    def record() -> dict:
        d = orig_record()
        d["ext"] = ext_record()
        return d

    record._ef_ext = True
    EF.grounded, EF.Gate, EF.record = grounded, Gate, record


def install_v2(V) -> None:
    """The banner names the external formats too (after exec_forcing.install_v2; idempotent)."""
    if getattr(V.banner, "_ef_ext", False):
        return
    orig = V.banner

    def banner(hc) -> str:
        return orig(hc) + " [+ external answer formats: experiments/harness_v3/exec_forcing_ext.py]"

    banner._ef_ext = True
    V.banner = banner


def install_runner(R) -> None:
    """harness/runner.py: run_one := the original, then rec["ef"] = the episode's gate log, and the problem's JSON is
    rewritten atomically with it (idempotent)."""
    if getattr(R.run_one, "_ef_ext", False):
        return
    orig = R.run_one

    def run_one(pdir, agent_spec, outdir, timeout):
        _TLS.gate = None
        rec = orig(pdir, agent_spec, outdir, timeout)
        g = getattr(_TLS, "gate", None)
        _TLS.gate = None
        rec["ef"] = dict(g.log) if g is not None else None
        p = Path(outdir) / f"{rec['id']}.json"
        tmp = p.with_name(f".{p.name}.ef.tmp")
        tmp.write_text(json.dumps(rec, indent=2, default=str))
        os.replace(tmp, p)
        return rec

    run_one._ef_ext = True
    run_one._orig = orig
    R.run_one = run_one
