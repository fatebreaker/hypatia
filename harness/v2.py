"""HARNESS V2: evidence-producing components as independent opt-in switches. docs/HARNESS_V2.md.

Every switch is OFF by default, and none of the files that running jobs import is touched: harness/sandbox.py,
harness/agents.py, harness/runner.py, rl/dispersion.py and rl/grpo_onpolicy.py are byte-identical. The components
reach an episode only through the v2 ENTRY POINTS, which install them into those modules' namespaces the way
rl/dispersion_sib.py installs siblings:

    experiments/harness_v2/dispersion_v2.py   generated families      (rl/dispersion.py main, Sandbox/one_rollout)
    experiments/harness_v2/runner_v2.py       DABench/QRData/BixBench (harness/runner.py main, Sandbox/make_agent)
    rl/grpo_v2.py                             RL rollouts             (rl/grpo_onpolicy.py main, one_rollout)

Switches (environment, read by HarnessConfig.from_env):
    GBP_KERNEL=1          KERNEL    persistent Python process per episode (harness/kernel.py) instead of cell replay
    GBP_PROFILE=1         PROFILE   descriptive data profile after the file listing (harness/data_profile.py)
    GBP_APIDOCS=1         APIDOCS   real signature + docstring head appended to argument/attribute errors; help_api()
    GBP_LIBRARY=1|PATH    LIBRARY   GBP_LIBRARY_K (default 3) verified worked programs, code-only, <= 24k chars in all,
                                    retrieved by TF-IDF similarity of the task text + table headers from
                                    experiments/harness_v2/library/ (or the manifest PATH), the target
                                    family's own program -- and every variant/twin of it -- EXCLUDED
    GBP_SIBLINGS=K        SIBLINGS  K unlabeled sibling datasets of the same family (rl/siblings.py); generated
                                    families only (externals and single-instance real targets run without)
    GBP_CHECKS=1          CHECKS    `import diagnostics` works in the sandbox (harness/sandbox_lib/diagnostics.py)
    GBP_SYSTEM=science    PROMPT    the science-neutral system prompt (already in harness/agents.py; read there)

With every switch off, V2Sandbox IS Sandbox (run, listing, preamble delegate unchanged) and the rollout wrapper adds only
bookkeeping fields ("seconds", "v2") to the returned dict -- tested in experiments/harness_v2/tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from kernel import LIB_DIR, KernelSandbox
from sandbox import Sandbox, sandbox_python

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY = ROOT / "experiments" / "harness_v2" / "library" / "manifest.json"

CHECKS_NOTE = ("Your Python session can also import `diagnostics`: functions that compute descriptive statistics from "
               "data and return numbers (`help(diagnostics)` lists them).")
APIDOCS_NOTE = ("`help_api(\"package.function\")` is defined in your Python session: it prints the installed "
                "signature and documentation of a function, class or module.")


# --------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class HarnessConfig:
    kernel: bool = False
    profile: bool = False
    apidocs: bool = False
    library: str = ""            # "" = off; else a manifest path
    library_k: int = 3
    siblings: int = 0
    checks: bool = False

    @classmethod
    def from_env(cls, env=None) -> "HarnessConfig":
        env = os.environ if env is None else env
        on = lambda k: env.get(k, "0").strip().lower() in ("1", "true", "yes", "on")   # noqa: E731
        lib = env.get("GBP_LIBRARY", "").strip()
        if lib.lower() in ("0", "", "false", "off", "no"):
            lib = ""
        elif lib.lower() in ("1", "true", "yes", "on"):
            lib = str(DEFAULT_LIBRARY)
        return cls(kernel=on("GBP_KERNEL"), profile=on("GBP_PROFILE"), apidocs=on("GBP_APIDOCS"), library=lib,
                   library_k=int(env.get("GBP_LIBRARY_K", "3") or 3), siblings=int(env.get("GBP_SIBLINGS", "0") or 0),
                   checks=on("GBP_CHECKS"))

    def any(self) -> bool:
        return any((self.kernel, self.profile, self.apidocs, bool(self.library), self.siblings, self.checks))

    def tag(self) -> str:
        """Short name for run directories: '' when everything is off."""
        parts = [p for p, on in (("k", self.kernel), ("pf", self.profile), ("ad", self.apidocs),
                                 (f"lb{self.library_k}", bool(self.library)), (f"sb{self.siblings}", self.siblings > 0),
                                 ("ck", self.checks)) if on]
        return "-".join(parts)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["system"] = os.environ.get("GBP_SYSTEM", "default")
        d["echo_last"] = os.environ.get("GBP_ECHO_LAST", "0")
        return d


# --------------------------------------------------------------------------- PROFILE
_PROFILE_CACHE: dict = {}
_PROFILE_LOCK = threading.Lock()
PROFILE_MAX_CHARS = 6000
PROFILE_TIMEOUT = 90


def _signature(data_dir: Path) -> str:
    h = hashlib.sha1()
    for root, dirs, files in os.walk(data_dir, followlinks=True):
        rel = Path(root).relative_to(data_dir)
        if rel.parts and rel.parts[0] == "siblings":
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if not (not rel.parts and d == "siblings"))
        for f in sorted(files):
            try:
                st = os.stat(Path(root) / f)
            except OSError:
                continue
            h.update(f"{rel / f}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def profile_text(data_dir) -> str:
    """The data profile of a task's files (sibling datasets skipped), computed once per distinct file set."""
    if not data_dir or not Path(data_dir).is_dir():
        return ""
    key = _signature(Path(data_dir))
    with _PROFILE_LOCK:
        if key in _PROFILE_CACHE:
            return _PROFILE_CACHE[key]
    try:
        res = subprocess.run([sandbox_python(), str(ROOT / "harness" / "data_profile.py"), str(data_dir),
                              "--max-chars", str(PROFILE_MAX_CHARS)],
                             capture_output=True, text=True, timeout=PROFILE_TIMEOUT,
                             env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                  "MKL_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"})
        text = res.stdout.strip() if res.returncode == 0 else ""
    except (subprocess.TimeoutExpired, OSError):
        text = ""
    with _PROFILE_LOCK:
        _PROFILE_CACHE[key] = text
    return text


# --------------------------------------------------------------------------- sandbox
_TLS = threading.local()          # the sandbox the current rollout thread built last (for per-rollout stats)


@dataclass
class V2Sandbox(KernelSandbox):
    """Sandbox with the v2 components its class-level HC enables. Constructor-compatible with Sandbox."""

    HC: ClassVar[HarnessConfig] = HarnessConfig()

    def __post_init__(self) -> None:
        hc = self.HC
        self.expose_lib = bool(hc.apidocs or hc.checks)
        self.apidocs = bool(hc.apidocs)
        self.kernel_preamble = ["from gbp_help import help_api"] if hc.apidocs else []
        super().__post_init__()
        self._listing_cache = None
        self.profile_chars = 0
        _TLS.last = self

    def run(self, code: str):
        if self.HC.kernel:
            return KernelSandbox.run(self, code)
        return Sandbox.run(self, code)

    def _limit_preamble(self) -> str:
        base = Sandbox._limit_preamble(self)
        hc = self.HC
        if not (hc.apidocs or hc.checks):
            return base
        extra = f"import sys as _gs\n_gs.path.append({str(LIB_DIR)!r})\ndel _gs\n"
        if hc.apidocs:
            extra += ("from gbp_help import help_api\nimport gbp_help as _gh\n_gh.install_excepthook()\ndel _gh\n")
        return base + extra

    def listing(self) -> str:
        hc = self.HC
        if not (hc.profile or hc.checks or hc.apidocs):
            return Sandbox.listing(self)
        if self._listing_cache is None:
            parts = [Sandbox.listing(self)]
            if hc.profile:
                p = profile_text(self.data_dir)
                self.profile_chars = len(p)
                if p:
                    parts.append("\n" + p)
            notes = ([CHECKS_NOTE] if hc.checks else []) + ([APIDOCS_NOTE] if hc.apidocs else [])
            if notes:
                parts.append("\n" + "\n".join(notes))
            self._listing_cache = "\n".join(parts)
        return self._listing_cache

    def stats(self) -> dict:
        s = KernelSandbox.stats(self) if self.HC.kernel else {}
        if self.HC.profile:
            s["profile_chars"] = self.profile_chars
        return s


def sandbox_class(hc: HarnessConfig):
    """A V2Sandbox subclass bound to `hc` (one class per configuration, so tests can hold several)."""
    return type(f"V2Sandbox_{hc.tag() or 'off'}", (V2Sandbox,), {"HC": hc})


# --------------------------------------------------------------------------- LIBRARY
# Library exclusion is by FAMILY GROUP: a family's variants (blind, toggled, fixed, reworded, sub-question pools) and
# its same-program twins (carrier_cnv/carrier_var, hic_loop/hic_var, popgen_pulse/popgen_var -- one trace builder,
# one program) are the same family for held-out honesty. Distinct families whose programs happen to be close
# (epi_rt/epi_r0) are NOT merged: that is the near-twin transfer the programme measured, reported separately.
_GROUP_ALIAS = {
    "carrier_cnv": "carrier", "carrier_var": "carrier",
    "hic_loop": "hic", "hic_var": "hic", "hic_loop_sage": "hic",
    "popgen_pulse": "popgen", "popgen_var": "popgen", "real_popgen": "popgen",
    "wf_selection": "wf", "wf_selection_shift": "wf", "wf_selection_wide": "wf", "real_wf": "wf",
    "scrna_1k": "scrna", "scrna_ambient": "scrna", "real_scrna": "scrna",
    "cis_mvmr_1k": "cis_mvmr", "rare_burden_1k": "rare_burden", "real_qtl": "mpp_qtl",
    "pk_real_theoph": "pk_oral", "hc_dx_real_spect": "hc_dx", "a2_enzyme_curr": "enzyme_mm",
}
_SUFFIXES = ("_blind", "_bl", "_fx", "_tg", "_rw", "_sq1", "_sq2", "_val")
_MULTI = {  # targets whose "own science" spans several library families
    "kepler_strict": {"astro_eb", "astro_rot", "astro_transit"},
    "causci": {"causal_att", "causal_confounded", "causal_did", "causal_iv", "causal_rdd"},
}


def group_of(name: str) -> str:
    n = name
    for s in _SUFFIXES:
        if n.endswith(s):
            n = n[: -len(s)]
            break
    return _GROUP_ALIAS.get(n, n)


def excluded_families(target: str | None, library_families) -> set:
    if not target:
        return set()
    out = {f for f in library_families if group_of(f) == group_of(target)}
    for prefix, fams in _MULTI.items():
        if target.startswith(prefix):
            out |= set(fams) & set(library_families)
    return out


_STOP = set("""the and for with from that this are was were you your each any all not but can has have its into
per than then them they will which when what where who whose would should could our out use used using also more most
other such only one two three may must given file files data column columns value values report estimate estimated
answer json object return final number numbers key keys float int string exactly omit shown example prose markdown
wrap before after do does python code run cell cells import print pandas numpy read csv tsv gz df np pd""".split())
_TOK = re.compile(r"[a-z][a-z0-9_]{2,}")


def _tokens(text: str) -> list:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


def _table_headers(data_dir, limit: int = 12) -> str:
    """Column names of the task's tables (first line of each; no parsing), for retrieval."""
    import gzip
    out = []
    try:
        files = sorted(p for p in Path(data_dir).rglob("*") if p.is_file() and "siblings" not in p.parts)[:limit]
    except OSError:
        return ""
    for p in files:
        out.append(p.name)
        if re.search(r"\.(csv|tsv|txt|tab)(\.gz)?$", p.name, re.I):
            try:
                with (gzip.open(p, "rt", errors="replace") if p.name.endswith(".gz") else
                      open(p, errors="replace")) as fh:
                    out.append(fh.readline()[:2000])
            except OSError:
                pass
    return " ".join(out)


class Library:
    """Verified worked programs (code-only message lists) with TF-IDF retrieval and family-group exclusion."""

    def __init__(self, manifest: str | Path, k: int = 3, max_chars: int = 24000):
        self.manifest_path = Path(manifest)
        man = json.loads(self.manifest_path.read_text())
        self.k, self.max_chars = k, int(man.get("max_chars", max_chars))
        # 2026-10-10 (H-RETRIEVE): a manifest with exact_match retrieves only entries whose "match" string occurs verbatim
        # in the query (e.g. 'Calculator: <name>\n' in a MedCalc task) and nothing otherwise; TF-IDF is not used then.
        self.exact = bool(man.get("exact_match"))
        self.entries = {}
        for fam, meta in sorted(man["programs"].items()):
            path = Path(meta["path"])
            if not path.is_absolute():
                path = ROOT / path
            msgs = json.loads(path.read_text())
            doc = " ".join(m["content"] for m in msgs if m["role"] in ("user", "assistant"))
            self.entries[fam] = {"messages": msgs, "tokens": Counter(_tokens(doc)), "chars": sum(len(m["content"]) for m in msgs),
                                 "match": meta.get("match")}
        n = len(self.entries)
        df = Counter()
        for e in self.entries.values():
            df.update(set(e["tokens"]))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        for e in self.entries.values():
            e["vec"] = self._vec(e["tokens"])

    def _vec(self, tf: Counter) -> dict:
        v = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def rank(self, target_family: str | None, query: str) -> list:
        excl = excluded_families(target_family, self.entries)
        if self.exact:
            return [fam for fam, e in sorted(self.entries.items()) if e.get("match") and e["match"] in query and fam not in excl]
        q = self._vec(Counter(_tokens(query)))
        scored = [(sum(q.get(t, 0.0) * x for t, x in e["vec"].items()), fam)
                  for fam, e in self.entries.items() if fam not in excl]
        return [fam for s, fam in sorted(scored, key=lambda z: (-z[0], z[1]))]

    def retrieve(self, target_family: str | None, query: str) -> tuple[list, list]:
        """(stacked message list, [families]) -- the top-k programs that fit LIBRARY_MAX_CHARS together (a program
        that would overflow the budget is skipped, the next one tried; the best one is always kept), most similar
        LAST (nearest the task)."""
        from agents import ICL_BRIDGE
        top, used = [], 0
        for fam in self.rank(target_family, query):
            c = self.entries[fam]["chars"]
            if top and used + c > self.max_chars:
                continue
            top.append(fam)
            used += c
            if len(top) >= self.k:
                break
        parts = [self.entries[f]["messages"] for f in reversed(top)]
        stacked = []
        for i, m in enumerate(parts):
            stacked += m if i == 0 else [{**m[0], "content": ICL_BRIDGE + m[0]["content"]}] + m[1:]
        return stacked, list(reversed(top))


_LIB_CACHE: dict = {}


def library_for(hc: HarnessConfig):
    if not hc.library:
        return None
    key = (hc.library, hc.library_k)
    if key not in _LIB_CACHE:
        _LIB_CACHE[key] = Library(hc.library, hc.library_k)
    return _LIB_CACHE[key]


def family_of_data_dir(data_dir) -> str | None:
    """generated/<family>/<split>/<instance>/data_files -> family (pool symlinks resolved)."""
    try:
        p = Path(data_dir).resolve()
        return p.parents[2].name if p.name == "data_files" else p.parents[1].name
    except (IndexError, OSError):
        return None


def query_for(cfg: dict, data_dir) -> str:
    return (cfg.get("task") or "") + "\n" + _table_headers(data_dir)


# --------------------------------------------------------------------------- the rollout wrapper
def wrap_rollout(one_rollout, hc: HarnessConfig, mode: str = "eval", tmp_root: str | None = None):
    """A drop-in replacement for rl.dispersion.one_rollout with the per-episode v2 components (LIBRARY, SIBLINGS) and
    per-rollout bookkeeping. mode "eval": siblings keyed "eval" (the same siblings for every checkpoint), the library
    in context but not in the stored episode; mode "rl": siblings keyed by the policy spec (one draw per GRPO group
    and step), the library kept in the stored episode with n_prefix set, so rl/grpo_onpolicy.train masks it."""
    lib = library_for(hc)
    key = "eval" if mode == "eval" else None

    def v2_rollout(cfg: dict, data_dir, spec: str, *a, **kw) -> dict:
        t0 = time.time()
        fam = family_of_data_dir(data_dir)
        info: dict = {}
        if lib is not None and kw.get("demo") is None and len(a) < 6:
            demo, fams = lib.retrieve(fam, query_for(cfg, data_dir))
            if demo:
                kw["demo"] = demo
                kw["demo_in_messages"] = (mode == "rl")
                info["library"] = fams
                info["library_chars"] = sum(len(m["content"]) for m in demo)
        d, run_cfg, sibs = data_dir, cfg, []
        if hc.siblings > 0:
            from rl import siblings as S
            try:
                sibs = S.pick(Path(data_dir).parent, hc.siblings, key if key is not None else spec)
            except (RuntimeError, OSError, ValueError):
                sibs = []
                info["siblings_skipped"] = True
            if sibs:
                d = S.composite(Path(data_dir), sibs, tmp_root)
                run_cfg = {**cfg, "task": S.with_note(cfg["task"], hc.siblings)}
        _TLS.last = None
        try:
            out = one_rollout(run_cfg, d, spec, *a, **kw)
        finally:
            if sibs:
                shutil.rmtree(d, ignore_errors=True)
        sb = getattr(_TLS, "last", None)
        if sb is not None and hasattr(sb, "stats"):
            info.update(sb.stats())
        if sibs:
            from rl import siblings as S
            out["siblings"] = [s.name for s in sibs]
            out["sib_ref"], out["sib_n"] = S.usage(out, hc.siblings)
        out["seconds"] = out.get("seconds", round(time.time() - t0, 1))
        out["v2"] = info
        return out

    v2_rollout.harness_v2 = hc
    return v2_rollout


# --------------------------------------------------------------------------- installers (used by the entry points)
def install_dispersion(dispersion_mod, hc: HarnessConfig, mode: str = "eval"):
    """rl/dispersion.py: its one_rollout builds `Sandbox(...)` from its own module globals and main() calls its
    module-global one_rollout -- both are replaced here."""
    dispersion_mod.Sandbox = sandbox_class(hc)
    dispersion_mod.one_rollout = wrap_rollout(dispersion_mod.one_rollout, hc, mode)
    return dispersion_mod


def install_grpo(grpo_mod, dispersion_mod, hc: HarnessConfig):
    """rl/grpo_onpolicy.py imported one_rollout by name; the Sandbox it builds lives in rl.dispersion's globals."""
    dispersion_mod.Sandbox = sandbox_class(hc)
    grpo_mod.one_rollout = wrap_rollout(grpo_mod.one_rollout, hc, mode="rl")
    return grpo_mod


def install_runner(runner_mod, hc: HarnessConfig):
    """harness/runner.py (external benchmarks): Sandbox, and the agent's in-context library per problem."""
    runner_mod.Sandbox = sandbox_class(hc)
    lib = library_for(hc)
    if lib is None:
        return runner_mod
    orig_make = runner_mod.make_agent

    def make_agent(spec):
        agent = orig_make(spec)
        if not hasattr(agent, "icl_demo"):
            return agent
        orig_solve = agent.solve

        def solve(cfg, sb):
            demo, fams = lib.retrieve(None, query_for(cfg, sb.data_dir))
            agent.icl_demo = demo or []
            return orig_solve(cfg, sb)

        agent.solve = solve
        return agent

    runner_mod.make_agent = make_agent
    return runner_mod


def banner(hc: HarnessConfig) -> str:
    on = [k for k, v in asdict(hc).items() if v and k != "library_k"]
    return (f"harness v2: {', '.join(on) if on else 'ALL COMPONENTS OFF'} "
            f"(system={os.environ.get('GBP_SYSTEM', 'default')}, echo={os.environ.get('GBP_ECHO_LAST', '0')})")
