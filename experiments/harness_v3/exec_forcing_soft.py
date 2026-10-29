"""EFS -- HARNESS v4 = v3 + THE SOFT EXECUTION GATE (docs/RL_PLAN.md 2026-09-29 04:40 UTC, "HARNESS v4 = v3 + SOFT EXECUTION
GATE (EFS), REGISTERED before any build or run"). The untrained Qwen3-8B in the v3 evaluation harness with the EF gate
(experiments/harness_v3/exec_forcing.py: an answer is accepted only if its numbers were printed by the agent's own executed
code; otherwise one fixed reply and the episode continues under the same limits), changed in EXACTLY ONE place:

  EF   an episode that ends without an accepted answer (the turn limit incl. the final forced call, the wall budget, or the
       stall abort) submits the last offered answer that is grounded against ALL cells the episode executed, else NONE.
  EFS  the same, but when none is grounded it submits the LAST answer the model offered (ungrounded), logged
       accepted = "soft_fallback". An episode in which the model offered no answer still ends with none (as EF, as plain v3).
The rule, the FEEDBACK text, the limits, the gate points and the per-rollout log (rollout["v2"]["ef"]; rec["ef"] on the
externals) are EF's own objects, not copies.

HOW, WITHOUT COPYING THE GATE OR THE LOOP. exec_forcing.py (sha-pinned: EF_2 / EF_4 record it and resume on it) is never
edited and never imported by name. The EF loop (exec_forcing.make_solve) creates its per-episode gate as `Gate(cfg,
sandbox)` -- a MODULE-GLOBAL lookup at call time -- and ends every unaccepted episode through `gate.fallback(transcript)`
(its two lines marked "# EF (was: None)"). The fallback is therefore a METHOD, not a local decision inside the copied loop,
and it is overridden with no code copied:
  load(ext)    a PRIVATE instance of exec_forcing.py, loaded by path as _gbp_efs_exec_forcing (never the EF hook's
               _gbp_exec_forcing, never `import exec_forcing`, so an EF instance in the same process is never touched);
               with ext=True (the externals) experiments/harness_v3/exec_forcing_ext.py -- EF_3's path, unchanged -- is
               install()ed on it first (as _gbp_efs_exec_forcing_x, with its own copy of exec_forcing_ext);
  soften(EF)   EF.Gate := SoftGate(EF.Gate). SoftGate's ONLY override is fallback(): EF's fallback runs first, verbatim
               (super().fallback); only when it returns None AND the model offered an answer, the last offered answer is
               returned and log["accepted"] = "soft_fallback". Nothing else changes: accept(), the log keys, the events.
  The solve installed on LLMAgent is exec_forcing.make_solve's own function object from that private instance (its
  __globals__ ARE that module's namespace, so `Gate` resolves to SoftGate); install_agents asserts it.
Why refused[-1] is "the last answer the model offered": the loop returns on the FIRST accepted answer (gate, n/a or a
fail-open gate error), so when fallback() runs every answer the model offered was refused and appended to gate.refused, in
order: log["offered"] == len(refused) (tests assert it on every scripted branch).
experiments/harness_v3/tests/test_efs.py replays every branch test_ef.py covers through EF and EFS side by side: identical
transcripts, prompts, cells and logs everywhere, identical submissions except exactly where EF returns none after an offer.

HOW IT REACHES THE HARNESS. Only through experiments/harness_v3/efs_site/sitecustomize.py, which
experiments/harness_v3/screen_v3efs.sh puts first on PYTHONPATH together with GBP_EFS=1 (and GBP_EFS_EXT=1 for the
externals); either alone installs nothing. After harness/agents.py's body: install_agents (LLMAgent.solve := the EF loop of
the private, softened instance). After harness/v2.py's body: install_v2 = exec_forcing.install_v2 (V2Sandbox.stats logs
the gate -> rollout["v2"]["ef"]; the EXEC-FORCING banner) [+ exec_forcing_ext.install_v2's banner on the externals], then
HarnessConfig.as_dict DROPS EF's "exec_forcing" key and records "exec_forcing_soft" (record() below: harness
"exec_forcing_soft", this file's and the site's sha, and the EF gate's own record -- rule, feedback, exec_forcing.py sha,
the external formats when installed -- under "gate"). So config.json / every family JSON / harness_v2.json name EFS, never
EF: screen_any.sh refuses to mix an EFS directory with rows of any other configuration, and read_ef.py / read_ef2.py see an
EFS directory as "NOT an EF directory". After harness/runner.py's body (externals only): exec_forcing_ext.install_runner
(each problem JSON carries rec["ef"]).
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EF_PY = HERE / "exec_forcing.py"
EXT_PY = HERE / "exec_forcing_ext.py"
SITE = HERE / "efs_site" / "sitecustomize.py"
FLAG, EXT_FLAG = "GBP_EFS", "GBP_EFS_EXT"
HARNESS = "exec_forcing_soft"
SOFT = "soft_fallback"
FALLBACK = ("at the end of an episode without an accepted answer (turn limit incl. the final forced call, wall budget, "
            "stall abort): the last offered answer grounded against every executed cell (as EF, logged 'fallback'); "
            "else the LAST answer the model offered, ungrounded (logged 'soft_fallback'); none only if no answer was "
            "offered")
EF_NAME = "_gbp_efs_exec_forcing"          # the private instance (TEST / DESIGN)
EF_NAME_X = "_gbp_efs_exec_forcing_x"      # the private instance with the external formats
EXT_NAME = "_gbp_efs_exec_forcing_ext"     # exec_forcing_ext.py, loaded for EF_NAME_X only
_STATE: dict = {"EF": None, "ext": None, "X": None}


def _sha16(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _load(name: str, path: Path):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def soften(EF) -> None:
    """EF.Gate := SoftGate(EF.Gate) in the given exec_forcing instance (idempotent). Only fallback() is overridden."""
    base = EF.Gate
    if getattr(base, "_efs", False):
        return

    class SoftGate(base):
        _efs = True

        def fallback(self, transcript: list):
            sub = super().fallback(transcript)          # EF's fallback, verbatim: the last grounded answer, else None
            if sub is not None or not self.refused:
                return sub                              # grounded fallback, or the model never offered an answer
            self.log["accepted"] = SOFT
            return self.refused[-1]                     # the LAST answer the model offered (every offer was refused)

    SoftGate.__qualname__ = SoftGate.__name__ = "SoftGate"
    SoftGate.__module__ = EF.__name__
    EF.Gate = SoftGate


def private_ef(ext: bool = False, name: str | None = None):
    """A softened private instance of exec_forcing.py (loaded by path under `name`; externals installed first if ext)."""
    name = name or (EF_NAME_X if ext else EF_NAME)
    EF = _load(name, EF_PY)
    X = None
    if ext:
        X = _load(EXT_NAME if name == EF_NAME_X else name + "_ext", EXT_PY)
        X.install(EF)                                   # EF_3's external formats (idempotent), before soften
    elif getattr(EF.grounded, "_ef_ext", False):
        raise RuntimeError(f"{name} already carries the external formats; ext=False refused")
    soften(EF)
    EF._efs_ext_module = X
    return EF


def load(ext: bool = False):
    """The process-wide EFS instance (the one the site installs); ext must be the same on every call."""
    if _STATE["EF"] is not None:
        if _STATE["ext"] != bool(ext):
            raise RuntimeError(f"EFS already loaded with ext={_STATE['ext']}; ext={bool(ext)} refused")
        return _STATE["EF"]
    EF = private_ef(bool(ext))
    _STATE.update(EF=EF, ext=bool(ext), X=EF._efs_ext_module)
    return EF


def current():
    return _STATE["EF"] if _STATE["EF"] is not None else load(False)


def record(EF=None) -> dict:
    """What config.json / each family JSON's "harness_v2" / each ext harness_v2.json records (key "exec_forcing_soft")."""
    EF = EF or current()
    return {"harness": HARNESS, "on": True, "fallback": FALLBACK, "soft_label": SOFT,
            "gate": EF.record(),
            "sha256": {"experiments/harness_v3/exec_forcing_soft.py": _sha16(Path(__file__).resolve()),
                       "experiments/harness_v3/efs_site/sitecustomize.py": _sha16(SITE)}}


def install_agents(A) -> None:
    """LLMAgent.solve := the EF loop of the softened private instance (idempotent; refuses a solve from any other gate)."""
    EF = current()
    EF.install_agents(A)
    solve = A.LLMAgent.solve
    if solve.__globals__ is not vars(EF):
        raise RuntimeError("LLMAgent.solve is an EF loop from another exec_forcing instance (the HARD gate) -- refusing")
    solve._efs = True


def install_v2(V) -> None:
    """EF's provenance / logging patch, then as_dict names EFS instead of EF, and the banner says so (idempotent)."""
    HC = V.HarnessConfig
    if getattr(HC.as_dict, "_efs", False):
        return
    EF = current()
    EF.install_v2(V)                                    # stats -> rollout["v2"]["ef"]; as_dict += "exec_forcing"; banner
    if _STATE["X"] is not None:
        _STATE["X"].install_v2(V)                       # the external-formats banner (EF_3's)
    ef_as_dict, ef_banner = HC.as_dict, V.banner

    def as_dict(self) -> dict:
        d = ef_as_dict(self)
        d.pop("exec_forcing", None)                     # an EFS row never carries EF's key
        d[HARNESS] = record(EF)
        return d

    def banner(hc) -> str:
        return ef_banner(hc) + " + SOFT FALLBACK (harness v4: last offered answer if none grounded; " \
                               "experiments/harness_v3/exec_forcing_soft.py)"

    as_dict._efs = as_dict._ef = banner._efs = banner._ef = True
    HC.as_dict, V.banner = as_dict, banner


def install_runner(R) -> None:
    """Externals only: exec_forcing_ext.install_runner (each problem JSON carries rec["ef"])."""
    if _STATE["X"] is None:
        raise RuntimeError("install_runner needs the externals instance (load(ext=True))")
    _STATE["X"].install_runner(R)
