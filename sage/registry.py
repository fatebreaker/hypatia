"""Family registry: every plug-in in sage/domains, by name.

A domain module that fails to import (its data source unavailable, its generator mid-edit)
does not take the others down: the failure is recorded and shown by `python -m sage list`.
"""
from __future__ import annotations

import importlib
import traceback

MODULES = ("sage.domains.causal", "sage.domains.astro",
           "sage.domains.enzyme",
           "sage.domains.epi", "sage.domains.pk",
           "sage.domains.ecology", "sage.domains.survival",
           "sage.domains.oscillator", "sage.domains.arrhenius",
           "sage.domains.healthcare")

_FAMILIES: dict = {}
_REAL: dict = {}
_ERRORS: dict = {}
_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
        except Exception as exc:                                  # noqa: BLE001
            _ERRORS[m] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
            continue
        for f in getattr(mod, "FAMILIES", []):
            _FAMILIES[f.name] = f
        _REAL.update(getattr(mod, "REAL_SOURCES", {}))
    _LOADED = True


def families() -> dict:
    _load()
    return dict(_FAMILIES)


def get(name: str):
    _load()
    if name not in _FAMILIES:
        hint = f" (import errors: {list(_ERRORS)})" if _ERRORS else ""
        raise KeyError(f"unknown family {name!r}; known: {sorted(_FAMILIES)}{hint}")
    return _FAMILIES[name]


def resolve(spec: str) -> list[str]:
    """'all', 'domain:<d>', or a comma list of names."""
    _load()
    if spec == "all":
        return sorted(_FAMILIES)
    if spec.startswith("domain:"):
        d = spec.split(":", 1)[1]
        out = sorted(n for n, f in _FAMILIES.items() if f.domain == d)
        if not out:
            raise KeyError(f"no family in domain {d!r}")
        return out
    return [get(s).name for s in spec.split(",") if s]


def real_source(name: str):
    _load()
    return _REAL[name]


def errors() -> dict:
    _load()
    return dict(_ERRORS)


def register(fam) -> None:
    """Add a family in this process only (`sage author --check` on a module not yet listed
    in MODULES; forked workers inherit it). Listing the module in MODULES is the real entry."""
    _load()
    _FAMILIES[fam.name] = fam
