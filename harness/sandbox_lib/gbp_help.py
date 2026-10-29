"""APIDOCS (harness v2, switch GBP_APIDOCS=1): the installed library's own documentation, by introspection.

docs/HARNESS_V2.md. Real transcripts spend many turns guessing library call signatures (10 turns guessing lifelines
CoxPHFitter.fit keyword names in one episode). The information is on the machine; nothing needs a network. Two parts:

  help_api(name_or_object)      a helper the cells can call: signature, docstring head, public members
  explain_exception(t, v, tb)   when a cell raises a TypeError about a call's ARGUMENTS, or an AttributeError, the
                                harness appends the real signature and docstring head of the callable involved (or the
                                closest attribute names) to the error output. Evidence, not advice: nothing here says
                                what the analysis should be.

Runs inside the sandbox interpreter (imported by the kernel worker, or by the replay script's preamble through
install_excepthook). Must never raise: every path is wrapped, and a failure simply adds nothing.
"""
from __future__ import annotations

import ast
import difflib
import importlib
import inspect
import linecache
import re
import sys
import types

HEADER = "[API reference -- introspected from the installed library]"
MAX_NOTE = 1600
DOC_LINES = 12

# TypeErrors that are about HOW a callable was called, not about the data it was given
ARG_ERR = re.compile(
    r"(unexpected keyword argument|required positional argument|required keyword-only argument|"
    r"positional arguments? but \d+ (?:were|was) given|takes (?:no|\d+|from \d+ to \d+) (?:positional |keyword )?"
    r"arguments?|multiple values for (?:keyword )?argument|takes no keyword arguments|"
    r"positional-only arguments? passed as keyword|missing \d+ required)")
WRAPPER_NAMES = {"f", "wrapper", "inner", "wrapped", "decorator", "new_func", "_f", "func", "wrapper_function",
                 "<lambda>", "__call__"}


def _qualname(obj) -> str:
    mod = getattr(obj, "__module__", None) or ""
    qn = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or type(obj).__name__
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    return f"{mod}.{qn}" if mod and mod != "builtins" else str(qn)


def _signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def _doc_head(obj, n: int = DOC_LINES) -> str:
    try:
        doc = inspect.getdoc(obj) or ""
    except Exception:                                         # noqa: BLE001
        doc = ""
    lines = [ln.rstrip() for ln in doc.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append("    " + ln)
        if len(out) >= n:
            out.append("    ...")
            break
    return "\n".join(out)


def _members(obj, limit: int = 40) -> str:
    try:
        names = [n for n in dir(obj) if not n.startswith("_")]
    except Exception:                                         # noqa: BLE001
        return ""
    if not names:
        return ""
    calls, other = [], []
    for n in names:
        try:
            v = getattr(obj, n)
        except Exception:                                     # noqa: BLE001
            other.append(n)
            continue
        (calls if callable(v) else other).append(n)
    parts = []
    if calls:
        shown = calls[:limit]
        parts.append("  callables: " + ", ".join(shown) + (f", ... (+{len(calls) - len(shown)})" if len(calls) > limit else ""))
    if other:
        shown = other[:limit]
        parts.append("  attributes: " + ", ".join(shown) + (f", ... (+{len(other) - len(shown)})" if len(other) > limit else ""))
    return "\n".join(parts)


def describe(obj, members: bool = True) -> str:
    kind = ("module" if isinstance(obj, types.ModuleType) else "class" if inspect.isclass(obj)
            else "function" if callable(obj) else type(obj).__name__)
    head = f"{_qualname(obj)}{_signature(obj) if callable(obj) and not isinstance(obj, types.ModuleType) else ''}"
    parts = [f"{head}   [{kind}]"]
    d = _doc_head(obj)
    if d:
        parts.append(d)
    if members and (inspect.isclass(obj) or isinstance(obj, types.ModuleType)):
        m = _members(obj)
        if m:
            parts.append(m)
    return "\n".join(parts)


def _resolve(name: str, frame=None):
    """'lifelines.CoxPHFitter.fit' / 'cph.fit' / 'np.linalg.lstsq' -> the object, without calling anything."""
    parts = [p for p in name.strip().split(".") if p]
    if not parts:
        raise LookupError(name)
    scopes = []
    if frame is not None:
        scopes = [frame.f_locals, frame.f_globals]
    import builtins as _b
    for scope in scopes + [vars(_b)]:
        if parts[0] in scope:
            obj = scope[parts[0]]
            for p in parts[1:]:
                obj = getattr(obj, p)
            return obj
    for i in range(len(parts), 0, -1):                      # longest importable module prefix
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:                                   # noqa: BLE001
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise LookupError(name)


def help_api(name_or_obj=None) -> None:
    """Print the signature, the head of the documentation and the public members of a function, class or module.

    help_api("scipy.optimize.curve_fit"); help_api("lifelines.CoxPHFitter.fit"); help_api(df.groupby); help_api("m.fit")
    (a dotted name is looked up in your session first, then imported). Nothing is called.
    """
    if name_or_obj is None:
        print(help_api.__doc__)
        return
    frame = sys._getframe(1)
    try:
        obj = _resolve(name_or_obj, frame) if isinstance(name_or_obj, str) else name_or_obj
    except Exception as exc:                                  # noqa: BLE001
        print(f"help_api: cannot resolve {name_or_obj!r} ({type(exc).__name__}: {exc})")
        return
    print(describe(obj))


# --------------------------------------------------------------------------- error notes
def _is_user(filename: str) -> bool:
    return filename.startswith("<cell") or filename.endswith("_episode.py")


def _callable_of_frame(frame):
    """The function object a library frame is executing, when it can be found without calling anything."""
    code = frame.f_code
    name = code.co_name
    if name in WRAPPER_NAMES or name.startswith("<"):
        return None
    loc = frame.f_locals
    for owner_name in ("self", "cls"):
        if owner_name in loc:
            owner = loc[owner_name]
            cls = owner if inspect.isclass(owner) else type(owner)
            for klass in getattr(cls, "__mro__", (cls,)):
                v = klass.__dict__.get(name)
                if v is not None:
                    return getattr(cls, name, v)
    v = frame.f_globals.get(name)
    if v is not None and callable(v):
        return v
    return None


def _safe_eval(node, frame):
    """Evaluate a Name/Attribute chain in the frame (no calls, no subscripts); Call(ClassName) -> the class itself."""
    if isinstance(node, ast.Name):
        for scope in (frame.f_locals, frame.f_globals, vars(__import__("builtins"))):
            if node.id in scope:
                return scope[node.id]
        raise LookupError(node.id)
    if isinstance(node, ast.Attribute):
        base = _safe_eval(node.value, frame)
        return getattr(base, node.attr)
    if isinstance(node, ast.Call):                          # `Cls(...).method` -> resolve on the class, do not call
        base = _safe_eval(node.func, frame)
        if inspect.isclass(base):
            return base
    raise LookupError(ast.dump(node)[:40])


def _calls_on_line(frame, lineno: int):
    src = linecache.getline(frame.f_code.co_filename, lineno)
    if not src.strip():
        return []
    try:
        tree = ast.parse(src.strip())
    except SyntaxError:
        # a continuation line of a multi-line statement: try the enclosing statement from the cached source
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                out.append(_safe_eval(node.func, frame))
            except Exception:                               # noqa: BLE001
                continue
    return out


def _name_in_msg(msg: str) -> str | None:
    m = re.match(r"^([\w.<>]+)\(\)", msg.strip())
    return m.group(1).split(".")[-1] if m else None


def explain_exception(etype, value, tb) -> str:
    """The API note for an exception raised by a cell ('' when there is nothing useful to add). Never raises."""
    try:
        return _explain(etype, value, tb)
    except Exception:                                         # noqa: BLE001
        return ""


def _explain(etype, value, tb) -> str:
    if etype is None or not issubclass(etype, (TypeError, AttributeError)):
        return ""
    msg = str(value)
    if issubclass(etype, TypeError) and not ARG_ERR.search(msg):
        return ""
    frames = []
    while tb is not None:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    users = [i for i, (f, _) in enumerate(frames) if _is_user(f.f_code.co_filename)]
    if not users:
        return ""
    ui = users[-1]
    uf, ul = frames[ui]
    notes = []
    raised_here = ui + 1 >= len(frames)                     # the exception was raised by the cell's own line
    attr_on_line = False
    if issubclass(etype, AttributeError) and getattr(value, "name", None):
        # `obj.name` written on the cell's line (pandas raises it from its own __getattr__, one frame down)
        src = linecache.getline(uf.f_code.co_filename, ul)
        attr_on_line = re.search(r"\.\s*" + re.escape(str(value.name)) + r"\b", src) is not None
    if issubclass(etype, AttributeError) and (raised_here or attr_on_line):
        obj = getattr(value, "obj", None)
        name = getattr(value, "name", None)
        if obj is not None and name:
            try:
                cands = [n for n in dir(obj) if not n.startswith("_")]
            except Exception:                                 # noqa: BLE001
                cands = []
            close = difflib.get_close_matches(name, cands, n=6, cutoff=0.5)
            low = name.lower()
            sub = [n for n in cands if low in n.lower() or n.lower() in low][:6]
            both = list(dict.fromkeys(close + sub))[:8]
            tname = _qualname(obj) if isinstance(obj, types.ModuleType) or inspect.isclass(obj) else \
                f"{type(obj).__module__}.{type(obj).__qualname__} object"
            line = f"{tname} has no attribute {name!r}."
            if both:
                notes.append(line + " Closest public names: " + ", ".join(both))
                for n in both[:1]:
                    try:
                        v = getattr(obj, n)
                    except Exception:                         # noqa: BLE001
                        continue
                    if callable(v):
                        notes.append(describe(v, members=False))
            else:
                notes.append(line)
                m = _members(obj, limit=30)
                if m:
                    notes.append(m)
    else:
        # a call made on the cell's line: the callable the message names, else the first public library frame
        want = _name_in_msg(msg) if issubclass(etype, TypeError) else None
        cands = _calls_on_line(uf, ul)
        pick = None
        if want:
            for c in cands:
                nm = getattr(c, "__name__", None)
                if nm == want or (inspect.isclass(c) and want == "__init__"):
                    pick = c
                    break
            if pick is None:
                for c in cands:                              # `Cls(...).fit` was resolved to the class
                    if inspect.isclass(c) and hasattr(c, want):
                        pick = getattr(c, want)
                        break
        if pick is None and not raised_here:
            for f, _ in frames[ui + 1:]:
                obj = _callable_of_frame(f)
                if obj is not None:
                    pick = obj
                    break
        if pick is None and len(cands) == 1:
            pick = cands[0]
        if pick is not None:
            notes.append(describe(pick, members=False))
    if not notes:
        return ""
    text = HEADER + "\n" + "\n".join(notes)
    if len(text) > MAX_NOTE:
        text = text[: MAX_NOTE - 20].rstrip() + "\n    ... [truncated]"
    return text


def install_excepthook() -> None:
    """Replay-sandbox mode: print the normal traceback, then the API note, for an exception that ends the script."""
    prev = sys.excepthook

    def hook(etype, value, tb):
        prev(etype, value, tb)
        note = explain_exception(etype, value, tb)
        if note:
            try:
                sys.stderr.write(note + "\n")
                sys.stderr.flush()
            except Exception:                                 # noqa: BLE001
                pass

    sys.excepthook = hook
