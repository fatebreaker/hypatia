"""H-GROUND-V2 (docs/RL_PLAN.md, registered 2026-10-04 14:38 UTC): an AUDIT-ONLY narrowing of the registered 'grounded'
rule's numeric-literal exclusion. The original rule (harness/process_signals.py, Analyzer.literals) discounts a printed
number that equals ANY numeric literal anywhere in the cell, which throws out correctly computed small integers (a
clinical score of 3 printed by code that also contains '3 <= age <= 14'). Here a printed number is discounted only if it
equals a number the cell TYPED as an answer: (a) passed directly to print() (also inside f-string fields and literal
dict/list displays there), (b) assigned directly to a name (name = 3, name = -3, or a literal display of numbers), or
(c) left as the bare last expression. Literals used in arithmetic or comparisons no longer discount a printed result.

Importing this module patches process_signals.Analyzer.run IN THIS PROCESS ONLY (the audit). It is never imported by the
trainer: the grounded REWARD of every RL arm keeps the original rule. Everything else -- which cells count as executed,
the evidence (STDOUT minus typed strings), the rounding- and sign-aware match, the SQL rule -- is unchanged.
"""
from __future__ import annotations

import ast

import process_signals as PS

V2 = True


def _num(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return float(n.value)
    if (isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)) and isinstance(n.operand, ast.Constant)
            and isinstance(n.operand.value, (int, float)) and not isinstance(n.operand.value, bool)):
        v = float(n.operand.value)
        return -v if isinstance(n.op, ast.USub) else v
    return None


def _typed(n, out: set):
    """Numbers written directly into a value: a bare number, a literal display (dict/list/tuple/set), an f-string's
    fields, or the arguments of a call wrapping one (json.dumps({...}), round(3.0), str(3))."""
    v = _num(n)
    if v is not None:
        out.add(v); out.add(abs(v)); return
    if isinstance(n, ast.Dict):
        for x in n.values:
            if x is not None:
                _typed(x, out)
    elif isinstance(n, (ast.List, ast.Tuple, ast.Set)):
        for x in n.elts:
            _typed(x, out)
    elif isinstance(n, ast.JoinedStr):
        for x in n.values:
            if isinstance(x, ast.FormattedValue):
                _typed(x.value, out)
    elif isinstance(n, ast.Call):
        for a in n.args:
            _typed(a, out)


def typed_number_literals(tree) -> set:
    out: set = set()
    if tree is None:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print":
            for a in n.args:
                _typed(a, out)
        elif isinstance(n, ast.Assign):
            _typed(n.value, out)
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            _typed(n.value, out)
    body = getattr(tree, "body", [])
    if body and isinstance(body[-1], ast.Expr):
        _typed(body[-1].value, out)
    return out


_orig_run = PS.Analyzer.run


def _run_v2(self, cells):
    res = _orig_run(self, cells)
    for c in cells:
        if getattr(c, "tree", None) is not None:
            self.literals[c.idx] = typed_number_literals(c.tree)
    return res


PS.Analyzer.run = _run_v2


# H-GROUND-GR (docs/RL_PLAN.md, registered 2026-10-06 15:38 UTC; SQL-NO-CONSTANT read NOT MET -> its registered consequence):
# a submitted SQL query with no FROM clause -- a constant query such as 'SELECT 1', or julianday() of timestamps typed in
# from earlier output -- is NOT grounded, even though it ran. Audit only, like the rest of this module: rl/grpo_rg.py's
# training reward is unchanged; this patches rl.grpo_rg.sql_grounded IN THIS PROCESS ONLY (RG.grounded looks it up at call time).
import re as _re
from rl import grpo_rg as _RG
_FROM = _re.compile(r"\bfrom\b", _re.I)
_orig_sql_grounded = _RG.sql_grounded


def constant_queries(answer) -> list:
    qs = {k: v for k, v in _RG.answer_obj(answer).items()
          if isinstance(v, str) and v.strip() and v.strip().lower() not in ("null", "none")}
    return sorted(k for k, q in qs.items() if not _FROM.search(q))


def _sql_grounded_v21(messages, answer):
    fired, why = _orig_sql_grounded(messages, answer)
    if fired == 1.0:
        const = constant_queries(answer)
        if const:
            return 0.0, f"constant query (no FROM clause): {const}"
    return fired, why


_RG.sql_grounded = _sql_grounded_v21
