"""HARNESS V3.1 PROCESS SIGNALS -- the `reest` detector (docs/RL_PLAN.md s.18, entry of 2026-09-26 20:03 UTC, directly
below the 19:15 entry, which is why it exists). harness/process_signals.py is RUNNING (the s.16 arms and every v3
evaluation import it) and stays byte-identical: this file SUBCLASSES it and is used ONLY by the v3.1 ELICIT arms
(rl/grpo_v31.py) and their reader.

WHY. The science_check prompt ELICITS second estimates on causal_rdd -- a bandwidth change, a kernel after an OLS, an
added covariate -- but process_signals' `spec` / M3 `check` credit a later-cell re-estimate only when the estimator family
(as it classifies it), the weights or an explicit transform changes, and do not see KernelReg, lowess or a
difference-in-means at all. The v3.1 process term would therefore pay exactly what the s.16 PR arm's pays (19:15 UTC).

THE SIGNAL. reest = 1 iff the rollout executed, in a LATER cell than its first successful estimate of a response Y, a
SECOND estimate of the SAME response Y whose cell ran to the end and printed a number, with a specification that no
earlier successful estimate of Y had, differing in at least one of
    family   estimator family: OLS (sm.OLS/smf.ols/LinearRegression/linregress/lstsq/np.polyfit), robust (RLM, Huber,
             Theil-Sen), GLM (+ its family), quantile, kernel (KernelReg, + reg_type / bw), lowess (+ frac),
             difference-in-means (two .mean()s of Y subtracted, ttest_ind), rdrobust, curve_fit (+ the model function),
             the survival / classifier families of process_signals; WLS = OLS + weights (the weights are part of the spec)
    terms    the regressor set: each regressor as a CANONICAL TERM -- a real column, or a derived one resolved to its
             definition (c = x - 60 is x; d = (x >= 60) is ind(x); c**2, I(c**2), np.power(c, 2) and a derived c2 column
             are all pow2(x); d:c, d*c and a derived d*c column are all mul(ind(x), x); C(g) and get_dummies(g) are
             cat(g)); the intercept is the term "1" (add_constant, formulas, sklearn's default)
    form     the response's transform (log y vs y) and the model function / degree (a polyfit of degree 2 is OLS on
             {1, x, pow2(x)})
    sample   the row restriction on the data the fit used: a boolean mask / .loc / .query / dropna / head on the frame,
             carried through derived frames; constants substituted and affine comparisons normalised, so a +-10 window
             and a +-5 window differ while 55 <= x <= 65, -5 <= x - 60 <= 5, abs(x - 60) <= 5 and c.between(-5, 5)
             (c = x - 60) are one window
An estimate under an `if` / `while`, or in a `try` body that has an except handler, is CONDITIONAL and never counts
(which branch ran is not visible, and a caught exception leaves the cell "ok" although the fit never ran); a name the
branches leave different gets a deterministic merged value. A "successful estimate" ran and was followed by printed
output: its cell ran to the end and printed a number, or its cell raised LATER than the estimate's top-level statement
with a print() between the estimate and the failure (the failing statement is located from the traceback's
`File "<cell-N>", line L, in <module>`, matching the executed text -- the notebook echo rewrites the cell -- and its
source line). NOT credited: a re-run of the same fit (same signature in a later cell, however re-typed: another API,
reloaded data, renamed frames, a patsy term vs a derived column), a fit whose only predecessor is an ERRORED attempt (a
fit after an error of the same fit), two fits inside ONE cell, a different response, or a different estimand class (a
Cox model vs a KM curve). Nothing is read from prose or comments; only executed cells, their STDOUT/STDERR and the
task's data columns (process_signals' Vocab).

WHAT IT CANNOT SEE (stated): fits inside user-defined functions (process_signals skips function bodies); a
difference-in-means written through groupby(...).mean().diff() or by indexing a groupby result; two masks that select
the same rows in non-affine ways (isin vs a range) read as a sample change (rare; the model did write a new restriction).

THE SCORER. ProcessScorerV31(ProcessScorer).signals() returns the PARENT's signals (inspect, strata, grounded, the other
trap modes: byte-for-byte the same computation) with
    spec   := reest      (detail: "reest <0|1>: <where> [v3 spec <x>: <its where> | v3 check <y>]")
    check  := the parent's check with its M3_functional mode set to max(parent M3, reest), re-averaged over the modes
              (families whose trap modes include M3; every other family keeps the parent's check exactly)
so p = mean over applicable signals and rl/grpo_v3.py's reward composition (compose, the sham permutation, the cap) are
unchanged. The old values ride along in the spec detail (the only per-signal text process.jsonl stores), so the reader
can print them beside.

    from rl.process_signals_v31 import ProcessScorerV31
    ps = ProcessScorerV31()
    sig = ps.signals(messages, cfg, data_dir, answer=..., n_prefix=...)   # sig.fired["spec"] is reest
    ps.reest(messages, cfg, data_dir, n_prefix)                          # (value, where, pairs)
"""
from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "harness") not in sys.path:
    sys.path.insert(0, str(ROOT / "harness"))
import process_signals as PS  # noqa: E402  (the SAME module object rl/grpo_v3.py imports)
from process_signals import (Analyzer, ProcessScorer, ProcessSignals, _fname, estimand_class,  # noqa: E402
                             executed_cells, num_tokens)

DETECTOR = "reest-v2"
TAG = "reest"

# --------------------------------------------------------------------------- estimator vocabulary (families)
SM_FAMILY = {"OLS": "ols", "WLS": "ols", "GLS": "gls", "GLSAR": "gls", "GLM": "glm", "Logit": "logit", "Probit": "probit",
             "Poisson": "poisson", "NegativeBinomial": "negbin", "RLM": "robust", "QuantReg": "quantreg", "PHReg": "cox",
             "MixedLM": "mixed", "IV2SLS": "iv", "IVGMM": "iv", "MNLogit": "logit", "OrderedModel": "ordinal"}
FORMULA_FAMILY = {"ols": "ols", "wls": "ols", "gls": "gls", "glsar": "gls", "glm": "glm", "logit": "logit",
                  "probit": "probit", "poisson": "poisson", "negativebinomial": "negbin", "rlm": "robust",
                  "quantreg": "quantreg", "phreg": "cox", "mixedlm": "mixed", "mnlogit": "logit"}
SK_FAMILY = {"LinearRegression": "ols", "Ridge": "ridge", "Lasso": "lasso", "ElasticNet": "enet",
             "HuberRegressor": "robust", "TheilSenRegressor": "robust", "RANSACRegressor": "robust",
             "LogisticRegression": "logit", "PoissonRegressor": "poisson", "GammaRegressor": "glm",
             "QuantileRegressor": "quantreg", "KNeighborsRegressor": "knn", "GaussianProcessRegressor": "gp",
             "RandomForestRegressor": "rf", "GradientBoostingRegressor": "gbm", "SVR": "svr",
             "DecisionTreeRegressor": "tree", "CoxPHFitter": "cox", "KaplanMeierFitter": "km", "WeibullFitter": "weibull",
             "WeibullAFTFitter": "weibull", "NelsonAalenFitter": "km", "ExponentialFitter": "expfit",
             "LogNormalFitter": "lognorm"}
PARAM_KW = {"kernel": ("reg_type", "bw", "ckertype", "defaults"), "lowess": ("frac", "it", "delta"),
            "rdrobust": ("c", "h", "b", "p", "q", "kernel", "bwselect"), "glm": ("family",), "robust": ("M",),
            "quantreg": ("q",), "ridge": ("alpha",), "lasso": ("alpha",), "enet": ("alpha", "l1_ratio"),
            "knn": ("n_neighbors",), "iv": ("instrument",)}
PARAM_DEFAULT = {("kernel", "reg_type"): "'ll'", ("kernel", "bw"): "'cv_ls'", ("lowess", "frac"): "0.666667",
                 ("lowess", "it"): "3"}
PASS_METHODS = {"astype", "copy", "values", "to_numpy", "reshape", "squeeze", "ravel", "flatten", "to_frame",
                "reset_index", "sort_index", "sort_values", "array", "asarray", "Series", "float", "int", "tolist",
                "center", "standardize", "scale", "infer_objects", "clip"}
FRAME_PASS = PASS_METHODS | {"dropna", "fillna", "assign", "rename", "drop"}
READERS = {"read_csv", "read_table", "read_excel", "read_parquet", "read_json", "read_fwf", "read_feather",
           "read_pickle", "read_hdf"}
ROWSEL = {"head", "tail", "sample", "nlargest", "nsmallest"}
TRANSFORM = {"log": "log", "log10": "log", "log2": "log", "log1p": "log", "exp": "exp", "expm1": "exp",
             "sqrt": "sqrt", "reciprocal": "inv", "arcsinh": "asinh", "cbrt": "cbrt"}
CMP = {ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne", ast.In: "in",
       ast.NotIn: "notin", ast.Is: "is", ast.IsNot: "isnot"}
MODULES = {"np", "pd", "sm", "smf", "stats", "math", "scipy", "sp", "numpy", "pandas", "statsmodels", "plt"}
CANON_MAX = 240                       # longer canonical strings are hashed
CELL_TB = re.compile(r'File "<cell-\d+>", line (\d+), in <module>\n([^\n]*)')


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.6g}"


def _cap(s: str) -> str:
    """A canonical string longer than CANON_MAX becomes a short deterministic hash of itself."""
    return s if len(s) <= CANON_MAX else "h:" + hashlib.sha1(s.encode()).hexdigest()[:12]


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


@dataclass
class EstR:
    """One estimate as the re-estimate rule sees it."""
    cell: int
    top: int                                  # index of the top-level statement of its cell that holds it
    family: str
    dep: frozenset                            # the response: real data columns (process_signals' resolution)
    ycanon: str                               # the response as written, canonically (its transform)
    terms: frozenset = frozenset()            # canonical regressor terms
    sample: frozenset = frozenset()           # canonical row restrictions
    params: tuple = ()                        # estimate-changing arguments (bandwidth, degree, family, weights, fn)
    call: str = ""
    cond: bool = False                        # inside an if / while / try-with-handlers: may not have run

    def signature(self):
        return (self.family, self.ycanon, self.terms, self.sample, self.params)

    def differs(self, o: "EstR") -> list:
        out = []
        if self.family != o.family:
            out.append(f"family {o.family}->{self.family}")
        if self.terms != o.terms:
            add, rem = sorted(self.terms - o.terms), sorted(o.terms - self.terms)
            out.append("terms" + (f" +{add}" if add else "") + (f" -{rem}" if rem else ""))
        if self.ycanon != o.ycanon:
            out.append(f"response form {o.ycanon}->{self.ycanon}")
        if self.sample != o.sample:
            out.append(f"sample {sorted(o.sample) or 'all rows'}->{sorted(self.sample) or 'all rows'}")
        if self.params != o.params:
            out.append(f"params {dict(o.params)}->{dict(self.params)}")
        return out


# --------------------------------------------------------------------------- the analyzer
class ReestAnalyzer(Analyzer):
    """process_signals.Analyzer (its column resolution, dataflow and walk order), plus: canonical terms of every
    regressor, the row restriction of every frame, numeric / string constants, and an EstR for every estimate of the
    families above (self.rests)."""

    def __init__(self, vocab):
        super().__init__(vocab)
        self.consts: dict = {}       # name -> float
        self.strs: dict = {}         # name -> str
        self.strlists: dict = {}     # name -> [str]
        self.ncanon: dict = {}       # name -> canonical expression (term mode)
        self.ncanon_m: dict = {}     # name -> canonical expression (mask mode: constants kept)
        self.dcanon: dict = {}       # column name (derived, or real rewritten) -> canonical definition (term mode)
        self.dcanon_m: dict = {}
        self.tset: dict = {}         # name -> frozenset of terms (design matrices)
        self.samp: dict = {}         # name -> frozenset of row restrictions
        self.skind: dict = {}        # name -> estimator family of an unfitted estimator object (sklearn / lifelines)
        self.fbody: dict = {}        # function name -> canonical body
        self.rests: list = []
        self.daff: dict = {}         # column -> (core, a, b): its affine definition (mask mode)
        self.naff: dict = {}         # name -> (core, a, b)
        self.polydeg: dict = {}      # name -> PolynomialFeatures degree (and include_bias)
        self._top: dict = {}         # id(ast node) -> top-level statement index in its cell
        self._augs = 0
        self._cond = 0               # > 0 while walking code whose execution is conditional

    # ------------------------------------------------------------------ per-cell setup (hooked on the top-level body)
    def _stmts(self, c, body, loop_groups):
        if c.tree is not None and body is c.tree.body:
            for i, st in enumerate(body):
                for n in ast.walk(st):
                    self._top[id(n)] = i
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    b = copy.deepcopy(st)
                    b.name = "_"
                    self.fbody[st.name] = "fn:" + hashlib.sha256(ast.dump(b).encode()).hexdigest()[:10]
            for n in ast.walk(c.tree):
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda) and len(n.targets) == 1 \
                        and isinstance(n.targets[0], ast.Name):
                    self.fbody[n.targets[0].id] = "fn:" + hashlib.sha256(ast.dump(n.value).encode()).hexdigest()[:10]
        for st in body:
            if isinstance(st, (ast.If, ast.While)):
                # which branch ran is not visible: estimates inside are CONDITIONAL (never counted), and a name or
                # column the branches leave different gets a deterministic merged value (a re-run stays equal)
                self._calls(c, st.test, loop_groups)
                before = self._snapshot()
                self._cond += 1
                self._stmts(c, st.body, loop_groups)
                after_body = self._snapshot()
                self._restore(before)
                self._stmts(c, st.orelse, loop_groups)
                self._cond -= 1
                self._merge(after_body, self._snapshot())
            elif isinstance(st, ast.Try) and st.handlers:
                # a caught exception leaves the cell 'ok' although the fit never ran: the try body is conditional too
                before = self._snapshot()
                self._cond += 1
                self._stmts(c, st.body, loop_groups)
                after_body = self._snapshot()
                self._restore(before)
                for h in st.handlers:
                    self._stmts(c, h.body, loop_groups)
                self._cond -= 1
                self._merge(after_body, self._snapshot())
                self._stmts(c, st.orelse, loop_groups)
                self._stmts(c, st.finalbody, loop_groups)
            else:
                super()._stmts(c, [st], loop_groups)

    STATE = ("consts", "strs", "strlists", "ncanon", "ncanon_m", "dcanon", "dcanon_m", "tset", "samp", "skind",
             "daff", "naff", "polydeg")

    def _snapshot(self) -> dict:
        return {k: dict(getattr(self, k)) for k in self.STATE}

    def _restore(self, snap: dict) -> None:
        for k, v in snap.items():
            setattr(self, k, dict(v))

    def _merge(self, a: dict, b: dict) -> None:
        """State after `if: A else: B`: keys equal in both keep their value; a key the branches leave different is
        a deterministic merge of both (strings, sets) or unknown (numbers, estimator objects)."""
        for k in self.STATE:
            da, db, out = a[k], b[k], {}
            for key in set(da) | set(db):
                va, vb = da.get(key), db.get(key)
                if va == vb:
                    out[key] = va
                elif k in ("ncanon", "ncanon_m", "dcanon", "dcanon_m"):
                    out[key] = _cap("cond(" + "|".join(sorted([str(va), str(vb)])) + ")")
                elif k in ("tset", "samp"):
                    out[key] = frozenset({_cap("cond(" + "|".join(sorted([",".join(sorted(va or ())),
                                                                          ",".join(sorted(vb or ()))])) + ")")})
            setattr(self, k, out)

    # ------------------------------------------------------------------ constants
    def _const(self, node):
        """Numeric value of a constant expression (literals and known numeric names), else None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in self.consts:
            return self.consts[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = self._const(node.operand)
            return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
        if isinstance(node, ast.BinOp):
            a, b = self._const(node.left), self._const(node.right)
            if a is None or b is None:
                return None
            try:
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b
                if isinstance(node.op, ast.Pow):
                    return float(a ** b)
                if isinstance(node.op, ast.FloorDiv):
                    return float(a // b)
            except (ZeroDivisionError, OverflowError, ValueError, TypeError):
                return None
        return None

    def _str(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.strs:
            return self.strs[node.id]
        if isinstance(node, ast.JoinedStr):
            out = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    s = self._str(v.value)
                    if s is None:
                        k = self._const(v.value)
                        if k is None:
                            return None
                        s = _num(k)
                    out.append(s)
                else:
                    return None
            return "".join(out)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = self._str(node.left), self._str(node.right)
            return None if a is None or b is None else a + b
        return None

    # ------------------------------------------------------------------ columns
    def _colsel(self, sl):
        """Column names a subscript key SELECTS (df['a'], df[['a', 'b']], df[cols]); None if it is not a selection."""
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:            # .loc[rows, cols]
            return self._colsel(sl.elts[1])
        s = self._str(sl)
        if s is not None:
            return [s]
        if isinstance(sl, ast.Name) and sl.id in self.strlists:
            return list(self.strlists[sl.id])
        if isinstance(sl, (ast.List, ast.Tuple)) and sl.elts:
            names = [self._str(e) for e in sl.elts]
            if all(n is not None for n in names):
                return names
        return None

    def col_canon(self, s: str, mask: bool = False) -> str:
        d = (self.dcanon_m if mask else self.dcanon).get(s)
        if d is not None:
            return d
        return ("c:" if s in self.cols else "c?:") + s

    # ------------------------------------------------------------------ canonical expressions
    def canon(self, node, mask: bool = False, bind: dict | None = None, fnames: bool = False) -> str:
        """_canon, with a long result replaced by a short deterministic hash of it (x = f(x, x) re-assigned in a loop
        would otherwise grow the string exponentially; equal expressions still get equal strings)."""
        return _cap(self._canon(node, mask, bind, fnames))

    def _canon(self, node, mask: bool = False, bind: dict | None = None, fnames: bool = False) -> str:
        """A canonical string for an expression. TERM mode (mask=False) is invariant to shifting / scaling by a constant
        and to how an indicator is written ((x >= 60).astype(int), np.where(x > 60, 1, 0), x.apply(lambda v: 1 if v > 60
        else 0) are all ind(c:x)); MASK mode keeps constants and comparison directions (a +-10 window != a +-5 one)."""
        bind = bind or {}

        def C(n):
            return self.canon(n, mask, bind, fnames)
        if node is None:
            return "_"
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or v is None:
                return str(v)
            if isinstance(v, (int, float)):
                return _num(v)
            if isinstance(v, str) and (v in self.cols or v in self.dcanon):
                return self.col_canon(v, mask)
            return repr(v)
        if isinstance(node, ast.Name):
            if node.id in bind:
                return bind[node.id]
            if node.id in self.consts:
                return _num(self.consts[node.id])
            if fnames and (node.id in self.cols or node.id in self.dcanon):
                return self.col_canon(node.id, mask)
            m = (self.ncanon_m if mask else self.ncanon).get(node.id)
            if m is not None:
                return m
            if node.id in self.strs and self.strs[node.id] in self.cols:
                return self.col_canon(self.strs[node.id], mask)
            return "v:" + node.id
        k = self._const(node)
        if k is not None:
            return _num(k)
        if isinstance(node, ast.Subscript):
            names = self._colsel(node.slice)
            if names is not None and len(names) == 1:
                return self.col_canon(names[0], mask)
            if names:
                return "cols(" + ",".join(sorted(self.col_canon(n, mask) for n in names)) + ")"
            if isinstance(node.value, ast.Attribute) and node.value.attr in ("c_", "r_"):
                return "stack(" + C(node.slice) + ")"
            return C(node.value)                  # a row-filtered / positional selection of the same quantity
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id not in MODULES and \
                    (node.attr in self.cols or node.attr in self.dcanon):
                return self.col_canon(node.attr, mask)
            if node.attr in ("values", "T", "loc", "iloc", "str", "dt", "array"):
                return C(node.value)
            base = node.value.id if isinstance(node.value, ast.Name) and node.value.id in MODULES else C(node.value)
            return base + "." + node.attr
        if isinstance(node, ast.Call):
            f = _fname(node.func)
            recv = node.func.value if isinstance(node.func, ast.Attribute) else None
            if isinstance(recv, ast.Name) and recv.id in MODULES:
                recv = None
            args = node.args
            if f in PASS_METHODS or f == "add_constant":      # (terms() adds the intercept term itself)
                if recv is not None:
                    return C(recv)
                if args:
                    return C(args[0])
            if f == "where" and len(args) == 3:
                if self._const(args[1]) is not None and self._const(args[2]) is not None:
                    return C(args[0])
                return f"where({C(args[0])},{C(args[1])},{C(args[2])})"
            if f in ("apply", "map") and recv is not None and args and isinstance(args[0], ast.Lambda) \
                    and args[0].args.args:
                nb = dict(bind)
                nb[args[0].args.args[0].arg] = C(recv)
                return self.canon(args[0].body, mask, nb, fnames)
            if f == "power" and len(args) >= 2 and self._const(args[1]) is not None:
                return f"pow{_num(self._const(args[1]))}({C(args[0])})"
            if f == "square" and (args or recv is not None):
                return f"pow2({C(args[0] if args else recv)})"
            if f in TRANSFORM and (args or recv is not None):
                return f"{TRANSFORM[f]}({C(args[0] if args else recv)})"
            if f in ("abs", "absolute", "fabs") and (args or recv is not None):
                return f"abs({C(args[0] if args else recv)})"
            if f in ("ones", "ones_like"):
                return "1"
            if f == "between" and recv is not None and len(args) >= 2:
                if not mask:
                    return f"ind({C(recv)})"
                lo, hi = self._const(args[0]), self._const(args[1])
                base, a, b = self.affine(recv, bind, fnames)
                if lo is not None and hi is not None and a:
                    ends = sorted([(lo - b) / a, (hi - b) / a])
                    return _andor("and", [f"ge({base},{_num(ends[0])})", f"le({base},{_num(ends[1])})"])
                return f"between({C(recv)},{C(args[0])},{C(args[1])})"
            if f == "isin" and recv is not None:
                return f"isin({C(recv)},{C(args[0]) if args else '_'})" if mask else f"ind({C(recv)})"
            parts = [C(a) for a in args] + [f"{kw.arg}={C(kw.value)}" for kw in node.keywords if kw.arg]
            return f"{f}(" + (C(recv) + ";" if recv is not None else "") + ",".join(parts) + ")"
        if isinstance(node, ast.BinOp):
            lc, rc = self._const(node.left) is not None, self._const(node.right) is not None
            if not mask and (lc != rc):
                other = node.right if lc else node.left
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    return C(other)                                     # a shift is the same regressor
                if isinstance(node.op, ast.Mult):
                    return C(other)                                     # so is a rescaling
                if isinstance(node.op, ast.Div):
                    return f"inv({C(node.right)})" if lc else C(node.left)
            if isinstance(node.op, ast.Pow) and rc:
                return f"pow{_num(self._const(node.right))}({C(node.left)})"
            if isinstance(node.op, ast.Mult):
                return "mul(" + ",".join(sorted([C(node.left), C(node.right)])) + ")"
            if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                return _andor("and" if isinstance(node.op, ast.BitAnd) else "or", [C(node.left), C(node.right)])
            return f"{type(node.op).__name__}({C(node.left)},{C(node.right)})"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Invert) or isinstance(node.op, ast.Not):
                return _negate(C(node.operand))
            if isinstance(node.op, ast.USub) and mask:
                return f"neg({C(node.operand)})"
            return C(node.operand)
        if isinstance(node, ast.Compare):
            if not mask:                                                # an indicator, however written
                side = node.left if self._const(node.left) is None else node.comparators[0]
                return f"ind({C(side)})"
            parts, left = [], node.left
            for op, r in zip(node.ops, node.comparators):
                parts.append(self._cmp_mask(left, op, r, bind, fnames))
                left = r
            return parts[0] if len(parts) == 1 else _andor("and", parts)
        if isinstance(node, ast.BoolOp):
            return _andor("and" if isinstance(node.op, ast.And) else "or", [C(v) for v in node.values])
        if isinstance(node, ast.IfExp):
            if self._const(node.body) is not None and self._const(node.orelse) is not None:
                return C(node.test) if mask else f"ind({self._ind_side(node.test, bind, fnames)})"
            return f"ifexp({C(node.test)},{C(node.body)},{C(node.orelse)})"
        if isinstance(node, (ast.List, ast.Tuple)):
            return "[" + ",".join(C(e) for e in node.elts) + "]"
        return type(node).__name__

    def affine(self, node, bind=None, fnames=False) -> tuple:
        """(core, a, b) with node == a * core + b, core a mask-mode canonical string (constants folded through +, -,
        *, / by a constant, unary minus, and derived columns / names defined that way); (canon, 1, 0) otherwise."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            kl, kr = self._const(node.left), self._const(node.right)
            if (kl is None) != (kr is None):
                core, a, b = self.affine(node.right if kl is not None else node.left, bind, fnames)
                k = kl if kl is not None else kr
                if isinstance(node.op, ast.Add):
                    return core, a, b + k
                if isinstance(node.op, ast.Sub):
                    return (core, a, b - k) if kr is not None else (core, -a, k - b)
                if isinstance(node.op, ast.Mult):
                    return core, a * k, b * k
                if isinstance(node.op, ast.Div) and kr:
                    return core, a / k, b / k
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            core, a, b = self.affine(node.operand, bind, fnames)
            return core, -a, -b
        name = None
        if isinstance(node, ast.Subscript):
            sel = self._colsel(node.slice)
            name = ("col", sel[0]) if sel and len(sel) == 1 else None
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id not in MODULES:
            name = ("col", node.attr)
        elif isinstance(node, ast.Name) and not (bind and node.id in bind):
            name = ("col", node.id) if fnames and (node.id in self.cols or node.id in self.dcanon) else ("name", node.id)
        if name is not None:
            aff = (self.daff if name[0] == "col" else self.naff).get(name[1])
            if aff is not None:
                return aff
        if isinstance(node, (ast.Subscript, ast.Attribute, ast.Name, ast.Call)) and \
                not (isinstance(node, ast.Call) and _fname(node.func) not in PASS_METHODS):
            if isinstance(node, ast.Call):
                recv = node.func.value if isinstance(node.func, ast.Attribute) else None
                inner = recv if recv is not None and not (isinstance(recv, ast.Name) and recv.id in MODULES) \
                    else (node.args[0] if node.args else None)
                if inner is not None:
                    return self.affine(inner, bind, fnames)
        return self.canon(node, True, bind, fnames), 1.0, 0.0

    def _cmp_mask(self, left, op, right, bind, fnames) -> str:
        """One comparison in MASK mode, normalised: `core op value` for an affine side (x - 60 >= -5 is ge(x,55)),
        abs(affine) <= h as a two-sided range, > / >= merged (ge) and < / <= merged (le), an indicator == 1 as itself."""
        kl, kr = self._const(left), self._const(right)
        o = CMP.get(type(op), "cmp")
        if (kl is None) == (kr is None) or o not in ("gt", "ge", "lt", "le", "eq", "ne"):
            return f"{o}({self.canon(left, True, bind, fnames)},{self.canon(right, True, bind, fnames)})"
        side, k = (left, kr) if kr is not None else (right, kl)
        if kl is not None:                                           # 55 <= x  ->  x >= 55
            o = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge"}.get(o, o)
        o = {"gt": "ge", "lt": "le"}.get(o, o)
        if isinstance(side, ast.Call) and _fname(side.func) in ("abs", "absolute", "fabs") and o in ("le", "ge"):
            inner = side.args[0] if side.args else (side.func.value if isinstance(side.func, ast.Attribute) else None)
            if inner is not None:
                core, a, b = self.affine(inner, bind, fnames)
                if a:
                    ends = sorted([(-k - b) / a, (k - b) / a])
                    if o == "le":
                        return _andor("and", [f"ge({core},{_num(ends[0])})", f"le({core},{_num(ends[1])})"])
                    return _andor("or", [f"le({core},{_num(ends[0])})", f"ge({core},{_num(ends[1])})"])
        core, a, b = self.affine(side, bind, fnames)
        if not a:
            return f"{o}({self.canon(side, True, bind, fnames)},{_num(k)})"
        v = (k - b) / a
        if a < 0:
            o = {"ge": "le", "le": "ge"}.get(o, o)
        if o in ("eq", "ne") and _is_indicator(core) and v in (0.0, 1.0):
            return core if (o == "eq") == (v == 1.0) else _negate(core)
        return f"{o}({core},{_num(v)})"

    def _ind_side(self, test, bind, fnames) -> str:
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            side = test.left if self._const(test.left) is None else test.comparators[0]
            return self.canon(side, False, bind, fnames)
        return self.canon(test, False, bind, fnames)

    # ------------------------------------------------------------------ regressor terms
    def terms(self, node) -> frozenset:
        """Canonical regressor terms of a design matrix / regressor expression."""
        if node is None:
            return frozenset()
        if isinstance(node, ast.Name):
            if node.id in self.tset:
                return self.tset[node.id]
            return frozenset({self.canon(node)})
        if isinstance(node, (ast.List, ast.Tuple)):
            out = frozenset()
            for e in node.elts:
                out |= self.terms(e)
            return out
        if isinstance(node, ast.Subscript):
            names = self._colsel(node.slice)
            if names:
                return frozenset(self.col_canon(n) for n in names)
            if isinstance(node.value, ast.Attribute) and node.value.attr in ("c_", "r_"):
                return self.terms(node.slice)
            return self.terms(node.value)
        if isinstance(node, ast.Attribute):
            if node.attr in ("values", "T", "loc", "iloc", "array"):
                return self.terms(node.value)
            return frozenset({self.canon(node)})
        if isinstance(node, ast.Call):
            f = _fname(node.func)
            recv = node.func.value if isinstance(node.func, ast.Attribute) else None
            if isinstance(recv, ast.Name) and recv.id in MODULES:
                recv = None
            args = node.args
            if f == "add_constant":
                return self.terms(args[0] if args else None) | {"1"}
            if f in ("column_stack", "hstack", "concat", "vstack", "stack", "DataFrame") and args:
                return self.terms(args[0]) if f != "DataFrame" else self._frame_terms(args[0])
            if f in ("ones", "ones_like"):
                return frozenset({"1"})
            if f == "get_dummies" and args:
                return frozenset({f"cat({self.canon(args[0])})"})
            if f == "fit_transform" and recv is not None:
                pf = self._polyfeat(recv) if isinstance(recv, ast.Call) else \
                    (self.polydeg.get(recv.id) if isinstance(recv, ast.Name) else None)
                base = self.terms(args[0] if args else None)
                core = base - {"1"}
                if pf is not None and pf[0] and len(core) == 1:
                    b = next(iter(core))
                    return frozenset({b} | {f"pow{_num(j)}({b})" for j in range(2, int(pf[0]) + 1)}
                                     | ({"1"} if pf[1] else set()))
            if f == "drop" and recv is not None:
                gone = set()
                for a in list(args) + [k.value for k in node.keywords if k.arg in ("columns", "labels")]:
                    gone |= {self.col_canon(n) for n in (self._colsel(a) or [])}
                return self.terms(recv) - gone
            if f == "assign" and recv is not None:
                return self.terms(recv) | {self.canon(k.value) for k in node.keywords if k.arg}
            if f in PASS_METHODS | {"dropna", "fillna"}:
                if recv is not None:
                    return self.terms(recv)
                if args:
                    return self.terms(args[0])
        return frozenset({self.canon(node)})

    def _polyfeat(self, value):
        """(degree, include_bias) of a PolynomialFeatures(...) constructor call, else None."""
        if isinstance(value, ast.Call) and _fname(value.func) == "PolynomialFeatures":
            kw = {k.arg: k.value for k in value.keywords if k.arg}
            deg = self._const(value.args[0] if value.args else kw.get("degree", ast.Constant(2)))
            bias = not ("include_bias" in kw and self._const(kw["include_bias"]) == 0)
            return (deg, bias)
        return None

    def _frame_terms(self, node) -> frozenset:
        if isinstance(node, ast.Dict):
            return frozenset(self.canon(v) for v in node.values)
        return self.terms(node)

    # ------------------------------------------------------------------ row restrictions
    def _mask_key(self, node, how: str = "m") -> str:
        return f"{how}:" + self.canon(node, mask=True)

    def sample_of(self, node) -> frozenset:
        if node is None:
            return frozenset()
        if isinstance(node, ast.Name):
            return self.samp.get(node.id, frozenset())
        if isinstance(node, ast.Subscript):
            base, acc = node.value, None
            if isinstance(base, ast.Attribute) and base.attr in ("loc", "iloc", "at", "iat"):
                acc, base = base.attr, base.value
            s = self.sample_of(base)
            sl = node.slice
            if acc is None and self._colsel(sl) is not None:
                return s
            rows = sl.elts[0] if (acc and isinstance(sl, ast.Tuple) and sl.elts) else sl
            if isinstance(rows, ast.Slice) and rows.lower is None and rows.upper is None and rows.step is None:
                return s
            if isinstance(rows, ast.Name) and rows.id in self.strs:
                return s
            return s | {self._mask_key(rows, acc or "m")}
        if isinstance(node, ast.Attribute):
            return self.sample_of(node.value)
        if isinstance(node, ast.Call):
            f = _fname(node.func)
            recv = node.func.value if isinstance(node.func, ast.Attribute) else None
            if isinstance(recv, ast.Name) and recv.id in MODULES:
                recv = None
            if f in READERS:
                return frozenset()
            base = self.sample_of(recv) if recv is not None else frozenset()
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if recv is not None and f == "query" and node.args:
                q = self._str(node.args[0])
                return base | {"q:" + (re.sub(r"\s+", "", q) if q is not None else self.canon(node.args[0], True))}
            if recv is not None and f == "dropna":
                sub = kw.get("subset")
                cols = self._colsel(sub) if sub is not None else None
                return base | {"dropna:" + (",".join(sorted(self.col_canon(c, True) for c in cols)) if cols else "*")}
            if recv is not None and f in ROWSEL:
                return base | {f"{f}:" + ",".join(self.canon(a, True) for a in node.args)}
            out = base
            for a in list(node.args) + list(kw.values()):
                out |= self.sample_of(a)
            return out
        out = frozenset()
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.expr):
                out |= self.sample_of(ch)
        return out

    # ------------------------------------------------------------------ dataflow (assignments, in-place calls)
    def _assign_target(self, t, value, aug: bool = False):
        self._track(t, value, aug)
        super()._assign_target(t, value, aug)

    def _track(self, t, value, aug):
        if isinstance(t, ast.Name):
            k = self._const(value)
            s = self._str(value)
            for d in (self.consts, self.strs, self.strlists):
                d.pop(t.id, None)
            if aug:                                  # x += ...: its value is now unknown (never the stale one)
                self._augs += 1
                self.ncanon[t.id] = self.ncanon_m[t.id] = f"aug:{t.id}:{self._augs}"
                self.naff.pop(t.id, None)
                return
            if k is not None:
                self.consts[t.id] = k
            elif s is not None:
                self.strs[t.id] = s
            elif isinstance(value, (ast.List, ast.Tuple)) and value.elts and \
                    all(self._str(e) is not None for e in value.elts):
                self.strlists[t.id] = [self._str(e) for e in value.elts]
            self.ncanon[t.id] = self.canon(value)
            self.ncanon_m[t.id] = self.canon(value, mask=True)
            self.naff[t.id] = self.affine(value)
            self.tset[t.id] = self.terms(value)
            pf = self._polyfeat(value)
            if pf is not None:
                self.polydeg[t.id] = pf
            else:
                self.polydeg.pop(t.id, None)
            self.samp[t.id] = self.sample_of(value)
            fam = self._obj_family(value)
            if fam:
                self.skind[t.id] = fam
            else:
                self.skind.pop(t.id, None)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for el in t.elts:
                if isinstance(el, ast.Name):
                    for d in (self.consts, self.strs, self.strlists, self.ncanon, self.ncanon_m, self.tset, self.naff,
                              self.polydeg):
                        d.pop(el.id, None)
                    self.samp[el.id] = self.sample_of(value)
                    self.ncanon[el.id] = "v:" + el.id
                    self.ncanon_m[el.id] = "v:" + el.id
        elif isinstance(t, ast.Subscript):
            key = t.slice
            if isinstance(key, ast.Tuple) and len(key.elts) == 2:
                if not (isinstance(key.elts[0], ast.Slice) and key.elts[0].lower is None
                        and key.elts[0].upper is None):
                    return                           # df.loc[mask, 'x'] = ...: a partial rewrite, not a definition
                key = key.elts[1]
            name = self._str(key)
            if name is None:
                return
            if aug:                                  # df['x'] -= ...: a new, unknown quantity
                self._augs += 1
                self.dcanon[name] = self.dcanon_m[name] = f"aug:{name}:{self._augs}"
                self.daff.pop(name, None)
                return
            tc, mc, af = self.canon(value), self.canon(value, mask=True), self.affine(value)
            self.dcanon[name], self.dcanon_m[name], self.daff[name] = tc, mc, af
            frame = t.value
            if isinstance(frame, ast.Name) and frame.id in self.tset and frame.id not in self._frames():
                self.tset[frame.id] = self.tset[frame.id] | {tc}   # X['c2'] = x**2 on a design matrix
        if isinstance(value, ast.Call) and _fname(value.func) == "assign":
            for kw in value.keywords:
                if kw.arg:
                    self.dcanon[kw.arg], self.dcanon_m[kw.arg] = self.canon(kw.value), self.canon(kw.value, True)
                    self.daff[kw.arg] = self.affine(kw.value)

    def _frames(self) -> set:
        """Names that hold the task's data frames (read from a file, or row-subsets of one): assigning a column to them
        defines a derived column, not a regressor of a design matrix."""
        return {n for n, v in self.ncanon.items() if v.startswith(tuple(f"{r}(" for r in READERS))}

    def _obj_family(self, value):
        for n in ast.walk(value):
            if isinstance(n, ast.Call) and _fname(n.func) in SK_FAMILY:
                return SK_FAMILY[_fname(n.func)]
        return None

    def _calls(self, c, node, loop_groups):
        super()._calls(c, node, loop_groups)
        if node is None:
            return
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                kw = {k.arg: k.value for k in n.keywords if k.arg}
                if "inplace" in kw and self._const(kw["inplace"]) == 1:
                    nm, f = n.func.value.id, n.func.attr
                    if f in ("dropna", "query") or f in ROWSEL:
                        cp = copy.copy(n)
                        cp.keywords = [k for k in n.keywords if k.arg != "inplace"]
                        self.samp[nm] = self.sample_of(cp)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
                self._dim(c, n)

    # ------------------------------------------------------------------ estimates
    def _estimator(self, c, n, f, kw, loop_groups):
        e = super()._estimator(c, n, f, kw, loop_groups)
        try:
            r = self._rest(c, n, f, kw)
        except RecursionError:
            r = None
        if r is not None and r.dep:
            r.cond = self._cond > 0
            self.rests.append(r)
        return e

    def _mk(self, c, n, family, yexpr, terms, sample, params=(), ycanon=None) -> EstR:
        dep = frozenset(self.resolve(yexpr)[0] & self.cols) if yexpr is not None else frozenset()
        return EstR(c.idx, self._top.get(id(n), 0), family, dep,
                    ycanon if ycanon is not None else self.canon(yexpr), frozenset(terms), frozenset(sample),
                    tuple(sorted(params)), _fname(n.func))

    def _params(self, family, kw, extra=()) -> list:
        out = [(k, self.canon(kw[k], mask=True)) for k in PARAM_KW.get(family, ()) if k in kw]
        have = {k for k, _ in out}
        out += [(k, v) for (fam, k), v in PARAM_DEFAULT.items() if fam == family and k not in have]
        return out + list(extra)

    def _rest(self, c, n: ast.Call, f: str, kw: dict):
        args = n.args
        recv = n.func.value if isinstance(n.func, ast.Attribute) else None
        # statsmodels classes: OLS(endog, exog), WLS(..., weights=), GLM(..., family=), RLM(..., M=), KernelReg(...)
        if f in SM_FAMILY and not (isinstance(n.func, ast.Attribute) and n.func.attr == "from_formula"):
            fam = SM_FAMILY[f]
            y = args[0] if args else kw.get("endog")
            x = args[1] if len(args) > 1 else kw.get("exog")
            if y is None:
                return None
            extra = []
            if f == "WLS" or "weights" in kw:
                w = kw.get("weights", args[2] if f == "WLS" and len(args) > 2 else None)
                extra.append(("weights", self.canon(w, mask=True)))
            if f == "GLM" and "family" not in kw and len(args) > 2:
                extra.append(("family", self.canon(args[2], mask=True)))
            return self._mk(c, n, fam, y, self.terms(x), self.sample_of(y) | self.sample_of(x),
                            self._params(fam, kw, extra))
        if f == "KernelReg":
            y = args[0] if args else kw.get("endog")
            x = args[1] if len(args) > 1 else kw.get("exog")
            if y is None:
                return None
            kk = dict(kw)
            if len(args) > 3 and "reg_type" not in kk:
                kk["reg_type"] = args[3]
            if len(args) > 4 and "bw" not in kk:
                kk["bw"] = args[4]
            return self._mk(c, n, "kernel", y, self.terms(x), self.sample_of(y) | self.sample_of(x),
                            self._params("kernel", kk))
        if f == "lowess":
            y = args[0] if args else kw.get("endog")
            x = args[1] if len(args) > 1 else kw.get("exog")
            if y is None:
                return None
            kk = dict(kw)
            if len(args) > 2 and "frac" not in kk:
                kk["frac"] = args[2]
            return self._mk(c, n, "lowess", y, self.terms(x), self.sample_of(y) | self.sample_of(x),
                            self._params("lowess", kk))
        if f == "rdrobust":
            y = args[0] if args else kw.get("y")
            x = args[1] if len(args) > 1 else kw.get("x")
            if y is None:
                return None
            return self._mk(c, n, "rdrobust", y, self.terms(x), self.sample_of(y) | self.sample_of(x),
                            self._params("rdrobust", kw))
        # formula API: smf.ols('y ~ x', data=df), sm.OLS.from_formula(...)
        if f in FORMULA_FAMILY or f == "from_formula":
            fs = args[0] if args else kw.get("formula")
            s = self._str(fs) if fs is not None else None
            if s is None or "~" not in s:
                return None
            fam = FORMULA_FAMILY.get(f) or SM_FAMILY.get(_fname(n.func.value) if isinstance(n.func, ast.Attribute)
                                                         else "", "ols")
            data = kw.get("data", args[1] if len(args) > 1 else None)
            lhs, _, rhs = s.partition("~")
            dep = frozenset(self.formula(s)[2] & self.cols)
            ycan = self._fexpr(lhs)
            extra = []
            if f == "wls" or "weights" in kw:
                extra.append(("weights", self.canon(kw.get("weights"), mask=True)))
            smp = self.sample_of(data)
            if "subset" in kw:
                smp = smp | {self._mask_key(kw["subset"])}
            r = EstR(c.idx, self._top.get(id(n), 0), fam, dep, ycan, frozenset(self._fterms(rhs)), frozenset(smp),
                     tuple(sorted(self._params(fam, kw, extra))), f)
            return r
        # .fit(...) of an estimator object: sklearn fit(X, y); lifelines fit(df, duration_col=..., ...)
        if f == "fit" and recv is not None:
            fam = None
            if isinstance(recv, ast.Name):
                fam = self.skind.get(recv.id)
            elif isinstance(recv, ast.Call):
                fam = SK_FAMILY.get(_fname(recv.func))
            if not fam:
                return None
            ctor = recv if isinstance(recv, ast.Call) else None
            ckw = {k.arg: k.value for k in ctor.keywords if k.arg} if ctor is not None else {}
            if "duration_col" in kw or fam in ("cox", "weibull", "expfit", "lognorm") and args and len(args) == 1:
                ycols = [kw.get("duration_col"), kw.get("event_col")]
                dep = frozenset().union(*[self.resolve(y)[0] for y in ycols if y is not None]) & self.cols
                data = args[0] if args else kw.get("df")
                fs = self._str(kw.get("formula")) if "formula" in kw else None
                terms = self._fterms(fs) if fs else {self.col_canon(x) for x in self.resolve(data)[0] - dep}
                return EstR(c.idx, self._top.get(id(n), 0), fam, frozenset(dep),
                            "|".join(self.canon(y) for y in ycols if y is not None), frozenset(terms),
                            self.sample_of(data), tuple(sorted(self._params(fam, ckw))), f)
            if fam in ("km", "weibull", "expfit", "lognorm") and args:
                ev = kw.get("event_observed", args[1] if len(args) > 1 else None)
                dep = frozenset((self.resolve(args[0])[0] | (self.resolve(ev)[0] if ev is not None else set()))
                                & self.cols)
                return EstR(c.idx, self._top.get(id(n), 0), fam, dep, self.canon(args[0]), frozenset(),
                            self.sample_of(args[0]), (), f)
            if len(args) >= 2:
                X, y = args[0], args[1]
                terms = self.terms(X)
                if fam in ("ols", "ridge", "lasso", "enet", "robust", "logit", "poisson", "glm", "quantreg") and \
                        not ("fit_intercept" in ckw and self._const(ckw["fit_intercept"]) == 0):
                    terms = terms | {"1"}
                extra = [("weights", self.canon(kw["sample_weight"], mask=True))] if "sample_weight" in kw else []
                return self._mk(c, n, fam, y, terms, self.sample_of(X) | self.sample_of(y),
                                self._params(fam, ckw, extra))
            return None
        # scipy / numpy estimators
        if f in ("linregress", "polyfit", "lstsq", "theilslopes", "siegelslopes", "curve_fit"):
            if f == "curve_fit":
                fn = args[0] if args else kw.get("f")
                x = args[1] if len(args) > 1 else kw.get("xdata")
                y = args[2] if len(args) > 2 else kw.get("ydata")
                if y is None:
                    return None
                fid = self.fbody.get(fn.id, "fn:" + fn.id) if isinstance(fn, ast.Name) else \
                    ("fn:" + hashlib.sha256(ast.dump(fn).encode()).hexdigest()[:10] if fn is not None else "fn:?")
                extra = [("fn", fid)] + ([("sigma", self.canon(kw["sigma"], mask=True))] if "sigma" in kw else [])
                return self._mk(c, n, "curve_fit", y, self.terms(x), self.sample_of(x) | self.sample_of(y), extra)
            if f in ("theilslopes", "siegelslopes"):
                y = args[0] if args else kw.get("y")
                x = args[1] if len(args) > 1 else kw.get("x")
                if y is None:
                    return None
                return self._mk(c, n, "robust", y, self.terms(x) | {"1"}, self.sample_of(x) | self.sample_of(y))
            x = args[0] if args else kw.get("x" if f != "lstsq" else "a")
            y = args[1] if len(args) > 1 else kw.get("y" if f != "lstsq" else "b")
            if y is None:
                return None
            if f == "polyfit":
                deg = self._const(args[2] if len(args) > 2 else kw.get("deg"))
                base = self.terms(x)
                if deg is None or len(base) != 1:
                    terms = base | {"1", f"deg:{self.canon(args[2] if len(args) > 2 else kw.get('deg'))}"}
                else:
                    b = next(iter(base))
                    terms = {"1"} | ({b} if deg >= 1 else set()) | {f"pow{_num(j)}({b})" for j in range(2, int(deg) + 1)}
                extra = [("weights", self.canon(kw["w"], mask=True))] if "w" in kw else []
                return self._mk(c, n, "ols", y, terms, self.sample_of(x) | self.sample_of(y), extra)
            terms = self.terms(x) | ({"1"} if f == "linregress" else set())
            return self._mk(c, n, "ols", y, terms, self.sample_of(x) | self.sample_of(y))
        if f in ("ttest_ind", "mannwhitneyu", "ttest_ind_from_stats") and len(args) >= 2:
            a, b = args[0], args[1]
            da, db = self.resolve(a)[0] & self.cols, self.resolve(b)[0] & self.cols
            if da and da == db:
                fam = "dim" if f != "mannwhitneyu" else "rank"
                pair = "|".join(sorted([",".join(sorted(self.sample_of(a))), ",".join(sorted(self.sample_of(b)))]))
                return self._mk(c, n, fam, a, {"1"}, {"pair:" + pair})
        return None

    def _mean_of(self, node):
        """(response expression, its row restriction) of `<Y restricted>.mean()` / np.mean(<Y restricted>)."""
        if isinstance(node, ast.Call) and _fname(node.func) in ("mean", "average", "nanmean"):
            recv = node.func.value if isinstance(node.func, ast.Attribute) else None
            if isinstance(recv, ast.Name) and recv.id in MODULES:
                recv = None
            tgt = recv if recv is not None else (node.args[0] if node.args else None)
            if tgt is not None:
                return tgt
        return None

    def _dim(self, c, n: ast.BinOp):
        a, b = self._mean_of(n.left), self._mean_of(n.right)
        if a is None or b is None:
            return
        da, db = self.resolve(a)[0] & self.cols, self.resolve(b)[0] & self.cols
        if not da or da != db or len(da) != 1:
            return
        pair = "|".join(sorted([",".join(sorted(self.sample_of(a))) or "all", ",".join(sorted(self.sample_of(b))) or "all"]))
        self.rests.append(EstR(c.idx, self._top.get(id(n), 0), "dim", frozenset(da), self.canon(a), frozenset({"1"}),
                               frozenset({"pair:" + pair}), (), "mean-mean", self._cond > 0))

    # ------------------------------------------------------------------ formulas
    def _fexpr(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "_"
        try:
            node = ast.parse(text, mode="eval").body
        except SyntaxError:
            return "raw:" + re.sub(r"\s+", "", text)
        return self._ffactor(node)

    def _ffactor(self, node) -> str:
        if isinstance(node, ast.Call):
            f = _fname(node.func)
            if f == "C" and node.args:
                return f"cat({self.canon(node.args[0], fnames=True)})"
            if f == "I" and node.args:
                return self.canon(node.args[0], fnames=True)
            if f == "Q" and node.args:
                s = self._str(node.args[0])
                return self.col_canon(s) if s else self.canon(node.args[0], fnames=True)
        return self.canon(node, fnames=True)

    def _fterms(self, rhs: str) -> set:
        """Canonical terms of a patsy right-hand side: '+' / '-' separated terms, ':' interactions, '*' expanded,
        C() / I() / Q() / transforms, an intercept term '1' unless removed."""
        terms, intercept = set(), True
        for sign, t in _split_top(rhs or "", "+-"):
            t = t.strip()
            if not t:
                continue
            if t in ("0", "1", "-1"):
                if t == "0" or sign == "-" or t == "-1":
                    intercept = False
                continue
            if sign == "-":
                continue
            stars = [s for _, s in _split_top(t, "*")]
            if len(stars) > 1:
                fs = [self._finter(s) for s in stars]
                for k in range(1, len(fs) + 1):
                    for comb in itertools.combinations(fs, k):
                        terms.add(_inter(list(itertools.chain.from_iterable(comb))))
                continue
            terms.add(_inter(self._finter(t)))
        if intercept:
            terms.add("1")
        return terms

    def _finter(self, t: str) -> list:
        return [self._fexpr(p) for _, p in _split_top(t, ":")]


IND_RE = re.compile(r"^(ge|le|eq|ne|and|or|not|isin|between)\(")


def _is_indicator(core: str) -> bool:
    return bool(IND_RE.match(core))


def _negate(x: str) -> str:
    m = re.match(r"^(ge|le)\((.*)\)$", x)
    if m and "," in m.group(2):
        return ("le(" if m.group(1) == "ge" else "ge(") + m.group(2) + ")"
    if x.startswith("not(") and x.endswith(")"):
        return x[4:-1]
    return f"not({x})"


def _andor(kind: str, parts: list) -> str:
    """and(...) / or(...) with nested same-kind groups flattened, parts sorted and de-duplicated."""
    flat = []
    for p in parts:
        if p.startswith(kind + "(") and p.endswith(")") and _balanced(p[len(kind) + 1:-1]):
            flat += _split_args(p[len(kind) + 1:-1])
        else:
            flat.append(p)
    flat = sorted(set(flat))
    return flat[0] if len(flat) == 1 else f"{kind}(" + ",".join(flat) + ")"


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        d += ch == "("
        d -= ch == ")"
        if d < 0:
            return False
    return d == 0


def _split_args(s: str) -> list:
    out, d, cur = [], 0, []
    for ch in s:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
        if ch == "," and d == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _inter(parts: list) -> str:
    parts = sorted(parts)
    return parts[0] if len(parts) == 1 else "mul(" + ",".join(parts) + ")"


def _split_top(s: str, seps: str) -> list:
    """[(sign, piece)] split at depth-0 separator characters; '**' is never split on '*'."""
    out, depth, cur, sign, i = [], 0, [], "+", 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in seps:
            if ch == "*" and (s[i + 1:i + 2] == "*" or (i > 0 and s[i - 1] == "*")):
                cur.append(ch)
                i += 1
                continue
            if ch == "-" and not "".join(cur).strip() and out == [] and sign == "+":
                sign = "-"
                i += 1
                continue
            out.append((sign, "".join(cur)))
            cur, sign = [], ("-" if ch == "-" else "+")
            i += 1
            continue
        cur.append(ch)
        i += 1
    out.append((sign, "".join(cur)))
    return out


# --------------------------------------------------------------------------- which statements of a cell ran
def _echo_transform(code: str) -> str | None:
    """harness/sandbox._echo_last_expression, reproduced (the kernel runs the dedented cell through it under
    GBP_ECHO_LAST=1): a final bare expression becomes an assignment + a repr print, and the cell is ast.unparse-d, so
    its line numbers differ from the model's text."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return None
    expr = tree.body[-1].value
    tree.body[-1] = ast.Assign(targets=[ast.Name(id="_gbp_echo", ctx=ast.Store())], value=expr)
    tree.body.append(ast.parse("if _gbp_echo is not None:\n    print(repr(_gbp_echo))").body[0])
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001
        return None


def failed_at(cell) -> int | None:
    """None if the cell ran to the end; else the index of the top-level statement that raised (statements before it
    ran), or -1 if that cannot be located (a timeout without a traceback, a lost kernel): then nothing counts as run."""
    if cell.ok:
        return None
    if cell.tree is None:
        return -1
    m = CELL_TB.search(cell.stderr or "")
    if not m:
        return -1
    line, text = int(m.group(1)), m.group(2).strip()
    import textwrap
    code = textwrap.dedent(cell.code)
    echoed = _echo_transform(code)             # the v3 harness always echoes: try the rewritten text first
    variants = ([echoed] if echoed is not None else []) + [code]
    n_top = len(cell.tree.body)
    for v in variants:
        lines = v.splitlines()
        if not (1 <= line <= len(lines)) or lines[line - 1].strip() != text:
            continue
        try:
            body = ast.parse(v).body
        except SyntaxError:
            continue
        for i, st in enumerate(body):
            if st.lineno <= line <= (st.end_lineno or st.lineno):
                return min(i, n_top - 1)
    return -1


def _prints(cell) -> set:
    """Top-level statement indices of a cell that print (print() anywhere inside, or display())."""
    out = set()
    for i, st in enumerate(getattr(cell.tree, "body", []) or []):
        for n in ast.walk(st):
            if isinstance(n, ast.Call) and _fname(n.func) in ("print", "display"):
                out.add(i)
                break
    return out


# --------------------------------------------------------------------------- the rule
def run_status(cells: list, A: ReestAnalyzer) -> tuple[list, list]:
    """(estimates that were SUCCESSFUL, estimates whose cell RAN TO THE END and printed a number), each sorted by
    (cell, statement); only estimates with a real response column."""
    fail = {c.idx: failed_at(c) for c in cells}
    prints = {c.idx: _prints(c) for c in cells}
    nums = {c.idx: len(num_tokens(c.evidence)) >= 1 for c in cells}

    def successful(e: EstR) -> bool:
        k = fail[e.cell]
        if not nums[e.cell]:
            return False
        if k is None:
            return True
        return k >= 0 and e.top < k and any(e.top <= j < k for j in prints[e.cell])

    ests = sorted((e for e in A.rests if e.dep and not e.cond and e.cell in fail), key=lambda e: (e.cell, e.top))
    return [e for e in ests if successful(e)], [e for e in ests if fail[e.cell] is None and nums[e.cell]]


def decide(cells: list, A: ReestAnalyzer) -> tuple[float, str, list]:
    """(reest in {0, 1}, where, credited pairs) -- the module docstring's rule."""
    good, ended = run_status(cells, A)
    pairs = []
    for b in ended:
        prior = [a for a in good if a.cell < b.cell and a.dep == b.dep
                 and estimand_class(a.family) == estimand_class(b.family)]
        if not prior:
            continue
        if b.signature() in {a.signature() for a in prior}:
            continue
        pairs.append((prior[0], b))
    n_est = len({(e.cell, e.top, e.signature()) for e in good})
    if pairs:
        a, b = pairs[0]
        return 1.0, (f"cells {a.cell},{b.cell}: {sorted(b.dep)} re-estimated in a later cell "
                     f"({'; '.join(b.differs(a))[:300]})"), pairs
    if not good:
        return 0.0, "no successful estimate with an explicit response", pairs
    return 0.0, f"{n_est} successful estimate(s); no later-cell re-estimate under a new specification", pairs


# --------------------------------------------------------------------------- the scorer
def _fmt(x) -> str:
    return "None" if x is None else f"{x:g}"


class ProcessScorerV31(ProcessScorer):
    """process_signals.ProcessScorer with spec := reest and M3 check := max(M3, reest) (module docstring)."""

    detector = DETECTOR

    def reest(self, messages, cfg: dict, data_dir, n_prefix: int = 0) -> tuple[float, str, list]:
        V = self.vocab(data_dir, cfg)
        cells, _ = executed_cells(messages or [], n_prefix)
        if not cells:
            return 0.0, "no executed cell", []
        A = ReestAnalyzer(V).run(cells)
        return decide(cells, A)

    def signals(self, messages, cfg: dict, data_dir, answer=None, n_prefix: int = 0) -> ProcessSignals:
        s = super().signals(messages, cfg, data_dir, answer=answer, n_prefix=n_prefix)
        r, where, _ = self.reest(messages, cfg, data_dir, n_prefix)
        fired, detail = dict(s.fired), dict(s.detail)
        old_spec, old_check = s.fired.get("spec"), s.fired.get("check")
        v3_spec_where = s.detail.get("spec") if isinstance(s.detail.get("spec"), str) else ""
        fired["spec"] = r
        detail["spec"] = (f"{TAG} {_fmt(r)}: {where} [v3 spec {_fmt(old_spec)}: {v3_spec_where} | "
                          f"v3 check {_fmt(old_check)}]")
        chk = s.detail.get("check")
        if "M3_functional" in s.modes and isinstance(chk, dict) and isinstance(chk.get("modes"), dict):
            modes, wh = dict(chk["modes"]), dict(chk.get("where") or {})
            if modes.get("M3_functional") is not None and r > modes["M3_functional"]:
                modes["M3_functional"] = r
                wh["M3_functional"] = f"{TAG}: {where}"
            vals = [v for v in modes.values() if v is not None]
            fired["check"] = (sum(vals) / len(vals)) if vals else None
            detail["check"] = {"modes": modes, "where": wh}
        detail["v3"] = {"spec": old_spec, "check": old_check}
        detail[TAG] = {"value": r, "where": where, "detector": DETECTOR}
        return ProcessSignals(fired, detail, s.modes)


def detector_record() -> dict:
    """What rl/grpo_v31.py writes into harness_v31.json: the detector's name and this file's sha256."""
    return {"detector": DETECTOR, "detector_sha256": _sha16(Path(__file__)),
            "signals_sha256": _sha16(Path(PS.__file__))}


def parse_logged(where_spec) -> dict:
    """Read one process.jsonl `where["spec"]` string: {'v2': bool, 'reest': 0/1 or None, 'v3_spec', 'v3_check',
    'v3_est' (the old detector's estimate count)}; a v3 (old-detector) string gives v2 False and its own values."""
    s = str(where_spec or "")
    out = {"v2": False, "reest": None, "v3_spec": None, "v3_check": None, "v3_est": None}
    m = re.match(rf"^{TAG} (\d(?:\.\d+)?): ", s)
    if m:
        out["v2"], out["reest"] = True, float(m.group(1))
        mv = re.search(r"\[v3 spec (\S+?): (.*) \| v3 check (\S+)\]$", s)
        if mv:
            out["v3_spec"] = None if mv.group(1) == "None" else float(mv.group(1))
            out["v3_check"] = None if mv.group(3) == "None" else float(mv.group(3))
            s = mv.group(2)
    me = re.match(r"^(\d+) estimate\(s\) with an explicit response", s)
    if me:
        out["v3_est"] = int(me.group(1))
    elif re.match(r"^cells \d+,\d+: same response", s):
        out["v3_est"] = 2
    return out
