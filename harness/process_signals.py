"""HARNESS V3 PROCESS SIGNALS: verified intermediate behaviours of one analysis episode (docs/RL_PLAN.md s.15).

WHY. Answer-only reward on a family mixture taught the model to do LESS analysis (R2 s2: code cells per rollout ~5 ->
1.8, DABench -24.2 at p = 4e-13), and the P14 14B single-family runs saturate at reward 1.0 with every GRPO group dead
by step ~40-50 (causal_iv: cells 8.8 -> 2.0). Harness v2's evidence components were never used because nothing
rewarded using them. v3 therefore rewards the PROCESS, but only what the sandbox can VERIFY: every signal below is a
deterministic function of (a) the code the harness actually executed, (b) the output the sandbox returned for it, and
(c) the task's own data files and trap table. Nothing is read from the model's prose, comments or reasoning.

WHAT COUNTS AS EXECUTED. The harness's own parsers decide (harness/agents.py): an assistant reply is executed iff it
carries no parseable JSON answer (extract_answer -> None: an answer ends the episode before any code runs), the turn is
not a forced-answer turn, and extract_code finds a ```python block (the FIRST block only -- the others are never run).
The next message is that cell's feedback (STDOUT:/STDERR:/timeout). Only STDOUT counts as evidence.

THE FIVE SIGNALS (each in [0, 1]; None = not applicable to this task, and left out of the mean). "Shown" always means
in the cell's EVIDENCE = its STDOUT minus the text the cell typed into print() (see ANTI-GAMING):
  inspect   the data were read and SUMMARISED before modelling: some executed cell that calls no estimator, placed
            before the first estimator cell that RAN TO THE END (no traceback), showed >= min(3, #columns) of the data
            files' real column names and >= 3 numbers, with a data file read at or before it.  N/A: no tabular data.
  strata    group structure present in the data was examined: an executed cell keyed a grouping operation
            (groupby / value_counts / unique / crosstab / pivot_table / C(g) / strata= / a loop over groups) on a
            real LOW-CARDINALITY column g (2..20 distinct values, <= half the rows), and shows >= 2 of g's actual
            values. The outcome's mean by TREATMENT (the naive effect, a trap) does not count.  N/A: no such column.
  check     the family's hazard checks: the fraction of the family's trap MODES (scripts/trap_modes.py, from the
            instance's own ablation table, accepted variants excluded) whose check was executed:
              M1 censoring       a 2-valued column (not the treatment) summarised (counts/means) -- a WEAK detector
              M2 heterogeneity   a group-aware analysis: another column compared ACROSS a grouping column (shown), a
                                 model with a C(g)/strata(g)/strata=/groups= term or a non-numeric grouping column, or
                                 per-group fits in a loop over the groups
              M3 functional      the same response re-estimated in a later cell in another functional form
                                 (estimator, transform, weights, degree), or residuals examined
              M4 measurement     the measuring context tabulated: another column compared across a grouping column
                                 (instrument / unit / site vs the measured value) -- the M2 descriptive detector
              M5 selection       the same response re-estimated in a later cell with a covariate set that differs by
                                 >= 1 real column
              M6 identification  the FIRST STAGE examined: the treatment regressed / correlated on the instrument
                                 (roles from the instance) with the instrument's row shown, or a first-stage statistic
                                 (params, F, R^2, t, p, summary) printed
              M7 entity          uniqueness checked (duplicated / drop_duplicates / nunique, or an id-like column)
            N/A: no mapped mode with a detector (or M6 without roles).
  spec      robustness: the same response (identical real columns after resolving derived ones: ln k and k are both
            k_per_s; detrended x is x AND time) estimated in a LATER cell under a specification not fitted in the earlier
            one (estimator, regressors, transforms, weights, groups), both cells ran to the end and printed numbers, same
            estimand class (a Cox model and a KM curve answer different questions). A pipeline re-run unchanged never
            pairs with itself: the model must have seen one estimate and then estimated the same response differently.
  grounded  the answer is the analysis's own: every numeric answer value is shown (rounding-aware, sign included) by an
            executed cell, and not as a numeric literal of that cell.  N/A: no numeric answer keys.

ANTI-GAMING (tested in experiments/harness_v3/tests): every column-based signal requires the column to exist in the
task's data files AND its values/name to be SHOWN by the sandbox; code that never ran (second ```python blocks, blocks
beside a JSON answer, forced turns) earns nothing; column names, group labels or numbers the cell typed into print()
(plain, f-string, concatenation, .format) are removed from the evidence; credit per signal is capped at 1 and the score
is a mean over applicable signals, so repeating a behaviour earns nothing more. Against an IRRELEVANT column set (the
same transcripts scored with decoy column names, decoy_vocab) every column-based signal is 0.

WHAT IT CANNOT SEE (stated, not hidden): a check done inside a helper whose result is never printed; alternatives fitted
and compared within ONE cell (only a later-cell re-estimate counts as spec/M3/M5); group structure in a column that is
not low-cardinality; M4 hazards that are not group-shaped (drift / detrending in a time series); M1 on a family whose
censoring indicator is not a 2-valued column. These read as 0 (or N/A), never as fired. At 14B, inspect and grounded
are near-saturated on the untrained model (>= 84% on the training families), so their gradient there is small.

    from process_signals import ProcessScorer
    ps = ProcessScorer()
    sig = ps.signals(messages, cfg, data_dir, answer=out["answer"], n_prefix=out["n_prefix"])
    sig.score, sig.fired, sig.detail
"""
from __future__ import annotations

import ast
import csv
import gzip
import io
import math
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from agents import extract_answer, extract_code  # noqa: E402


def _load_trap_modes() -> dict:
    """scripts/trap_modes.py's TRAP_MODE, loaded by file path (putting scripts/ on sys.path could shadow modules the
    trainer imports)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gbp_trap_modes", HERE.parent / "scripts" / "trap_modes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.TRAP_MODE)


TRAP_MODE = _load_trap_modes()

SIGNALS = ("inspect", "strata", "check", "spec", "grounded")
DETECTABLE_MODES = ("M1_censoring", "M2_heterogeneity", "M3_functional", "M4_measurement", "M5_selection",
                    "M6_identification", "M7_entity")
FORCE_PREFIX = "STOP. You have"
MAX_ROWS = 20000                  # rows read per table for column statistics
MAX_BYTES = 64 * 1024 * 1024      # a larger file contributes its header only
GROUP_MAX = 20                    # a grouping column has 2..GROUP_MAX distinct values ...
GROUP_FRAC = 0.5                  # ... and at most this fraction of the rows
KEEP_VALUES = 50                  # distinct values remembered per column

# --------------------------------------------------------------------------- vocabularies of calls
SM_CLASSES = {"OLS": "ols", "WLS": "ols", "GLS": "ols", "GLSAR": "ols", "GLM": "glm", "Logit": "logit",
              "Probit": "probit", "Poisson": "poisson", "NegativeBinomial": "negbin", "RLM": "robust",
              "QuantReg": "quantreg", "PHReg": "cox", "MixedLM": "mixed", "IV2SLS": "iv", "IVGMM": "iv",
              "MNLogit": "logit", "OrderedModel": "ordinal", "IV2SLS_": "iv"}
FORMULA_FUNCS = {"ols": "ols", "wls": "ols", "gls": "ols", "glsar": "ols", "glm": "glm", "logit": "logit",
                 "probit": "probit", "poisson": "poisson", "negativebinomial": "negbin", "rlm": "robust",
                 "quantreg": "quantreg", "phreg": "cox", "mixedlm": "mixed", "mnlogit": "logit"}
FIT_CLASSES = {"LinearRegression": "ols", "Ridge": "ridge", "Lasso": "lasso", "ElasticNet": "enet",
               "HuberRegressor": "robust", "TheilSenRegressor": "robust", "LogisticRegression": "logit",
               "PoissonRegressor": "poisson", "CoxPHFitter": "cox", "KaplanMeierFitter": "km",
               "WeibullFitter": "weibull", "WeibullAFTFitter": "weibull", "NelsonAalenFitter": "km",
               "ExponentialFitter": "expfit", "LogNormalFitter": "lognorm", "GaussianProcessRegressor": "gp",
               "RandomForestRegressor": "rf", "Polynomial": "poly"}
FUNC_EST = {"curve_fit": "curve_fit", "least_squares": "optim", "leastsq": "optim", "minimize": "optim",
            "minimize_scalar": "optim", "linregress": "ols", "theilslopes": "robust", "siegelslopes": "robust",
            "polyfit": "poly", "lstsq": "ols", "ODR": "odr", "odr": "odr", "inv": "linalg", "pinv": "linalg",
            "solve": "linalg", "fmin": "optim", "basinhopping": "optim", "differential_evolution": "optim"}
# Estimates are compared (spec / M3 / M5) only within one estimand class: a Cox model and a Kaplan-Meier curve of the
# same follow-up answer different questions, so fitting both is not a robustness check.
SURVIVAL_MODELS = {"cox", "weibull", "expfit", "lognorm"}
SURVIVAL_CURVES = {"km"}


def estimand_class(kind: str) -> str:
    return "survival_model" if kind in SURVIVAL_MODELS else "survival_curve" if kind in SURVIVAL_CURVES else "regression"


CORR_FUNCS = {"corr", "corrcoef", "pearsonr", "spearmanr", "kendalltau", "cov", "corrwith"}
TRANSFORMS = {"log": "log", "log10": "log", "log2": "log", "log1p": "log", "exp": "exp", "expm1": "exp",
              "sqrt": "sqrt", "power": "pow", "square": "pow", "reciprocal": "inv", "arcsinh": "asinh"}
GROUP_AGGS = {"mean", "median", "std", "var", "sum", "min", "max", "describe", "agg", "aggregate", "apply",
              "count", "quantile", "first", "last", "unique", "nunique", "value_counts", "sem", "transform", "diff",
              "size", "head", "tail", "mad", "prod", "cumsum"}
FIRST_STAGE_STATS = {"params", "fvalue", "f_pvalue", "rsquared", "rsquared_adj", "tvalues", "pvalues", "bse",
                     "summary", "conf_int", "f_test", "wald_test", "coef_", "summary2"}
RESID_ATTRS = {"resid", "resid_pearson", "resid_deviance", "resid_response", "resid_working", "resid_anscombe",
               "residuals", "wresid", "outlier_test", "get_influence"}
READERS = {"read_csv", "read_table", "read_excel", "read_parquet", "read_json", "loadtxt", "genfromtxt", "load",
           "read_fwf", "open", "fromfile", "read_hdf", "read_feather", "read_pickle", "read_h5ad", "mmread"}
FORMULA_WORDS = {"C", "I", "Q", "np", "log", "log10", "log2", "exp", "sqrt", "center", "standardize", "scale", "bs",
                 "cr", "cc", "Treatment", "Sum", "Poly", "power", "abs"}

NUM_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w])")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# --------------------------------------------------------------------------- the task's data
@dataclass
class Column:
    name: str
    n_distinct: int                      # capped at KEEP_VALUES + 1
    values: list                         # the distinct raw strings if n_distinct <= KEEP_VALUES, else []
    numeric: bool
    rows: int


@dataclass
class Vocab:
    """What the task's data files actually contain. Every column-based signal is checked against this."""
    columns: dict = field(default_factory=dict)          # name -> Column (all tabular files, first file wins)
    files: list = field(default_factory=list)            # data file basenames (tabular or not)
    roles: dict = field(default_factory=dict)            # _provenance.roles (causal families), columns only

    @property
    def grouping(self) -> set:
        return {c.name for c in self.columns.values()
                if 2 <= c.n_distinct <= GROUP_MAX and c.n_distinct <= max(2, GROUP_FRAC * c.rows)}

    @property
    def binary(self) -> set:
        return {c.name for c in self.columns.values() if c.n_distinct == 2}

    @property
    def id_like(self) -> set:
        return {c.name for c in self.columns.values() if c.rows >= 10 and c.n_distinct > KEEP_VALUES
                and not c.numeric} | {c.name for c in self.columns.values()
                                      if c.rows >= 10 and c.values and len(c.values) >= 0.9 * c.rows}


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _read_table(path: Path) -> tuple[list, list] | None:
    name = path.name.lower()
    if not re.search(r"\.(csv|tsv|txt|tab|dat)(\.gz)?$", name):
        return None
    try:
        big = path.stat().st_size > MAX_BYTES
        raw = gzip.open(path, "rt", errors="replace", newline="") if name.endswith(".gz") else \
            open(path, errors="replace", newline="")
        with raw as fh:
            head = fh.readline()
            if not head.strip():
                return None
            delim = "\t" if ("\t" in head and head.count("\t") >= head.count(",")) else ","
            if delim == "," and "," not in head:
                delim = None
            header = next(csv.reader(io.StringIO(head), delimiter=delim or " ", skipinitialspace=True))
            header = [h.strip() for h in header if h.strip()]
            rows = []
            if not big:
                reader = csv.reader(fh, delimiter=delim or " ", skipinitialspace=True)
                for i, r in enumerate(reader):
                    if i >= MAX_ROWS:
                        break
                    rows.append(r)
            return header, rows
    except (OSError, UnicodeError, csv.Error, StopIteration):
        return None


def build_vocab(data_dir, cfg: dict | None = None) -> Vocab:
    """Column statistics of every tabular file under data_dir (sibling datasets skipped)."""
    v = Vocab()
    d = Path(data_dir) if data_dir else None
    if d is None or not d.is_dir():
        return v
    for p in sorted(x for x in d.rglob("*") if x.is_file() and "siblings" not in x.relative_to(d).parts):
        v.files.append(p.name)
        t = _read_table(p)
        if not t:
            continue
        header, rows = t
        for j, h in enumerate(header):
            if h in v.columns or not IDENT_RE.fullmatch(h) and not h:
                continue
            vals, seen = [], set()
            numeric = True
            for r in rows:
                if j >= len(r):
                    continue
                s = r[j].strip()
                if s == "":
                    continue
                if numeric and not _is_num(s):
                    numeric = False
                if s not in seen and len(seen) <= KEEP_VALUES:
                    seen.add(s)
                    vals.append(s)
            v.columns[h] = Column(h, len(seen), vals if len(seen) <= KEEP_VALUES else [], numeric, len(rows))
    roles = ((cfg or {}).get("_provenance") or {}).get("roles") or {}
    for k, x in roles.items():
        if isinstance(x, str) and x in v.columns:
            v.roles[k] = x
        elif isinstance(x, list):
            v.roles[k] = [c for c in x if isinstance(c, str) and c in v.columns]
    return v


def family_modes(cfg: dict) -> list:
    """The trap modes of this instance: its ablation table's MISTAKES (fails_at_step set), mapped by trap_modes."""
    abl = ((cfg.get("_provenance") or {}).get("ablations")) or {}
    modes = {TRAP_MODE.get(n) for n, a in abl.items() if isinstance(a, dict) and a.get("fails_at_step") is not None}
    return sorted(m for m in modes if m in DETECTABLE_MODES)


# --------------------------------------------------------------------------- the transcript
@dataclass
class Cell:
    idx: int                 # order among executed cells
    code: str
    stdout: str
    stderr: str
    tree: ast.AST | None

    @property
    def evidence(self) -> str:
        """STDOUT minus the text the cell TYPED into it: string literals passed to print (plain, f-string parts,
        concatenations, .format templates) and a bare string echoed as the last expression. Column names or numbers the
        model wrote into a message are not evidence that the data showed them."""
        ev = getattr(self, "_evidence", None)
        if ev is None:
            ev = self.stdout
            for lit in sorted(_typed_literals(self.tree), key=len, reverse=True):
                if len(lit) >= 3:
                    ev = ev.replace(lit, " ")
                else:                                # a short typed word ('A', 'H1'): remove it as a whole token
                    ev = re.sub(r"(?<![\w.])" + re.escape(lit) + r"(?![\w])", " ", ev)
            self._evidence = ev
        return ev

    @property
    def ok(self) -> bool:
        """Ran to the end: no traceback, no timeout, no kernel loss."""
        e = self.stderr or ""
        return not ("Traceback (most recent call last)" in e or "execution timed out" in e or "[kernel]" in e
                    or "ERROR: execution timed out" in self.stdout)


def _typed_literals(tree) -> list:
    """Literal text a cell prints itself (module docstring: ANTI-GAMING). Pieces of 3+ characters are removed from the
    evidence as substrings, shorter ones only when they are a whole word ('A', 'H1'); a typed ':' or ', ' is left alone
    (removing it would erase real output)."""
    out = []
    if tree is None:
        return out

    def strings(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    yield v.value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
            yield from strings(node.left)
            yield from strings(node.right)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "join"):
            yield from strings(node.func.value)

    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print":
            for a in n.args:
                out += list(strings(a))
            for kw in n.keywords:
                if kw.arg in ("sep", "end"):
                    continue
    body = getattr(tree, "body", [])
    if body and isinstance(body[-1], ast.Expr):
        out += list(strings(body[-1].value))
    pieces = []
    for s in out:
        for part in re.split(r"[\n]+", s):
            part = part.strip()
            if len(part) >= 3 or re.fullmatch(r"\w+", part or "-"):
                pieces.append(part)
    return pieces


def _split_feedback(fb: str) -> tuple[str, str]:
    fb = fb or ""
    out, err = "", ""
    m = re.match(r"STDOUT:\n(.*?)(?:\nSTDERR:\n(.*))?\Z", fb, re.DOTALL)
    if m:
        out, err = m.group(1) or "", m.group(2) or ""
    elif fb.startswith("STDERR:\n"):
        err = fb[len("STDERR:\n"):]
    else:
        err = fb
    return out, err


def executed_cells(messages: list, n_prefix: int = 0) -> tuple[list, dict | None]:
    """(executed cells in order, the final answer object or None), by the harness's own rules."""
    first = (n_prefix or 1) + 1
    cells, answer, forcing = [], None, False
    for i in range(first, len(messages)):
        m = messages[i]
        if m.get("role") != "assistant":
            forcing = str(m.get("content") or "").startswith(FORCE_PREFIX)
            continue
        text = m.get("content") or ""
        ans = extract_answer(text)
        if ans is not None:
            answer = ans
            continue
        if forcing:
            continue
        code = extract_code(text)
        if not code:
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if nxt is None or nxt.get("role") == "assistant":
            continue
        out, err = _split_feedback(nxt.get("content") or "")
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            tree = None
        cells.append(Cell(len(cells), code, out, err, tree))
    return cells, answer


# --------------------------------------------------------------------------- output evidence
def num_tokens(text: str) -> list:
    """(value, half-unit of the last printed digit) for every number printed in text."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        s = m.group(0)
        try:
            v = float(s)
        except ValueError:
            continue
        if not math.isfinite(v):
            continue
        mant, _, ex = s.lower().partition("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        e = int(ex) if ex else 0
        try:
            half = 0.5 * 10.0 ** (e - dec)
        except OverflowError:   # 10-25: e.g. '0.0e500' -- a finite value with an exponent past float range is no printed number to match
            continue
        out.append((v, half, dec > 0 or e < 0))
    return out


def _value_shown(val: str, text: str, toks: list) -> bool:
    if _is_num(val):
        v = float(val)
        integral = float(v).is_integer()
        for t, half, has_dec in toks:
            if not has_dec and not integral:
                continue
            if abs(t - v) <= max(half, 1e-12 * max(1.0, abs(v))):
                return True
        return False
    return bool(val) and re.search(r"(?<![\w])" + re.escape(val) + r"(?![\w])", text) is not None


def values_shown(col: Column, text: str) -> int:
    toks = num_tokens(text)
    return sum(_value_shown(v, text, toks) for v in col.values)


def names_shown(names, text: str) -> set:
    return {n for n in names if re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", text or "")}


# --------------------------------------------------------------------------- code analysis
@dataclass
class Est:
    cell: int
    kind: str
    dep: frozenset
    regs: frozenset
    transforms: frozenset
    weighted: bool
    groups: frozenset
    extra: str = ""
    per_group: frozenset = frozenset()

    def signature(self):
        return (self.kind, self.regs, self.transforms, self.weighted, self.groups, self.extra, self.per_group)


@dataclass
class GroupOp:
    cell: int
    keys: frozenset          # grouping columns (real)
    targets: frozenset       # other columns summarised by group ("*" = all)
    kind: str


def _fname(func) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _root_name(node) -> str:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else ""


class Analyzer:
    """Walks the executed cells in order, keeping a light dataflow: names and derived columns -> the real data
    columns (and transforms) they were computed from."""

    def __init__(self, vocab: Vocab):
        self.V = vocab
        self.cols = set(vocab.columns)
        self.derived: dict = {}      # derived column name -> (base cols, transforms)
        self.env: dict = {}          # variable name -> (base cols, transforms)
        self.kind_of: dict = {}      # variable name -> estimator kind (sklearn/lifelines objects, statsmodels models)
        self.est_of: dict = {}       # variable name -> Est (models / results)
        self.funcs: dict = {}        # user function name -> transforms in its body
        self.ests: list = []
        self.groupops: list = []
        self.corrs: list = []        # (cell, cols)
        self.est_cells: set = set()
        self.reads_data: set = set()
        self.attrs: dict = {}        # cell -> set of (root name, attr)
        self.literals: dict = {}     # cell -> set of numeric literals
        self.nonnum_groups = {g for g in vocab.grouping if not vocab.columns[g].numeric}
        self.modified: dict = {}     # real column overwritten from OTHER columns (df['x'] -= baseline): its new basis

    # ---- column resolution
    def _col_of_const(self, s: str):
        if s in self.modified:
            return set(self.modified[s][0]), set(self.modified[s][1])
        if s in self.cols:
            return {s}, set()
        if s in self.derived:
            return set(self.derived[s][0]), set(self.derived[s][1])
        return None

    def _selected(self, sl) -> tuple[set, set] | None:
        """Columns a subscript SELECTS (df['a'], df[['a', 'b']], df.loc[mask, 'a']); None if it is not a selection."""
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:          # .loc[rows, cols]
            return self._selected(sl.elts[1])
        names = []
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            names = [sl.value]
        elif isinstance(sl, (ast.List, ast.Tuple)) and sl.elts and \
                all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in sl.elts):
            names = [e.value for e in sl.elts]
        if not names:
            return None
        cols, tr = set(), set()
        for s in names:
            r = self._col_of_const(s)
            if r:
                cols |= r[0]
                tr |= r[1]
        return cols, tr

    def resolve(self, node) -> tuple[set, set]:
        """Base data columns and transforms an expression depends on. A column SELECTION (df['a'], df[['a','b']])
        depends on the selected columns only, so filtering a frame (df = df[df.note.isna()]) does not leak 'note'
        into every column later taken from it."""
        cols, tr = set(), set()
        if node is None:
            return cols, tr
        if isinstance(node, ast.Subscript):
            sel = self._selected(node.slice)
            if sel is not None:
                return sel
            a, b = self.resolve(node.value), self.resolve(node.slice)
            return a[0] | b[0], a[1] | b[1]
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                r = self._col_of_const(node.value)
                if r:
                    return set(r[0]), set(r[1])
                if "~" in node.value:
                    fc, ft, _, _ = self.formula(node.value)
                    return set(fc), set(ft)
            return cols, tr
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id not in ("np", "pd", "sm", "smf", "stats", "math"):
                r = self._col_of_const(node.attr)
                if r:
                    return set(r[0]), set(r[1])
            return self.resolve(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return set(self.env[node.id][0]), set(self.env[node.id][1])
            return cols, tr
        if isinstance(node, ast.Call):
            f = _fname(node.func)
            if f in TRANSFORMS:
                tr.add(TRANSFORMS[f])
            elif f in self.funcs:
                tr |= set(self.funcs[f])
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                tr.add("pow")
            elif isinstance(node.op, ast.Div) and isinstance(node.left, ast.Constant) and node.left.value == 1:
                tr.add("inv")
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
                continue
            a = self.resolve(ch)
            cols |= a[0]
            tr |= a[1]
        return cols, tr

    def formula(self, f: str):
        """(all columns, transforms, dependent columns, grouping columns) of a patsy/lifelines formula string."""
        lhs, _, rhs = f.partition("~")
        if not _:
            lhs, rhs = "", f
        tr = set()
        for w, t in TRANSFORMS.items():
            if re.search(r"\b" + w + r"\s*\(", f):
                tr.add(t)
        if re.search(r"\*\*|\^", f):
            tr.add("pow")

        def cols_in(s):
            out = set()
            for w in IDENT_RE.findall(s) + re.findall(r"Q\(\s*['\"]([^'\"]+)['\"]", s):
                r = self._col_of_const(w)
                if r and w not in FORMULA_WORDS:
                    out |= r[0]
            return out

        dep, allc = cols_in(lhs), cols_in(lhs) | cols_in(rhs)
        # group terms: C(g) / strata(g) factors, or a non-numeric grouping column (patsy dummy-codes it). A numeric
        # low-cardinality covariate (ecog 0/1/2, a 0/1 treatment) is an ordinary regressor, not a group structure.
        groups = set()
        for m in re.finditer(r"\b(?:C|strata)\(\s*([A-Za-z_][A-Za-z0-9_]*)", rhs):
            r = self._col_of_const(m.group(1))
            if r:
                groups |= r[0]
        for w in IDENT_RE.findall(rhs):
            if w in self.nonnum_groups:
                groups.add(w)
        return allc, tr, dep, groups

    # ---- the walk
    def run(self, cells: list) -> "Analyzer":
        files = set(self.V.files)
        for c in cells:
            if c.tree is None:
                continue
            self.attrs[c.idx] = set()
            self.literals[c.idx] = set()
            for n in ast.walk(c.tree):
                if isinstance(n, ast.Constant):
                    if isinstance(n.value, str) and ("data_files" in n.value or Path(n.value).name in files):
                        self.reads_data.add(c.idx)
                    elif isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                        self.literals[c.idx].add(float(n.value))
                elif isinstance(n, ast.Attribute):
                    self.attrs[c.idx].add((_root_name(n.value), n.attr))
                elif isinstance(n, ast.FunctionDef):
                    _, tr = self.resolve(ast.Module(body=n.body, type_ignores=[]))
                    self.funcs[n.name] = frozenset(tr)
                elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda) and \
                        len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    self.funcs[n.targets[0].id] = frozenset(self.resolve(n.value.body)[1])
            self._stmts(c, c.tree.body, loop_groups=frozenset())
        return self

    def _stmts(self, c: Cell, body, loop_groups: frozenset):
        for st in body:
            if isinstance(st, (ast.For, ast.AsyncFor)):
                g = self._loop_groups(c, st)
                self._calls(c, st.iter, loop_groups)
                self._assign_target(st.target, st.iter)
                self._stmts(c, st.body, loop_groups | g)
                self._stmts(c, st.orelse, loop_groups)
            elif isinstance(st, (ast.If, ast.While)):
                self._calls(c, st.test, loop_groups)
                self._stmts(c, st.body, loop_groups)
                self._stmts(c, st.orelse, loop_groups)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                self._stmts(c, st.body, loop_groups)
            elif isinstance(st, ast.Try):
                for b in (st.body, *[h.body for h in st.handlers], st.orelse, st.finalbody):
                    self._stmts(c, b, loop_groups)
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            else:
                self._calls(c, st, loop_groups)
                if isinstance(st, ast.Assign):
                    for t in st.targets:
                        self._assign_target(t, st.value)
                elif isinstance(st, (ast.AnnAssign, ast.AugAssign)) and st.value is not None:
                    self._assign_target(st.target, st.value, aug=isinstance(st, ast.AugAssign))

    def _loop_groups(self, c: Cell, st) -> frozenset:
        """A loop over a grouping column's levels (for g, sub in df.groupby('site') / for s in df['site'].unique())."""
        out = set()
        for n in ast.walk(st.iter):
            if isinstance(n, ast.Call) and _fname(n.func) in ("groupby", "unique", "drop_duplicates", "value_counts"):
                keys = self._group_keys(n)
                out |= keys & self.V.grouping
        return frozenset(out)

    def _assign_target(self, t, value, aug: bool = False):
        cols, tr = self.resolve(value)
        if isinstance(t, ast.Name):
            if aug and t.id in self.env:
                cols |= set(self.env[t.id][0]); tr |= set(self.env[t.id][1])
            self.env[t.id] = (frozenset(cols), frozenset(tr))
            k = self._kind_of_value(value)
            if k:
                self.kind_of[t.id] = k
            e = self._est_of_value(value)
            if e is not None:
                self.est_of[t.id] = e
        elif isinstance(t, (ast.Tuple, ast.List)):
            for el in t.elts:
                self._assign_target(el, value, aug)
        elif isinstance(t, ast.Subscript):
            key = t.slice
            if isinstance(key, ast.Tuple) and len(key.elts) == 2:        # df.loc[mask, 'new'] = ...
                key = key.elts[1]
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value not in self.cols:
                self.derived[key.value] = (frozenset(cols), frozenset(tr))
            elif isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in self.cols:
                # a real column rewritten in place from other columns (detrended, calibrated): later uses of it are
                # a different quantity -- a fit of the detrended series is not an alternative fit of the raw one
                base = set(self._col_of_const(key.value)[0]) | cols
                if base - {key.value}:
                    old = self.modified.get(key.value, (frozenset({key.value}), frozenset()))
                    self.modified[key.value] = (frozenset(base | set(old[0])), frozenset(set(old[1]) | tr))
        if isinstance(value, ast.Call) and _fname(value.func) == "assign":
            for kw in value.keywords:
                if kw.arg and kw.arg not in self.cols:
                    kc, kt = self.resolve(kw.value)
                    self.derived[kw.arg] = (frozenset(kc), frozenset(kt))

    def _kind_of_value(self, value):
        for n in ast.walk(value):
            if isinstance(n, ast.Call):
                f = _fname(n.func)
                if f in FIT_CLASSES:
                    return FIT_CLASSES[f]
                if f in SM_CLASSES:
                    return SM_CLASSES[f]
        return None

    def _est_of_value(self, value):
        """The estimate a value holds: sm.OLS(y, X).fit() -> the OLS spec; model.fit() -> the model's spec."""
        found = None
        for n in ast.walk(value):
            e = getattr(n, "_gbp_est", None) if isinstance(n, ast.Call) else None
            if e is not None and e.kind:
                return e
            if isinstance(n, ast.Name) and n.id in self.est_of and found is None:
                found = self.est_of[n.id]
        return found

    # ---- calls
    def _group_keys(self, call: ast.Call) -> set:
        f = _fname(call.func)
        keys = set()
        if f == "groupby":
            args = list(call.args) + [kw.value for kw in call.keywords if kw.arg in ("by", "level")]
            for a in args:
                keys |= self.resolve(a)[0]
        elif f in ("value_counts", "unique", "nunique", "drop_duplicates"):
            if isinstance(call.func, ast.Attribute):
                recv = call.func.value
                if isinstance(recv, ast.Subscript) or isinstance(recv, ast.Attribute):
                    keys |= self.resolve(recv)[0]
                elif isinstance(recv, ast.Name) and recv.id in self.env:
                    keys |= set(self.env[recv.id][0])
            for a in list(call.args) + [kw.value for kw in call.keywords if kw.arg == "subset"]:
                keys |= self.resolve(a)[0]
        elif f in ("crosstab",):
            for a in call.args[:2]:
                keys |= self.resolve(a)[0]
            for kw in call.keywords:
                if kw.arg in ("index", "columns"):
                    keys |= self.resolve(kw.value)[0]
        elif f in ("pivot_table", "pivot"):
            for kw in call.keywords:
                if kw.arg in ("index", "columns"):
                    keys |= self.resolve(kw.value)[0]
        return keys

    def _calls(self, c: Cell, node, loop_groups: frozenset):
        if node is None:
            return
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                self._call(c, n, loop_groups)

    def _call(self, c: Cell, n: ast.Call, loop_groups: frozenset):
        f = _fname(n.func)
        kw = {k.arg: k.value for k in n.keywords if k.arg}
        # --- grouping operations
        if f in ("groupby", "value_counts", "unique", "nunique", "crosstab", "pivot_table", "pivot",
                 "drop_duplicates"):
            keys = self._group_keys(n) & self.cols
            targets = set()
            if f == "groupby":
                targets = self._groupby_targets(n)
            elif f == "crosstab":
                allk = set()
                for a in n.args[:2]:
                    allk |= self.resolve(a)[0]
                targets = allk
            elif f == "pivot_table" and "values" in kw:
                targets = self.resolve(kw["values"])[0]
            elif f == "drop_duplicates":
                targets = keys
            if keys:
                self.groupops.append(GroupOp(c.idx, frozenset(keys), frozenset(targets), f))
        if f == "duplicated":
            recv = n.func.value if isinstance(n.func, ast.Attribute) else None
            keys = (self.resolve(recv)[0] if recv is not None else set()) | self.resolve(ast.Tuple(elts=list(n.args),
                                                                                                     ctx=ast.Load()))[0]
            self.groupops.append(GroupOp(c.idx, frozenset(keys & self.cols), frozenset(), "duplicated"))
        # --- correlations
        if f in CORR_FUNCS:
            cols = self.resolve(n)[0]
            if isinstance(n.func, ast.Attribute):
                recv = n.func.value
                rc = self.resolve(recv)[0]
                if not rc and isinstance(recv, ast.Name) and recv.id in self.env and f in ("corr", "cov"):
                    rc = set(self.cols)                      # a whole data frame: df.corr() covers every column
                cols |= rc
            self.corrs.append((c.idx, frozenset(cols)))
        # --- estimators
        e = self._estimator(c, n, f, kw, loop_groups)
        if e is not None:
            n._gbp_est = e                                  # read back by _est_of_value on assignment
            self.est_cells.add(c.idx)
            if e.kind:
                self.ests.append(e)

    def _groupby_targets(self, n: ast.Call) -> set:
        """Columns summarised by a groupby: df.groupby(g)[x].mean() -> {x}; df.groupby(g).mean() -> {'*'}."""
        parent_chain = getattr(n, "_gbp_parent", None)
        return parent_chain or set()

    def _estimator(self, c: Cell, n: ast.Call, f: str, kw: dict, loop_groups: frozenset):
        args = n.args
        dep, regs, tr, weighted, groups, extra, kind = set(), set(), set(), False, set(), "", None
        if f in SM_CLASSES and not (isinstance(n.func, ast.Attribute) and n.func.attr == "from_formula"):
            kind = SM_CLASSES[f]
            endog = args[0] if args else kw.get("endog")
            exog = args[1] if len(args) > 1 else kw.get("exog")
            if endog is None:                       # constructed without data (rare) -> not an estimate
                return Est(c.idx, "", frozenset(), frozenset(), frozenset(), False, frozenset())
            dep, t1 = self.resolve(endog)
            regs, t2 = self.resolve(exog)
            if kind == "iv":
                inst = args[2] if len(args) > 2 else kw.get("instrument")
                regs |= self.resolve(inst)[0]
            tr = {f"y:{t}" for t in t1} | {f"x:{t}" for t in t2}
            weighted = f in ("WLS", "GLS") or "weights" in kw or "sigma" in kw
            for k in ("groups", "strata"):
                if k in kw:
                    groups |= self.resolve(kw[k])[0]
        elif f in FORMULA_FUNCS or f == "from_formula":
            fs = args[0] if args else kw.get("formula")
            if not (isinstance(fs, ast.Constant) and isinstance(fs.value, str)):
                fs_cols, fs_tr = self.resolve(fs)
                if not fs_cols:
                    return Est(c.idx, "", frozenset(), frozenset(), frozenset(), False, frozenset())
                kind = FORMULA_FUNCS.get(f, "ols")
                regs, tr = fs_cols, {f"x:{t}" for t in fs_tr}
            else:
                allc, ftr, dep, groups = self.formula(fs.value)
                kind = FORMULA_FUNCS.get(f, "ols") if f != "from_formula" else \
                    SM_CLASSES.get(_fname(n.func.value) if isinstance(n.func, ast.Attribute) else "", "ols")
                regs = allc - dep
                lhs = fs.value.partition("~")[0]
                tr = {("y:" if re.search(r"\b" + w + r"\s*\(", lhs) else "x:") + t for w, t in TRANSFORMS.items()
                      if re.search(r"\b" + w + r"\s*\(", fs.value)}
                for d in list(dep):
                    if d in self.derived:
                        tr |= {f"y:{t}" for t in self.derived[d][1]}
            weighted = "weights" in kw or f == "wls"
            if "groups" in kw:
                groups |= self.resolve(kw["groups"])[0]
        elif f == "fit" and isinstance(n.func, ast.Attribute):
            recv = n.func.value
            rname = _root_name(recv)
            if not args and not kw:
                # model.fit(): estimation happens here; the spec was recorded at construction
                return Est(c.idx, "", frozenset(), frozenset(), frozenset(), False, frozenset())
            kind = self.kind_of.get(rname) or (FIT_CLASSES.get(_fname(recv.func)) if isinstance(recv, ast.Call)
                                               else None) or "fit"
            if "duration_col" in kw or kind in ("cox", "weibull"):
                dep = self.resolve(kw.get("duration_col"))[0] | self.resolve(kw.get("event_col"))[0]
                data_cols = self.resolve(args[0])[0] if args else set()
                regs = data_cols - dep
                if "formula" in kw:
                    fc, ft, _, fg = self.formula(kw["formula"].value) if isinstance(kw["formula"], ast.Constant) \
                        else (set(), set(), set(), set())
                    regs, groups = fc - dep, groups | fg
                if "strata" in kw:
                    groups |= self.resolve(kw["strata"])[0]
                weighted = "weights_col" in kw
            elif len(args) >= 2:
                regs, t2 = self.resolve(args[0])
                dep, t1 = self.resolve(args[1])
                tr = {f"y:{t}" for t in t1} | {f"x:{t}" for t in t2}
                weighted = "sample_weight" in kw or "w" in kw
            else:
                dep, t1 = self.resolve(args[0]) if args else (set(), set())
                tr = {f"y:{t}" for t in t1}
                if "event_observed" in kw:
                    dep |= self.resolve(kw["event_observed"])[0]
        elif f in FUNC_EST:
            kind = FUNC_EST[f]
            if f == "curve_fit":
                fn = args[0] if args else kw.get("f")
                x = args[1] if len(args) > 1 else kw.get("xdata")
                y = args[2] if len(args) > 2 else kw.get("ydata")
                dep, t1 = self.resolve(y)
                regs, t2 = self.resolve(x)
                fname = fn.id if isinstance(fn, ast.Name) else ""
                ftr = self.funcs.get(fname, frozenset()) if fname else \
                    frozenset(self.resolve(fn.body)[1]) if isinstance(fn, ast.Lambda) else frozenset()
                tr = {f"y:{t}" for t in t1} | {f"x:{t}" for t in t2} | {f"f:{t}" for t in ftr}
                extra = "fn:" + ",".join(sorted(ftr))
                weighted = "sigma" in kw
            elif f in ("linregress", "lstsq", "polyfit"):
                x = args[0] if args else kw.get("x")
                y = args[1] if len(args) > 1 else kw.get("y")
                dep, t1 = self.resolve(y)
                regs, t2 = self.resolve(x)
                tr = {f"y:{t}" for t in t1} | {f"x:{t}" for t in t2}
                if f == "polyfit":
                    deg = args[2] if len(args) > 2 else kw.get("deg")
                    extra = f"deg:{deg.value}" if isinstance(deg, ast.Constant) else "deg:?"
                    weighted = "w" in kw
            elif f in ("theilslopes", "siegelslopes"):
                y = args[0] if args else kw.get("y")
                x = args[1] if len(args) > 1 else kw.get("x")
                dep, t1 = self.resolve(y)
                regs, t2 = self.resolve(x)
                tr = {f"y:{t}" for t in t1} | {f"x:{t}" for t in t2}
            else:                                    # optimisers, linear algebra: the response is not explicit
                allc, t = self.resolve(n)
                regs = allc
                fn = args[0] if args else None
                if isinstance(fn, ast.Name) and fn.id in self.funcs:
                    t = set(t) | set(self.funcs[fn.id])
                extra = f"fn:{fn.id}" if isinstance(fn, ast.Name) else ""
                tr = {f"x:{x}" for x in t}
                if f in ("inv", "pinv", "solve"):
                    # np.linalg.inv(X.T @ X) @ X.T @ y: a manual fit; its response is the other operand
                    return Est(c.idx, "", frozenset(), frozenset(), frozenset(), False, frozenset())
        else:
            return None
        groups |= regs & self.nonnum_groups              # e.g. pd.get_dummies(df['hospital']) among the regressors
        return Est(c.idx, kind, frozenset(dep & self.cols), frozenset((regs - dep) & self.cols), frozenset(tr),
                   weighted, frozenset(groups & self.cols), extra, loop_groups)


def _annotate_parents(tree: ast.AST, cols: set):
    """Mark every groupby call with the columns selected right after it: df.groupby(g)[x] / df.groupby(g).x /
    df.groupby(g).mean() (all columns, '*')."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Call) and _fname(n.value.func) == "groupby":
            sel = set()
            for s in ast.walk(n.slice):
                if isinstance(s, ast.Constant) and isinstance(s.value, str):
                    sel.add(s.value)
            n.value._gbp_parent = sel
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Call) and _fname(n.value.func) == "groupby":
            if n.attr in cols:
                n.value._gbp_parent = {n.attr}
            elif n.attr in GROUP_AGGS and n.attr not in ("size", "count", "head", "tail", "first", "last"):
                n.value._gbp_parent = {"*"}
            elif n.attr in ("size", "count"):
                n.value._gbp_parent = set()


# --------------------------------------------------------------------------- the signals
@dataclass
class ProcessSignals:
    fired: dict                         # signal -> value in [0, 1], or None (not applicable)
    detail: dict
    modes: list

    @property
    def score(self) -> float:
        vals = [v for v in self.fired.values() if v is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def as_dict(self) -> dict:
        return {"score": round(self.score, 4), "fired": self.fired, "modes": self.modes, "detail": self.detail}


class ProcessScorer:
    """Thread-safe; caches each task's data vocabulary (one per data directory)."""

    def __init__(self, vocab_override=None):
        self._cache: dict = {}
        self._lock = threading.Lock()
        self.vocab_override = vocab_override           # callable(Vocab, cfg) -> Vocab (the column sham, tests)

    def vocab(self, data_dir, cfg) -> Vocab:
        key = str(Path(data_dir).resolve()) if data_dir else ""
        with self._lock:
            v = self._cache.get(key)
        if v is None:
            v = build_vocab(data_dir, cfg)
            with self._lock:
                self._cache[key] = v
        return self.vocab_override(v, cfg) if self.vocab_override else v

    def signals(self, messages, cfg: dict, data_dir, answer=None, n_prefix: int = 0) -> ProcessSignals:
        V = self.vocab(data_dir, cfg)
        cells, parsed_answer = executed_cells(messages or [], n_prefix)
        if answer is None and isinstance(parsed_answer, dict):
            answer = parsed_answer.get("answer", parsed_answer)
        for c in cells:
            if c.tree is not None:
                _annotate_parents(c.tree, set(V.columns))
        A = Analyzer(V).run(cells)
        modes = family_modes(cfg)
        fired, detail = {}, {"cells": len(cells)}
        fired["inspect"], detail["inspect"] = self._inspect(cells, A, V)
        fired["strata"], detail["strata"] = self._strata(cells, A, V)
        fired["check"], detail["check"] = self._check(cells, A, V, modes)
        fired["spec"], detail["spec"] = self._spec(cells, A)
        fired["grounded"], detail["grounded"] = self._grounded(cells, A, cfg, answer)
        return ProcessSignals(fired, detail, modes)

    # ---- helpers
    @staticmethod
    def _has_numbers(c: Cell, k: int = 1) -> bool:
        return len(num_tokens(c.evidence)) >= k

    def _inspect(self, cells, A: Analyzer, V: Vocab):
        if not V.columns:
            return None, "no tabular data"
        # modelling starts at the first cell that calls an estimator AND ran to the end (a cell that raised may not
        # have reached its estimator; looking at the data after a failed fit and before any fit ran is still looking)
        ran = [i for i in A.est_cells if cells[i].ok]
        first_est = min(ran) if ran else len(cells)
        need = min(3, len(V.columns))
        read_by = min(A.reads_data) if A.reads_data else None
        for c in cells:
            if c.idx >= first_est:
                break
            if read_by is None or c.idx < read_by:
                continue
            shown = names_shown(V.columns, c.evidence)
            if len(shown) >= need and self._has_numbers(c, 3):
                return 1.0, f"cell {c.idx}: {len(shown)} columns shown before the first estimator (cell {first_est})"
        return 0.0, f"no summary of the data before the first estimator (cell {first_est})"

    def _strata(self, cells, A: Analyzer, V: Vocab):
        G = V.grouping
        if not G:
            return None, "no low-cardinality column"
        byidx = {c.idx: c for c in cells}
        cand = [(op.cell, k) for op in A.groupops for k in op.keys if k in G and not self._naive(op, k, V)]
        cand += [(e.cell, g) for e in A.ests for g in (e.groups | e.per_group) if g in G]
        for ci, g in cand:
            if values_shown(V.columns[g], byidx[ci].evidence) >= 2:
                return 1.0, f"cell {ci}: grouped by {g}"
        return 0.0, f"no grouping on {sorted(G)} shown in output"

    @staticmethod
    def _naive(op: GroupOp, g: str, V: Vocab) -> bool:
        """The outcome's mean by TREATMENT is the naive effect estimate (the no-adjustment trap of every causal
        family), not an examination of structure: it earns neither strata nor a group check."""
        t, y = V.roles.get("treatment"), V.roles.get("outcome")
        return bool(t and y and g == t and op.kind in ("groupby", "pivot_table") and op.targets
                    and "*" not in op.targets and op.targets <= {y, t})

    def _grouped_comparison(self, cells, A: Analyzer, V: Vocab):
        """A column compared ACROSS a grouping column, with >= 2 group values in the output."""
        byidx = {c.idx: c for c in cells}
        for op in A.groupops:
            if op.kind in ("value_counts", "unique", "nunique", "duplicated") and not op.targets:
                continue
            for g in op.keys & V.grouping:
                if self._naive(op, g, V):
                    continue
                others = (set(V.columns) - {g}) if "*" in op.targets else (op.targets - {g})
                if op.kind == "groupby" and not others:
                    continue
                if others and values_shown(V.columns[g], byidx[op.cell].evidence) >= 2 and \
                        self._has_numbers(byidx[op.cell], 2):
                    return f"cell {op.cell}: {sorted(others)[:3]} by {g}"
        return None

    def _group_model(self, cells, A: Analyzer, V: Vocab):
        byidx = {c.idx: c for c in cells}
        for e in A.ests:
            gs = (e.groups | e.per_group) & V.grouping
            if gs and self._has_numbers(byidx[e.cell]):
                return f"cell {e.cell}: {e.kind} with {sorted(gs)}"
        return None

    def _pairs(self, cells, A: Analyzer):
        """Estimates of the same response under a CHANGED specification: (a, b) with a in an earlier cell i and b in
        a later cell j, both cells ran to the end and printed numbers, the response is identical (the same real
        columns after resolving derived ones: ln k and k are both k_per_s; residuals of a detrending fit are x AND
        time, a pipeline step, not an alternative), the estimand class is the same, and b's specification is NEW
        relative to everything cell i fitted for that response. So a pipeline re-run unchanged in a later cell (fit a
        baseline, then the model, in every cell) never pairs with itself, and within-cell pipeline steps never pair:
        the model must have seen one estimate and then estimated the same response differently."""
        byidx = {c.idx: c for c in cells}
        ests = [e for e in A.ests if e.dep and byidx[e.cell].ok and self._has_numbers(byidx[e.cell])]
        sigs: dict = {}
        for e in ests:
            sigs.setdefault((e.cell, e.dep), set()).add(e.signature())
        out = []
        for a in ests:
            for b in ests:
                if b.cell <= a.cell or a.dep != b.dep or estimand_class(a.kind) != estimand_class(b.kind):
                    continue
                if b.signature() in sigs[(a.cell, a.dep)]:
                    continue
                out.append((a, b))
        return out

    def _check(self, cells, A: Analyzer, V: Vocab, modes: list):
        if not modes:
            return None, "no mapped trap mode"
        res, where = {}, {}
        byidx = {c.idx: c for c in cells}
        pairs = self._pairs(cells, A)
        for m in modes:
            hit = None
            if m == "M1_censoring":
                treat = V.roles.get("treatment")
                for op in A.groupops:
                    for b in (op.keys | op.targets) & V.binary - {treat}:
                        if self._has_numbers(byidx[op.cell]):
                            hit = f"cell {op.cell}: {b} summarised"
                            break
                    if hit:
                        break
            elif m == "M2_heterogeneity":
                hit = self._grouped_comparison(cells, A, V) or self._group_model(cells, A, V)
            elif m == "M4_measurement":
                hit = self._grouped_comparison(cells, A, V)
            elif m == "M3_functional":
                for a, b in pairs:
                    if (a.kind, a.transforms, a.weighted, a.extra) != (b.kind, b.transforms, b.weighted, b.extra):
                        hit = f"cells {a.cell},{b.cell}: {a.kind}{sorted(a.transforms)} vs {b.kind}{sorted(b.transforms)}"
                        break
                if not hit:
                    for c in cells:
                        if any(at in RESID_ATTRS for _, at in A.attrs.get(c.idx, ())) and self._has_numbers(c, 2):
                            hit = f"cell {c.idx}: residuals examined"
                            break
            elif m == "M5_selection":
                for a, b in pairs:
                    if a.regs ^ b.regs:
                        hit = f"cells {a.cell},{b.cell}: covariates {sorted(a.regs ^ b.regs)} differ"
                        break
            elif m == "M6_identification":
                t, z = V.roles.get("treatment"), V.roles.get("instrument")
                if not (t and z):
                    res[m] = None
                    continue
                fs_cells = [e.cell for e in A.ests if t in e.dep and z in e.regs] + \
                           [ci for ci, cols in A.corrs if {t, z} <= cols]
                fs_names = {nm for nm, e in A.est_of.items() if t in e.dep and z in e.regs}
                for ci in sorted(set(fs_cells)):
                    c = byidx[ci]
                    if names_shown([z], c.evidence) and self._has_numbers(c):
                        hit = f"cell {ci}: first stage shown"
                        break
                if not hit and fs_cells:
                    for c in cells:
                        if c.idx < min(fs_cells):
                            continue
                        if any(r in fs_names and at in FIRST_STAGE_STATS for r, at in A.attrs.get(c.idx, ())) \
                                and self._has_numbers(c):
                            hit = f"cell {c.idx}: first-stage statistic printed"
                            break
            elif m == "M7_entity":
                for op in A.groupops:
                    if op.kind in ("duplicated", "drop_duplicates", "nunique") or (op.keys & V.id_like):
                        if self._has_numbers(byidx[op.cell]):
                            hit = f"cell {op.cell}: uniqueness checked"
                            break
            res[m] = 1.0 if hit else 0.0
            if hit:
                where[m] = hit
        vals = [v for v in res.values() if v is not None]
        detail = {"modes": res, "where": where}
        if not vals:
            return None, detail
        return sum(vals) / len(vals), detail

    def _spec(self, cells, A: Analyzer):
        pairs = self._pairs(cells, A)
        if pairs:
            a, b = pairs[0]
            return 1.0, f"cells {a.cell},{b.cell}: same response {sorted(a.dep)}, specifications differ"
        return 0.0, f"{sum(1 for e in A.ests if e.dep)} estimate(s) with an explicit response; no comparison"

    def _grounded(self, cells, A: Analyzer, cfg: dict, answer):
        from rl.reward import StepReward
        try:
            keys = StepReward._fields(cfg)
        except (KeyError, TypeError):
            keys = []
        if not keys:
            return None, "no numeric answer keys"
        obj = answer if isinstance(answer, dict) else {}
        if "answer" in obj and isinstance(obj["answer"], dict):
            obj = obj["answer"]
        vals = []
        for k in keys:
            try:
                v = float(obj.get(k))
            except (TypeError, ValueError):
                return 0.0, f"answer key {k} missing or not numeric"
            if not math.isfinite(v):
                return 0.0, f"answer key {k} not finite"
            vals.append((k, v))
        toks = []
        for c in cells:
            lits = A.literals.get(c.idx, set())
            for t, half, has_dec in num_tokens(c.evidence):
                if any(abs(t - x) <= 1e-9 * max(1.0, abs(x)) for x in lits):
                    continue
                toks.append((t, half, has_dec))
        missing = []
        for k, v in vals:
            s = repr(v)
            mant, _, ex = s.lower().partition("e")
            dv = len(mant.split(".")[1]) if "." in mant else 0
            half_v = 0.5 * 10.0 ** ((int(ex) if ex else 0) - dv)
            integral = v.is_integer()
            ok = False
            for t, half_t, has_dec in toks:
                if not has_dec and not integral:
                    continue
                if abs(t - v) <= max(half_t, half_v) + 1e-12 * max(1.0, abs(v)):
                    ok = True
                    break
            if not ok:
                missing.append(k)
        if missing:
            return 0.0, f"not in any computed output: {missing}"
        return 1.0, "every numeric answer value was printed by an executed cell"


# --------------------------------------------------------------------------- the column sham (measurement only)
DECOY_COLUMNS = ["hospital", "ecog", "site", "unit", "curve", "instrument", "year", "adopted", "sex", "batch", "plate",
                 "donor", "region", "cohort", "operator", "lab", "wave", "stratum", "arm", "centre"]


def decoy_vocab(v: Vocab, cfg: dict | None = None) -> Vocab:
    """The same task with an IRRELEVANT column set: every real column is replaced by a decoy name that is not in the
    data (with the real column's statistics), roles follow the renaming. Used to MEASURE how often the detectors fire
    on columns the task does not have (the 'irrelevant column set' sham): it is ~never, which is why the registered
    PROCESS-SHAM arm permutes the real process term within each GRPO group instead (rl/grpo_v3.py)."""
    real = set(v.columns)
    pool = [d for d in DECOY_COLUMNS if d not in real] + [f"var_{i}" for i in range(64)]
    ren = {c: pool[i] for i, c in enumerate(sorted(real))}
    out = Vocab(columns={ren[k]: Column(ren[k], c.n_distinct, c.values, c.numeric, c.rows)
                         for k, c in v.columns.items()}, files=list(v.files))
    for k, x in v.roles.items():
        out.roles[k] = ren.get(x) if isinstance(x, str) else [ren[c] for c in x if c in ren]
    return out
