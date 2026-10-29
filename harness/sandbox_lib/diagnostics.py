"""Descriptive diagnostics that return numbers (harness v2 CHECKS, switch GBP_CHECKS=1; docs/HARNESS_V2.md).

Every function computes statistics and returns them (a pandas DataFrame or a dict). None of them chooses or names
an analysis.

  balance(df, group, covariates=None)                 per-group means of each covariate, standardized differences
  group_summary(df, value, group)                     per-group count / missing / mean / sd / quantiles of one column
  heterogeneity(df, outcome, group, exposure=None,    between-group differences in an outcome, or in the slope of
                covariates=None)                      outcome on exposure across groups (Cochran Q, I^2)
  functional_form([df,] x, y, bins=8)                 fit statistics of linear / quadratic / log-scale fits, RESET
                                                      test, binned residual means, residual runs
  censoring_summary([df,] time, event, group=None)    follow-up and event counts, censoring fractions, KM medians
  missingness(df, target=None)                        missing counts per column; with target, how the other columns
                                                      differ between rows where target is missing and present
  selection_summary(df, selected, covariates=None)    counts and covariate differences between selected / unselected
  trend([df,] t, y)                                   linear drift of y over t, residual autocorrelation
  floor_summary([df,] x)                              zeros, negatives, values at the minimum, most repeated values
  duplicates(df, subset=None)                         duplicated rows and duplicated keys

Arguments may be column names (with df) or arrays / Series.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["balance", "group_summary", "heterogeneity", "functional_form", "censoring_summary", "missingness",
           "selection_summary", "trend", "floor_summary", "duplicates"]


def _col(df, x, name="x") -> pd.Series:
    if isinstance(x, str):
        if df is None:
            raise ValueError(f"column name {x!r} given without a DataFrame")
        return df[x]
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x), name=name)


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / s) if s > 0 else float("nan")


def _numeric_design(df: pd.DataFrame, covariates) -> pd.DataFrame:
    cols = {}
    for c in covariates:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            cols[c] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            cols[c] = s.astype(float)
        else:
            levels = s.dropna().astype(str).value_counts().index[:12]
            for lv in levels:
                cols[f"{c}={lv}"] = (s.astype(str) == lv).astype(float).where(s.notna())
    return pd.DataFrame(cols, index=df.index)


def balance(df: pd.DataFrame, group, covariates=None) -> pd.DataFrame:
    """Mean of each covariate within each level of `group`, the standardized mean difference (SMD) and the variance
    ratio between levels (for more than two levels: the largest absolute pairwise SMD). Categorical covariates are
    expanded to one indicator per level (at most 12 levels)."""
    g = _col(df, group, "group")
    if covariates is None:
        covariates = [c for c in df.columns if c != (group if isinstance(group, str) else None)]
    X = _numeric_design(df, list(covariates))
    levels = list(pd.Series(g).dropna().unique())
    try:
        levels = sorted(levels)
    except TypeError:
        pass
    rows = []
    for c in X.columns:
        row = {"covariate": c}
        vals = {}
        for lv in levels:
            v = X.loc[(g == lv).values, c].to_numpy(dtype=float)
            vals[lv] = v
            row[f"mean[{lv}]"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        if len(levels) == 2:
            a, b = vals[levels[1]], vals[levels[0]]
            row["smd"] = _smd(a, b)
            va, vb = np.nanvar(a, ddof=1) if len(a) > 1 else np.nan, np.nanvar(b, ddof=1) if len(b) > 1 else np.nan
            row["var_ratio"] = float(va / vb) if vb and np.isfinite(vb) and vb > 0 else float("nan")
        elif len(levels) > 2:
            best = 0.0
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    d = _smd(vals[levels[i]], vals[levels[j]])
                    if np.isfinite(d):
                        best = max(best, abs(d))
            row["max_abs_smd"] = best
        rows.append(row)
    out = pd.DataFrame(rows).set_index("covariate") if rows else pd.DataFrame()
    out.attrs["n_per_group"] = {str(k): int((g == k).sum()) for k in levels}
    return out


def group_summary(df: pd.DataFrame, value, group) -> pd.DataFrame:
    """Per level of `group`: n, n_missing, mean, sd, min, q25, median, q75, max of `value`, and the group median
    divided by the overall median."""
    v = pd.to_numeric(_col(df, value, "value"), errors="coerce")
    g = _col(df, group, "group")
    overall = float(np.nanmedian(v.to_numpy(dtype=float))) if v.notna().any() else float("nan")
    rows = []
    for lv, s in v.groupby(g.values, dropna=False):
        a = s.to_numpy(dtype=float)
        ok = a[np.isfinite(a)]
        q = np.quantile(ok, [0, .25, .5, .75, 1]) if len(ok) else [np.nan] * 5
        rows.append({"group": lv, "n": int(len(a)), "n_missing": int(len(a) - len(ok)),
                     "mean": float(ok.mean()) if len(ok) else np.nan, "sd": float(ok.std(ddof=1)) if len(ok) > 1 else np.nan,
                     "min": q[0], "q25": q[1], "median": q[2], "q75": q[3], "max": q[4],
                     "median_ratio": float(q[2] / overall) if overall not in (0.0,) and np.isfinite(overall) else np.nan})
    return pd.DataFrame(rows).set_index("group")


def _ols(y: np.ndarray, X: np.ndarray):
    """(beta, se, resid, dof) of y ~ X (X includes the intercept column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return beta, se, resid, dof


def heterogeneity(df: pd.DataFrame, outcome, group, exposure=None, covariates=None) -> dict:
    """Without `exposure`: how much `outcome` differs between levels of `group` -- one-way ANOVA F and p,
    Kruskal-Wallis H and p, Levene p (equal variances), eta^2 (share of variance between groups), per-group summary.
    With `exposure`: the OLS slope of outcome on exposure (plus `covariates`) fitted separately in each group, with
    SE and n; Cochran's Q, its p-value and I^2 across the group slopes; the inverse-variance pooled slope; and the
    slope from one regression on all rows."""
    from scipy import stats
    y = pd.to_numeric(_col(df, outcome, "outcome"), errors="coerce")
    g = _col(df, group, "group")
    out: dict = {}
    if exposure is None:
        groups = [s.dropna().to_numpy(dtype=float) for _, s in y.groupby(g.values)]
        groups = [a for a in groups if len(a) > 0]
        if len(groups) >= 2:
            f = stats.f_oneway(*groups)
            k = stats.kruskal(*groups)
            lv = stats.levene(*groups) if all(len(a) > 1 for a in groups) else None
            allv = np.concatenate(groups)
            ss_tot = float(((allv - allv.mean()) ** 2).sum())
            ss_b = float(sum(len(a) * (a.mean() - allv.mean()) ** 2 for a in groups))
            out.update({"anova_F": float(f.statistic), "anova_p": float(f.pvalue), "kruskal_H": float(k.statistic),
                        "kruskal_p": float(k.pvalue), "levene_p": float(lv.pvalue) if lv is not None else float("nan"),
                        "eta_squared": ss_b / ss_tot if ss_tot > 0 else float("nan"), "n_groups": len(groups)})
        out["per_group"] = group_summary(pd.DataFrame({"y": y, "g": g.values}), "y", "g")
        return out
    x = pd.to_numeric(_col(df, exposure, "exposure"), errors="coerce")
    covs = list(covariates or [])
    Z = _numeric_design(df, covs) if covs else pd.DataFrame(index=y.index)
    rows = []
    for lv in pd.Series(g).dropna().unique():
        m = (g == lv).values & y.notna().values & x.notna().values & Z.notna().all(axis=1).values
        n = int(m.sum())
        if n < len(covs) + 3:
            continue
        X = np.column_stack([np.ones(n), x.values[m]] + [Z[c].values[m] for c in Z.columns])
        beta, se, _, _ = _ols(y.values[m].astype(float), X.astype(float))
        rows.append({"group": lv, "n": n, "slope": float(beta[1]), "se": float(se[1])})
    per = pd.DataFrame(rows)
    if len(per) >= 2 and (per["se"] > 0).all():
        w = 1.0 / per["se"] ** 2
        pooled = float((w * per["slope"]).sum() / w.sum())
        Q = float((w * (per["slope"] - pooled) ** 2).sum())
        dfq = len(per) - 1
        out.update({"cochran_Q": Q, "Q_df": dfq, "Q_p": float(1 - stats.chi2.cdf(Q, dfq)),
                    "I2": max(0.0, (Q - dfq) / Q) if Q > 0 else 0.0, "pooled_slope_iv": pooled})
    m = y.notna().values & x.notna().values & Z.notna().all(axis=1).values
    if m.sum() > len(covs) + 2:
        X = np.column_stack([np.ones(int(m.sum())), x.values[m]] + [Z[c].values[m] for c in Z.columns])
        beta, se, _, _ = _ols(y.values[m].astype(float), X.astype(float))
        out.update({"all_rows_slope": float(beta[1]), "all_rows_se": float(se[1])})
    out["per_group"] = per.set_index("group") if len(per) else per
    return out


def functional_form(*args, bins: int = 8, **kw) -> dict:
    """functional_form([df,] x, y, bins=8). Fit statistics for y against x: R^2 of linear, quadratic, log(y)~x, y~log(x) and log(y)~log(x) fits (the log
    fits only where the values are positive); the quadratic coefficient's t; a RESET F-test p-value (adding
    fitted^2 and fitted^3 to the linear fit); the residual runs-test z (residual signs ordered by x); and the mean
    linear-fit residual in `bins` quantile bins of x."""
    from scipy import stats
    df, (x, y) = _split_df(args, kw, ["x", "y"])
    x = np.asarray(pd.to_numeric(_col(df, x, "x"), errors="coerce"), dtype=float)
    y = np.asarray(pd.to_numeric(_col(df, y, "y"), errors="coerce"), dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    out: dict = {"n": n}
    if n < 5:
        return out

    def r2(yy, X):
        b, se, r, dof = _ols(yy, X)
        ss = float(((yy - yy.mean()) ** 2).sum())
        return (1 - float(r @ r) / ss if ss > 0 else float("nan")), b, se, r

    one = np.ones(n)
    out["r2_linear"], b, se, resid = r2(y, np.column_stack([one, x]))
    out["slope_linear"] = float(b[1])
    q, bq, seq, _ = r2(y, np.column_stack([one, x, x ** 2]))
    out["r2_quadratic"] = q
    out["t_quadratic_term"] = float(bq[2] / seq[2]) if seq[2] > 0 else float("nan")
    if (y > 0).all():
        out["r2_logy_x"] = r2(np.log(y), np.column_stack([one, x]))[0]
    if (x > 0).all():
        out["r2_y_logx"] = r2(y, np.column_stack([one, np.log(x)]))[0]
    if (x > 0).all() and (y > 0).all():
        out["r2_logy_logx"] = r2(np.log(y), np.column_stack([one, np.log(x)]))[0]
    fit = y - resid
    X2 = np.column_stack([one, x, fit ** 2, fit ** 3])
    _, _, r_full, dof_full = _ols(y, X2)
    rss_r, rss_f = float(resid @ resid), float(r_full @ r_full)
    if rss_f > 0 and dof_full > 0:
        F = ((rss_r - rss_f) / 2) / (rss_f / dof_full)
        out["reset_F"], out["reset_p"] = float(F), float(1 - stats.f.cdf(F, 2, dof_full))
    order = np.argsort(x)
    s = np.sign(resid[order])
    s = s[s != 0]
    if len(s) > 2:
        runs = 1 + int((s[1:] != s[:-1]).sum())
        n1, n2 = int((s > 0).sum()), int((s < 0).sum())
        if n1 and n2:
            mu = 2 * n1 * n2 / (n1 + n2) + 1
            var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
            out["runs"], out["runs_z"] = runs, float((runs - mu) / math.sqrt(var)) if var > 0 else float("nan")
    try:
        cut = pd.qcut(pd.Series(x), q=min(bins, max(2, n // 5)), duplicates="drop")
        tb = pd.DataFrame({"bin": cut, "x": x, "resid": resid}).groupby("bin", observed=True).agg(
            n=("resid", "size"), x_mean=("x", "mean"), resid_mean=("resid", "mean"))
        out["binned_residuals"] = tb.reset_index(drop=True)
    except Exception:                                          # noqa: BLE001
        pass
    return out


def _km_median(t: np.ndarray, e: np.ndarray) -> float:
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    s, at_risk = 1.0, len(t)
    for tt in np.unique(t):
        d = int(e[t == tt].sum())
        n_here = int((t == tt).sum())
        if at_risk > 0 and d > 0:
            s *= 1 - d / at_risk
        if s <= 0.5:
            return float(tt)
        at_risk -= n_here
    return float("nan")


def _split_df(args, kw, names):
    """Accept f(df, "a", "b") and f(a_array, b_array) alike: returns (df, values in `names` order)."""
    args = list(args)
    df = kw.pop("df", None)
    if args and isinstance(args[0], pd.DataFrame):
        df = args.pop(0)
    vals = []
    for i, nm in enumerate(names):
        vals.append(args[i] if i < len(args) else kw.get(nm))
    return df, vals


def censoring_summary(*args, **kw) -> pd.DataFrame:
    """censoring_summary([df,] time, event, group=None). Per group (and overall): n, events, censored, fraction censored, median / max time for events and for
    censored rows, the fraction of censored rows whose time is below the median event time, and the Kaplan-Meier
    median. `event` is 1 for an observed event and 0 for censored."""
    df, (time, event, group) = _split_df(args, kw, ["time", "event", "group"])
    t = pd.to_numeric(_col(df, time, "time"), errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(_col(df, event, "event"), errors="coerce").to_numpy(dtype=float)
    g = _col(df, group, "group").to_numpy() if group is not None else None
    rows = []
    base = np.isfinite(t) & np.isfinite(e)
    subsets = [("all", base)] + ([(lv, base & (g == lv)) for lv in pd.unique(g)] if g is not None else [])
    for lv, m in subsets:
        tt, ee = t[m], e[m]
        if len(tt) == 0:
            continue
        ev, ce = tt[ee == 1], tt[ee == 0]
        med_ev = float(np.median(ev)) if len(ev) else float("nan")
        rows.append({"group": lv, "n": int(len(tt)), "events": int((ee == 1).sum()), "censored": int((ee == 0).sum()),
                     "frac_censored": float((ee == 0).mean()),
                     "median_time_events": med_ev, "median_time_censored": float(np.median(ce)) if len(ce) else np.nan,
                     "max_time": float(tt.max()),
                     "frac_censored_before_median_event": float((ce < med_ev).mean()) if len(ce) and np.isfinite(med_ev) else np.nan,
                     "km_median": _km_median(tt, (ee == 1).astype(int))})
    return pd.DataFrame(rows).set_index("group")


def missingness(df: pd.DataFrame, target=None) -> pd.DataFrame:
    """Without `target`: missing count and fraction per column. With `target` (a column name): for every other
    column, its mean (numeric) or most common level's share (categorical) among rows where `target` is missing vs
    present, and the standardized difference for numeric columns."""
    if target is None:
        n = len(df)
        miss = df.isna().sum()
        return pd.DataFrame({"n_missing": miss, "frac_missing": miss / max(n, 1)}).sort_values("n_missing", ascending=False)
    m = df[target].isna().to_numpy()
    rows = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        row = {"column": c}
        if pd.api.types.is_numeric_dtype(s):
            a, b = s[m].to_numpy(dtype=float), s[~m].to_numpy(dtype=float)
            row.update({"mean_when_missing": float(np.nanmean(a)) if np.isfinite(a).any() else np.nan,
                        "mean_when_present": float(np.nanmean(b)) if np.isfinite(b).any() else np.nan,
                        "smd": _smd(a, b)})
        else:
            top = s.astype(str).value_counts().index[:1]
            if len(top):
                lv = top[0]
                row.update({"level": lv, "share_when_missing": float((s[m].astype(str) == lv).mean()) if m.any() else np.nan,
                            "share_when_present": float((s[~m].astype(str) == lv).mean()) if (~m).any() else np.nan})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("column")
    out.attrs["n_missing_target"] = int(m.sum())
    out.attrs["n_rows"] = int(len(m))
    return out


def selection_summary(df: pd.DataFrame, selected, covariates=None) -> pd.DataFrame:
    """Rows where `selected` is true vs false: their counts (in .attrs) and each covariate's mean in both, with the
    standardized difference -- balance() with the selection indicator as the group."""
    s = _col(df, selected, "selected")
    sel = s.astype(bool) if not pd.api.types.is_bool_dtype(s) else s
    tmp = df.copy()
    key = "__selected__"
    tmp[key] = sel.values
    if covariates is None:
        covariates = [c for c in df.columns if c != (selected if isinstance(selected, str) else None)]
    out = balance(tmp, key, covariates)
    out.attrs["n_selected"] = int(sel.sum())
    out.attrs["n_total"] = int(len(sel))
    return out


def trend(*args, **kw) -> dict:
    """trend([df,] t, y). Linear trend of y over t: slope, its SE and p-value, R^2, the slope in the first and second half of t, and the
    lag-1 autocorrelation of the residuals (rows ordered by t)."""
    from scipy import stats
    df, (t, y) = _split_df(args, kw, ["t", "y"])
    t = pd.to_numeric(_col(df, t, "t"), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(_col(df, y, "y"), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    o = np.argsort(t)
    t, y = t[o], y[o]
    if len(t) < 4:
        return {"n": int(len(t))}
    r = stats.linregress(t, y)
    resid = y - (r.intercept + r.slope * t)
    ac = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 2 and resid.std() > 0 else float("nan")
    h = len(t) // 2
    s1 = stats.linregress(t[:h], y[:h]).slope if h >= 3 else float("nan")
    s2 = stats.linregress(t[h:], y[h:]).slope if len(t) - h >= 3 else float("nan")
    return {"n": int(len(t)), "slope": float(r.slope), "slope_se": float(r.stderr), "p": float(r.pvalue),
            "r2": float(r.rvalue ** 2), "slope_first_half": float(s1), "slope_second_half": float(s2),
            "resid_lag1_autocorr": ac}


def floor_summary(*args, **kw) -> dict:
    """floor_summary([df,] x). n, missing, zeros, negatives, the minimum and how many values equal it, the number of distinct values, and the
    five most repeated values with their counts."""
    df, (x,) = _split_df(args, kw, ["x"])
    s = pd.to_numeric(_col(df, x, "x"), errors="coerce")
    a = s.dropna().to_numpy(dtype=float)
    if len(a) == 0:
        return {"n": int(len(s)), "n_missing": int(s.isna().sum())}
    vc = pd.Series(a).value_counts().head(5)
    mn = float(a.min())
    return {"n": int(len(s)), "n_missing": int(s.isna().sum()), "n_zero": int((a == 0).sum()),
            "n_negative": int((a < 0).sum()), "min": mn, "n_at_min": int((a == mn).sum()),
            "frac_at_min": float((a == mn).mean()), "n_distinct": int(len(np.unique(a))),
            "most_repeated": {float(k): int(v) for k, v in vc.items()}}


def duplicates(df: pd.DataFrame, subset=None) -> dict:
    """Fully duplicated rows, and (with `subset`, a column or list of columns) rows sharing the same key: counts and
    up to five example keys with their multiplicities."""
    out = {"n_rows": int(len(df)), "n_duplicated_rows": int(df.duplicated().sum())}
    if subset is not None:
        cols = [subset] if isinstance(subset, str) else list(subset)
        vc = df.groupby(cols, dropna=False).size()
        rep = vc[vc > 1].sort_values(ascending=False)
        out.update({"n_keys": int(len(vc)), "n_keys_repeated": int(len(rep)),
                    "rows_in_repeated_keys": int(rep.sum()),
                    "examples": {str(k): int(v) for k, v in rep.head(5).items()}})
    return out
