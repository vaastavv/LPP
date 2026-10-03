"""
Statistical helpers for the structural-profiling tables: bootstrap confidence
intervals, effect sizes, paired/unpaired tests, Benjamini-Hochberg correction,
and (partial) rank correlations.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _ss
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Independent-samples Cohen's d with pooled SD."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
    s = np.sqrt(pooled)
    return float((a.mean() - b.mean()) / s) if s > 1e-12 else 0.0


def cohens_dz(diff: np.ndarray) -> float:
    """Paired Cohen's dz from a vector of differences."""
    diff = np.asarray(diff, float)
    if len(diff) < 2:
        return 0.0
    s = diff.std(ddof=1)
    return float(diff.mean() / s) if s > 1e-12 else 0.0


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        boots[i] = stat(x[rng.integers(0, n, n)])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def mannwhitney(a, b) -> tuple[float, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not _HAVE_SCIPY or len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    try:
        u, p = _ss.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ValueError:
        return (float("nan"), float("nan"))


def wilcoxon(diff) -> tuple[float, float]:
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if not _HAVE_SCIPY or len(diff) < 2 or np.allclose(diff, 0):
        return (float("nan"), float("nan"))
    try:
        w, p = _ss.wilcoxon(diff)
        return float(w), float(p)
    except ValueError:
        return (float("nan"), float("nan"))


def paired_t(diff) -> tuple[float, float]:
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if not _HAVE_SCIPY or len(diff) < 2:
        return (float("nan"), float("nan"))
    t, p = _ss.ttest_1samp(diff, 0.0)
    return float(t), float(p)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Return BH-adjusted p-values; NaNs pass through."""
    p = np.asarray(pvals, float)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    pm = p[mask]
    m = len(pm)
    if m == 0:
        return out.tolist()
    order = np.argsort(pm)
    ranked = pm[order]
    adj = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    res = np.empty(m)
    res[order] = adj
    out[mask] = res
    return out.tolist()


def spearman(x, y) -> tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if not _HAVE_SCIPY or len(x) < 3:
        return (float("nan"), float("nan"))
    rho, p = _ss.spearmanr(x, y)
    return float(rho), float(p)


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman partial correlation of x,y controlling for z (rank-based)."""
    if not _HAVE_SCIPY or len(x) < 4:
        return (float("nan"), float("nan"))
    rx = _ss.rankdata(x); ry = _ss.rankdata(y); rz = _ss.rankdata(z)
    def resid(a, b):
        b1 = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex = resid(rx, rz); ey = resid(ry, rz)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return (float("nan"), float("nan"))
    r, p = _ss.pearsonr(ex, ey)
    return float(r), float(p)


def ols_fit(y, x):
    """
    Simple OLS of y ~ 1 + x (both standardized). Returns dict with beta, se,
    ci, p, partial_r2 (== r^2 here with a single predictor).
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(y)
    out = {"beta": float("nan"), "se": float("nan"),
           "ci": (float("nan"), float("nan")), "p": float("nan"),
           "r2": float("nan")}
    if n < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return out
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    X = np.vstack([np.ones(n), xs]).T
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    resid = ys - X @ coef
    dof = n - 2
    if dof <= 0:
        return out
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    beta = coef[1]; beta_se = se[1]
    out["beta"] = float(beta)
    out["se"] = float(beta_se)
    ss_tot = (ys @ ys)
    ss_res = (resid @ resid)
    out["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    if _HAVE_SCIPY and beta_se > 1e-12:
        tval = beta / beta_se
        p = 2 * (1 - _ss.t.cdf(abs(tval), dof))
        tcrit = _ss.t.ppf(0.975, dof)
        out["p"] = float(p)
        out["ci"] = (float(beta - tcrit * beta_se), float(beta + tcrit * beta_se))
    return out
