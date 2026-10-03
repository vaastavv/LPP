"""
Tables VI-VIII (RQ3): association between latent structural metrics and
downstream SE task performance.

Consumes the per-model structural results (for peak SRS/CFS/DFBS) and the
per-model downstream score files, and builds:

  Table VI   -- Spearman association of each metric with each task + best predictor
  Table VII  -- OLS regression (task ~ metric) for the dependency-sensitive tasks
  Table VIII -- partial correlation controlling for model scale (parameters)
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from .metrics import load_result
from .downstream import TASKS, TASK_METRIC
from . import stats as S
from .tables import _f, _p, _ci, to_latex


def _peak_metric(res, q: str) -> float:
    arr = {"SRS": res.SRS, "CFS": res.CFS, "DFBS": res.DFBS}[q]
    if arr is None or len(arr) <= 1:
        return float("nan")
    return float(np.max(arr[1:]))


def _parse_params(s: str) -> float:
    if not s:
        return float("nan")
    m = re.search(r"([\d.]+)\s*([MB])", s)
    if not m:
        return float("nan")
    val = float(m.group(1))
    return val * (1e6 if m.group(2) == "M" else 1e9)


def load_downstream(struct_dir: str):
    """Return (models, metric[model][q], perf[model][task], params[model])."""
    results = {}
    for path in sorted(glob.glob(os.path.join(struct_dir, "struct_*.json"))):
        r = load_result(path)
        results[r.model] = r
    perf, params = {}, {}
    for path in sorted(glob.glob(os.path.join(struct_dir, "downstream_*.json"))):
        with open(path) as f:
            d = json.load(f)
        perf[d["model"]] = d["scores"]
    metric = {}
    for model, r in results.items():
        metric[model] = {q: _peak_metric(r, q) for q in ("SRS", "CFS", "DFBS")}
        key = None
        for p in glob.glob(os.path.join(struct_dir, "modelinfo_*.json")):
            with open(p) as f:
                info = json.load(f)
            if info.get("model_id") == model:
                params[model] = _parse_params(info.get("params", ""))
                break
    models = [m for m in results if m in perf]
    return models, metric, perf, params


def table6(models, metric, perf) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        y = np.array([perf[m].get(task, np.nan) for m in models], float)
        cell = {"Task": task.replace("_", " ").title(),
                "Performance Metric": TASK_METRIC[task]}
        rhos = {}
        for q in ("SRS", "CFS", "DFBS"):
            x = np.array([metric[m][q] for m in models], float)
            good = ~(np.isnan(x) | np.isnan(y))
            rho, _ = S.spearman(x[good], y[good]) if good.sum() >= 3 else (np.nan, np.nan)
            rhos[q] = rho
            cell[q] = _f(rho, 2)
        valid = {q: v for q, v in rhos.items() if not np.isnan(v)}
        best = max(valid, key=lambda q: abs(valid[q])) if valid else "--"
        cell["Best Latent Predictor"] = best
        rows.append(cell)
    return pd.DataFrame(rows)


def table7(models, metric, perf) -> pd.DataFrame:
    rows = []
    reg_tasks = ["bug_localization", "program_repair"]
    for task in reg_tasks:
        y = np.array([perf[m].get(task, np.nan) for m in models], float)
        for q in ("SRS", "CFS", "DFBS"):
            x = np.array([metric[m][q] for m in models], float)
            good = ~(np.isnan(x) | np.isnan(y))
            fit = S.ols_fit(y[good], x[good]) if good.sum() >= 3 else None
            if fit is None:
                fit = {"beta": np.nan, "se": np.nan, "ci": (np.nan, np.nan),
                       "p": np.nan, "r2": np.nan}
            rows.append({
                "Task": task.replace("_", " ").title(),
                "Predictor": q,
                "beta": _f(fit["beta"], 2),
                "Standard Error": _f(fit["se"], 2),
                "95% CI": _ci(*fit["ci"], nd=2),
                "p-value": _p(fit["p"]),
                "Partial R2": _f(fit["r2"], 2),
            })
    return pd.DataFrame(rows)


def table8(models, metric, perf, params) -> pd.DataFrame:
    spec = [("bug_localization", "DFBS"), ("bug_localization", "CFS"),
            ("program_repair", "DFBS"), ("code_summarization", "SRS")]
    rows = []
    pvals, cache = [], []
    for task, q in spec:
        y = np.array([perf[m].get(task, np.nan) for m in models], float)
        x = np.array([metric[m][q] for m in models], float)
        z = np.array([np.log10(params.get(m, np.nan)) for m in models], float)
        good = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        raw, _ = S.spearman(x[good], y[good]) if good.sum() >= 3 else (np.nan, np.nan)
        part, pp = S.partial_spearman(x[good], y[good], z[good]) if good.sum() >= 4 else (np.nan, np.nan)
        pvals.append(pp)
        cache.append((task, q, raw, part))
    adj = S.benjamini_hochberg(pvals)
    for (task, q, raw, part), ap in zip(cache, adj):
        rows.append({
            "Task": task.replace("_", " ").title(),
            "Latent Metric": q,
            "Raw Correlation": _f(raw, 2),
            "Partial Correlation": _f(part, 2),
            "Control Variables": "Parameters",
            "p-value": _p(ap),
        })
    return pd.DataFrame(rows)


def write_all(struct_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    models, metric, perf, params = load_downstream(struct_dir)
    if len(models) < 2:
        return {"models": models}
    t6 = table6(models, metric, perf)
    t7 = table7(models, metric, perf)
    t8 = table8(models, metric, perf, params)
    specs = [
        ("table6_association", t6, "Association between latent structural metrics and downstream SE performance.", "tab:assoc"),
        ("table7_regression", t7, "Regression analysis of latent metrics and downstream task performance.", "tab:reg"),
        ("table8_partial", t8, "Partial association after controlling for model scale.", "tab:partial"),
    ]
    tex_all = []
    for name, df, cap, lab in specs:
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        tex = to_latex(df, cap, lab)
        with open(os.path.join(out_dir, f"{name}.tex"), "w") as f:
            f.write(tex)
        tex_all.append(tex)
    with open(os.path.join(out_dir, "tables_VI_VIII.tex"), "w") as f:
        f.write("\n".join(tex_all))
    return {"table6": t6, "table7": t7, "table8": t8,
            "models": models, "metric": metric, "perf": perf}
