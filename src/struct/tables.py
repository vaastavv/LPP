"""
Build the paper's tables from saved structural results.

Tables I-V (RQ1/RQ2) are built here from `results/struct/struct_*.json` plus
the per-model metadata files. Tables VI-VIII (RQ3) are built in
`src.struct.downstream_tables` because they additionally need downstream task
scores.

Each builder returns a pandas DataFrame; `write_all` also emits a LaTeX block
(booktabs, matching the paper's table style) and a CSV per table.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from .metrics import load_result, StructResult
from .corpus import build_corpus, partition_corpus
from . import stats as S


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all(struct_dir: str) -> tuple[list[StructResult], dict]:
    results, info = [], {}
    for path in sorted(glob.glob(os.path.join(struct_dir, "struct_*.json"))):
        res = load_result(path)
        results.append(res)
        key = os.path.basename(path)[len("struct_"):-len(".json")]
        info_path = os.path.join(struct_dir, f"modelinfo_{key}.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info[res.model] = json.load(f)
    return results, info


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def _p(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def _ci(lo, hi, nd=3):
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (lo, hi)):
        return "--"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _peak_layer(res: StructResult, q: str) -> int:
    return int(res.peak[q]["l_star"])


# ---------------------------------------------------------------------------
# Table I -- model & corpus characteristics
# ---------------------------------------------------------------------------
def table1(results, info) -> pd.DataFrame:
    rows = []
    for res in results:
        m = info.get(res.model, {})
        rows.append({
            "Model": res.model,
            "Family": m.get("family", "?"),
            "Parameters": m.get("params", "?"),
            "Layers": m.get("layers", res.L),
            "Hidden Size": m.get("hidden", "?"),
            "Context Length": m.get("context", "?"),
            "Language Coverage": m.get("langs", "?"),
        })
    return pd.DataFrame(rows)


def corpus_stats() -> dict:
    from .transforms import make_contrast_pairs
    from .extract import extract_struct
    progs = build_corpus()
    sp = partition_corpus(progs)
    n_syn = n_cf = n_du = 0
    for p in progs:
        pr = make_contrast_pairs(p.source)
        if pr["syn_pos"]:
            n_syn += 1
        if pr["cf_pos"]:
            n_cf += 1
        try:
            n_du += len(extract_struct(p.source).def_uses)
        except SyntaxError:
            pass
    return {
        "Source programs": len(progs),
        "Syntactic contrast pairs": n_syn,
        "Control-flow contrast pairs": n_cf,
        "Definition-use pairs": n_du,
        "Repository-level partitions": len({p.repo for p in progs}),
        "Programming languages": 1,
    }


# ---------------------------------------------------------------------------
# Table II -- overall structural separability (pooled across models at peak)
# ---------------------------------------------------------------------------
def table2(results) -> pd.DataFrame:
    rows = []
    pvals = []
    raw = {}

    # syntax & control flow: independent P+ vs P- at each model's peak layer
    for dim, pos_attr, ctrl_attr, pkey in [
        ("Syntax", "srs_pos", "srs_ctrl", "syn"),
        ("Control flow", "cfs_pos", "cfs_ctrl", "cf"),
    ]:
        pos_all, ctrl_all = [], []
        for res in results:
            l = _peak_layer(res, pkey)
            pos = getattr(res, pos_attr); ctrl = getattr(res, ctrl_attr)
            if pos is not None and pos.shape[0]:
                pos_all.extend(pos[:, l].tolist())
            if ctrl is not None and ctrl.shape[0]:
                ctrl_all.extend(ctrl[:, l].tolist())
        pos_all = np.array(pos_all); ctrl_all = np.array(ctrl_all)
        diff = pos_all.mean() - ctrl_all.mean() if len(pos_all) and len(ctrl_all) else np.nan
        d = S.cohens_d(pos_all, ctrl_all)
        _, p = S.mannwhitney(pos_all, ctrl_all)
        pvals.append(p)
        raw[dim] = (pos_all.mean() if len(pos_all) else np.nan,
                    ctrl_all.mean() if len(ctrl_all) else np.nan, diff, d)

    # data flow: paired B vs N at each model's DF peak layer
    B_all, N_all = [], []
    for res in results:
        l = _peak_layer(res, "df")
        for r in res.dfbs_records:
            B_all.append(float(r.B_by_layer[l]))
            N_all.append(float(r.N_by_layer[l]))
    B_all = np.array(B_all); N_all = np.array(N_all)
    df_diff = (B_all - N_all).mean() if len(B_all) else np.nan
    df_d = S.cohens_dz(B_all - N_all)
    _, df_p = S.wilcoxon(B_all - N_all)
    pvals.append(df_p)
    raw["Data flow"] = (B_all.mean() if len(B_all) else np.nan,
                        N_all.mean() if len(N_all) else np.nan, df_diff, df_d)

    adj = S.benjamini_hochberg(pvals)
    for (dim, (contrast, control, diff, d)), ap in zip(raw.items(), adj):
        rows.append({
            "Structural Dimension": dim,
            "Contrast Condition": _f(contrast),
            "Control Condition": _f(control),
            "Mean Diff.": _f(diff),
            "Effect Size": _f(d, 2),
            "Adjusted p-value": _p(ap),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table III -- maximum structural representation locations per model
# ---------------------------------------------------------------------------
def table3(results) -> pd.DataFrame:
    rows = []
    for res in results:
        rows.append({
            "Model": res.model,
            "l*_syn": res.peak["syn"]["l_star"],
            "lambda*_syn": _f(res.peak["syn"]["lambda_star"], 2),
            "l*_cf": res.peak["cf"]["l_star"],
            "lambda*_cf": _f(res.peak["cf"]["lambda_star"], 2),
            "l*_df": res.peak["df"]["l_star"],
            "lambda*_df": _f(res.peak["df"]["lambda_star"], 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table IV -- layer-wise structural emergence (normalized-depth comparisons)
# ---------------------------------------------------------------------------
def table4(results) -> pd.DataFrame:
    lam = {q: np.array([res.peak[q]["lambda_star"] for res in results])
           for q in ("syn", "cf", "df")}
    comparisons = [("Syntax vs control flow", "syn", "cf"),
                   ("Syntax vs data flow", "syn", "df"),
                   ("Control flow vs data flow", "cf", "df")]
    rows = []
    pvals = []
    cache = []
    for name, a, b in comparisons:
        diff = lam[a] - lam[b]
        mean_diff = float(diff.mean())
        lo, hi = S.bootstrap_ci(diff)
        w, p = S.wilcoxon(diff)
        if np.isnan(p):                       # fall back to paired t
            w, p = S.paired_t(diff)
        eff = S.cohens_dz(diff)
        pvals.append(p)
        cache.append((name, mean_diff, lo, hi, w, eff))
    adj = S.benjamini_hochberg(pvals)
    for (name, md, lo, hi, w, eff), ap in zip(cache, adj):
        rows.append({
            "Comparison": name,
            "Mean Diff.": _f(md, 3),
            "95% CI": _ci(lo, hi),
            "Test Statistic": _f(w, 2),
            "Adjusted p-value": _p(ap),
            "Effect Size": _f(eff, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table V -- DFBS across token-distance strata
# ---------------------------------------------------------------------------
_STRATA = [("1-10 tokens", 1, 10), ("11-25 tokens", 11, 25),
           ("26-50 tokens", 26, 50), ("51-100 tokens", 51, 100),
           (">100 tokens", 101, 10**9)]


def table5(results) -> pd.DataFrame:
    # gather (tok_dist, B_peak, N_peak) pooled across models at each DF peak
    recs = []
    for res in results:
        l = _peak_layer(res, "df")
        for r in res.dfbs_records:
            recs.append((r.tok_dist, float(r.B_by_layer[l]), float(r.N_by_layer[l])))
    rows = []
    pvals = []
    cache = []
    for name, lo_d, hi_d in _STRATA:
        sel = [(b, n) for (d, b, n) in recs if lo_d <= d <= hi_d]
        if not sel:
            cache.append((name, np.nan, np.nan, np.nan, (np.nan, np.nan)))
            pvals.append(np.nan)
            continue
        B = np.array([b for b, n in sel]); N = np.array([n for b, n in sel])
        dfbs = B - N
        lo, hi = S.bootstrap_ci(dfbs)
        _, p = S.wilcoxon(dfbs)
        pvals.append(p)
        cache.append((name, float(B.mean()), float(N.mean()),
                      float(dfbs.mean()), (lo, hi)))
    adj = S.benjamini_hochberg(pvals)
    for (name, pb, nb, dfbs, (lo, hi)), ap in zip(cache, adj):
        rows.append({
            "Distance Stratum": name,
            "Positive Binding": _f(pb),
            "Negative Binding": _f(nb),
            "DFBS": _f(dfbs),
            "95% CI": _ci(lo, hi),
            "Adjusted p-value": _p(ap),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX emission (booktabs)
# ---------------------------------------------------------------------------
_HEADER_TEX = {
    "l*_syn": r"$l^{*}_{\mathrm{syn}}$", "lambda*_syn": r"$\lambda^{*}_{\mathrm{syn}}$",
    "l*_cf": r"$l^{*}_{\mathrm{cf}}$", "lambda*_cf": r"$\lambda^{*}_{\mathrm{cf}}$",
    "l*_df": r"$l^{*}_{\mathrm{df}}$", "lambda*_df": r"$\lambda^{*}_{\mathrm{df}}$",
    "Partial R2": r"Partial $R^{2}$", "beta": r"$\beta$",
    "p-value": r"$p$-value", "Adjusted p-value": r"Adjusted $p$-value",
}


def _header_tex(c: str) -> str:
    return _HEADER_TEX.get(c, _tex_escape(c))


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    spec = "l" + "r" * (len(cols) - 1)
    head = " & ".join(_header_tex(c) for c in cols) + r" \\"
    body = []
    for _, row in df.iterrows():
        body.append(" & ".join(_tex_escape(str(row[c])) for c in cols) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
        head, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}", "",
    ])


def _tex_escape(s: str) -> str:
    s = str(s)
    for a, b in [("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}")]:
        s = s.replace(a, b)
    return s


def write_all(struct_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    results, info = load_all(struct_dir)
    if not results:
        raise SystemExit(f"no struct results in {struct_dir}")

    t1 = table1(results, info)
    cstats = corpus_stats()
    t2 = table2(results)
    t3 = table3(results)
    t4 = table4(results)
    t5 = table5(results)

    specs = [
        ("table1_models", t1, "Characteristics of the evaluated CodeLLMs.", "tab:models"),
        ("table2_separability", t2, "Overall structural separability across the evaluated CodeLLMs.", "tab:sep"),
        ("table3_peaks", t3, "Maximum structural representation locations.", "tab:peaks"),
        ("table4_emergence", t4, "Statistical analysis of layer-wise structural emergence.", "tab:emerge"),
        ("table5_strata", t5, "Data-flow binding across token-distance strata.", "tab:strata"),
    ]
    tex_all = []
    for name, df, cap, lab in specs:
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        tex = to_latex(df, cap, lab)
        with open(os.path.join(out_dir, f"{name}.tex"), "w") as f:
            f.write(tex)
        tex_all.append(tex)

    # corpus sub-table (Table I lower block)
    corpus_df = pd.DataFrame([{"Corpus Component": k, "Number": v}
                              for k, v in cstats.items()])
    corpus_df.to_csv(os.path.join(out_dir, "table1_corpus.csv"), index=False)
    with open(os.path.join(out_dir, "table1_corpus.tex"), "w") as f:
        f.write(to_latex(corpus_df, "Program corpus used in the study.", "tab:corpus"))

    with open(os.path.join(out_dir, "tables_I_V.tex"), "w") as f:
        f.write("\n".join(tex_all))

    return {"table1": t1, "corpus": corpus_df, "table2": t2,
            "table3": t3, "table4": t4, "table5": t5}
