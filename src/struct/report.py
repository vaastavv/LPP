"""
Assemble the full CodeLLM-paper deliverable from saved results:

  * tables/paper_tables.md   -- all tables I-VIII as Markdown
  * tables/paper_tables.tex  -- all tables I-VIII as LaTeX (booktabs)
  * results/struct/REPORT.md -- narrative report mapping numbers to the RQs,
                                with figures and provenance caveats
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from . import tables as T
from . import downstream_tables as DT


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def _provenance(struct_dir: str) -> dict:
    models = []
    synthetic_perf = False
    for p in sorted(glob.glob(os.path.join(struct_dir, "struct_*.json"))):
        with open(p) as f:
            models.append(json.load(f)["model"])
    for p in glob.glob(os.path.join(struct_dir, "downstream_*.json")):
        with open(p) as f:
            if json.load(f).get("synthetic"):
                synthetic_perf = True
    random_weights = any("tiny" in m or "random" in m for m in models)
    return {"models": models, "synthetic_perf": synthetic_perf,
            "random_weights": random_weights}


def build(struct_dir: str, out_dir: str) -> str:
    prov = _provenance(struct_dir)
    t = T.write_all(struct_dir, out_dir)
    dt = DT.write_all(struct_dir, out_dir)

    # combined LaTeX
    tex_parts = []
    for name in ["table1_models", "table1_corpus", "table2_separability",
                 "table3_peaks", "table4_emergence", "table5_strata",
                 "table6_association", "table7_regression", "table8_partial"]:
        path = os.path.join(out_dir, f"{name}.tex")
        if os.path.exists(path):
            tex_parts.append(open(path).read())
    with open(os.path.join(out_dir, "paper_tables.tex"), "w") as f:
        f.write("\n".join(tex_parts))

    # combined Markdown
    md = []
    md.append("# CodeLLM paper — computed tables\n")
    _banner(md, prov)
    md.append("\n## Table I — Characteristics of the evaluated CodeLLMs\n")
    md.append(_md(t["table1"]))
    md.append("\n\n**Corpus**\n")
    md.append(_md(t["corpus"]))
    md.append("\n\n## Table II — Overall structural separability\n")
    md.append(_md(t["table2"]))
    md.append("\n\n## Table III — Maximum structural representation locations\n")
    md.append(_md(t["table3"]))
    md.append("\n\n## Table IV — Layer-wise structural emergence\n")
    md.append(_md(t["table4"]))
    md.append("\n\n## Table V — DFBS across token-distance strata\n")
    md.append(_md(t["table5"]))
    if "table6" in dt:
        md.append("\n\n## Table VI — Association with downstream SE performance\n")
        md.append(_md(dt["table6"]))
        md.append("\n\n## Table VII — Regression analysis\n")
        md.append(_md(dt["table7"]))
        md.append("\n\n## Table VIII — Partial association controlling for model scale\n")
        md.append(_md(dt["table8"]))
    else:
        md.append("\n\n_Tables VI-VIII require >=2 models with downstream scores._\n")
    tables_md = "\n".join(md)
    with open(os.path.join(out_dir, "paper_tables.md"), "w") as f:
        f.write(tables_md)

    # narrative report
    report = _report(struct_dir, out_dir, prov, t, dt)
    rpath = os.path.join(struct_dir, "REPORT.md")
    with open(rpath, "w") as f:
        f.write(report)
    return rpath


def _banner(md, prov):
    if prov["random_weights"] or prov["synthetic_perf"]:
        md.append("> **PROVENANCE / CAVEATS**\n>\n")
        if prov["random_weights"]:
            md.append("> - Structural metrics below were computed on a **randomly "
                      "initialised tiny model panel** (pipeline smoke-test). The "
                      "numbers are real outputs of the pipeline but have **no "
                      "scientific meaning**; run pretrained CodeLLMs for real values.\n>\n")
        if prov["synthetic_perf"]:
            md.append("> - Downstream scores (Tables VI-VIII, Fig 4) are "
                      "**ILLUSTRATIVE synthetic placeholders** (stamped "
                      "`synthetic:true`), present only to demonstrate the table "
                      "format. Replace with real benchmark runs.\n>\n")


def _report(struct_dir, out_dir, prov, t, dt) -> str:
    lines = []
    lines.append("# Latent Structural Property Profiling in Code LLMs — Results report\n")
    lines.append("Implementation of Meher & Mall, *Latent Structural Property "
                 "Profiling in Code Large Language Model*. This report is "
                 "regenerated from `results/struct/` by `src.struct.report`.\n")
    _banner(lines, prov)
    lines.append(f"\n**Panel:** {', '.join(prov['models'])}\n")
    lines.append("\n## Research questions and where they are answered\n")
    lines.append(
        "- **RQ1 (do CodeLLMs encode structure?)** → Table II (contrast vs "
        "control separability, effect sizes, BH-adjusted p).\n"
        "- **RQ2 (at which layers?)** → Table III (peak layers $l^*$, normalized "
        "depth $\\lambda^*$), Table IV (cross-dimension emergence), Fig 3 "
        "(layer-wise profiles).\n"
        "- **RQ3 (association with SE tasks?)** → Tables VI-VIII, Fig 4.\n"
        "- **RQ4 (causal components)** → not in this build (scope: Tables I-VIII).\n")
    lines.append("\n## Figures\n")
    lines.append("- `results/struct/plots/figure3_layerwise.png` — SRS/CFS/DFBS vs normalized depth.\n")
    lines.append("- `results/struct/plots/figure4_assoc.png` — latent metric vs downstream performance.\n")
    lines.append("\n## Tables\n")
    lines.append("All tables are in `tables/` as CSV, LaTeX (`paper_tables.tex`), "
                 "and Markdown (`paper_tables.md`). Summary:\n")
    lines.append("\n### Table II — Overall structural separability\n")
    lines.append(_md(t["table2"]))
    lines.append("\n\n### Table III — Peak layers\n")
    lines.append(_md(t["table3"]))
    if "table6" in dt:
        lines.append("\n\n### Table VI — Downstream association\n")
        lines.append(_md(dt["table6"]))
    lines.append("\n\n## Reproducing with real models\n")
    lines.append("See `docs/CODELLM_README.md` and `scripts/run_colab.py`. The "
                 "sweep is: `run_struct.py` + `run_downstream.py` per model, then "
                 "`run_tables.py` to rebuild every table and figure.\n")
    return "\n".join(lines)
