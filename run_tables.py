#!/usr/bin/env python3
"""
Build every table (I-VIII) and figure (3-4) and the report from the saved
per-model results in results/struct/.

  python run_tables.py

Outputs:
  tables/table*.csv, tables/table*.tex, tables/paper_tables.{md,tex}
  results/struct/plots/figure3_layerwise.png, figure4_assoc.png
  results/struct/REPORT.md
"""
import os

import config
from src.struct.tables import load_all
from src.struct.plots import figure3, figure4
from src.struct.downstream_tables import load_downstream
from src.struct.downstream import TASKS
from src.struct.report import build

STRUCT = os.path.join(config.RESULTS_DIR, "struct")
TABLES = os.path.join(config.ROOT, "tables")
PLOTS = os.path.join(STRUCT, "plots")


def main():
    results, _ = load_all(STRUCT)
    if not results:
        raise SystemExit(f"No results in {STRUCT}. Run run_struct.py first.")

    # figures
    figure3(results, os.path.join(PLOTS, "figure3_layerwise.png"))
    models, metric, perf, _ = load_downstream(STRUCT)
    if len(models) >= 2 and perf:
        figure4(metric, perf,
                ["bug_localization", "program_repair", "code_summarization"],
                os.path.join(PLOTS, "figure4_assoc.png"))

    # tables + report
    rpath = build(STRUCT, TABLES)
    print(f"Built tables in {TABLES}/ and report at {rpath}")
    print(f"Figures in {PLOTS}/")


if __name__ == "__main__":
    main()
