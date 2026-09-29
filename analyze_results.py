#!/usr/bin/env python3
"""
Cross-model analysis of the accuracy (Track 1) and latent (Track 2) results.

Reads every ``results/accuracy_*.jsonl`` and ``results/latent_*.jsonl`` file
(one per model), joins the two tracks by model label, and produces:

  results/summary.csv                     one row per model with the metrics
                                          the report cares about (baseline
                                          accuracy, accuracy by variant kind,
                                          degradation vs baseline, answer
                                          consistency, latent extremes).
  results/plots/degradation_by_transform.png
                                          grouped bar chart of degradation
                                          points per variant kind, one group
                                          per model (the GSM-Symbolic story).
  results/plots/accuracy_vs_latent.png    scatter of baseline accuracy against
                                          the min-entropy floor, labelled by
                                          model (the LPP story).

The same table is also printed to the console.

Aggregation reuses ``src.metrics.summarize`` rather than duplicating the
per-kind / consistency / degradation math.

Run from the project root, after at least one accuracy run has finished::

    python analyze_results.py

Works with one model or many; kinds that only some models produced show up
as blanks in the CSV and are skipped in that model's bar group.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")   # save PNGs without needing a display
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.metrics import summarize


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _model_from_name(path: str, prefix: str) -> str:
    """results/accuracy_llama-1b.jsonl -> 'llama-1b'."""
    base = os.path.basename(path)
    assert base.startswith(prefix) and base.endswith(".jsonl")
    return base[len(prefix):-len(".jsonl")]


def load_accuracy(results_dir: str) -> dict:
    """model_label -> list[record] for every accuracy_*.jsonl found."""
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "accuracy_*.jsonl"))):
        recs = _read_jsonl(p)
        if recs:
            out[_model_from_name(p, "accuracy_")] = recs
    return out


def load_latent(results_dir: str) -> dict:
    """model_label -> list[record] for every latent_*.jsonl found."""
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "latent_*.jsonl"))):
        recs = _read_jsonl(p)
        if recs:
            out[_model_from_name(p, "latent_")] = recs
    return out


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------
def _latent_extremes(records: list) -> dict:
    """min entropy, max ER, max PR across all variants — matches run_latent.py."""
    return {
        "min_entropy": min(r["min_entropy"] for r in records),
        "max_ER":      max(r["max_ER"]      for r in records),
        "max_PR":      max(r["max_PR"]      for r in records),
    }


def build_table(acc_by_model: dict, lat_by_model: dict) -> pd.DataFrame:
    """One row per model. Columns are sorted so kinds stay adjacent."""
    # summarize() takes a flat list of records tagged with 'model'
    flat = [r for recs in acc_by_model.values() for r in recs]
    acc_summary = summarize(flat) if flat else {}

    # every kind that turned up in any model, so columns line up across rows
    all_kinds = set()
    for s in acc_summary.values():
        all_kinds.update(s["accuracy_by_kind"].keys())
    # baseline first, then the rest alphabetically
    kinds = (["baseline"] if "baseline" in all_kinds else []) \
            + sorted(k for k in all_kinds if k != "baseline")

    models = sorted(set(acc_by_model) | set(lat_by_model))
    rows = []
    for m in models:
        row: dict = {"model": m}
        s = acc_summary.get(m)
        if s:
            row["baseline_accuracy"] = s["baseline_accuracy"]
            row["answer_consistency"] = s["answer_consistency"]
            row["n_families"] = s["n_families"]
            row["n_records"] = s["n_records"]
            for k in kinds:
                row[f"accuracy_{k}"] = s["accuracy_by_kind"].get(k)
            for k in kinds:
                if k == "baseline":
                    continue
                row[f"degradation_pts_{k}"] = s["degradation_points_by_kind"].get(k)
        lat = lat_by_model.get(m)
        if lat:
            ex = _latent_extremes(lat)
            row["min_entropy"] = round(ex["min_entropy"], 4)
            row["max_ER"] = round(ex["max_ER"], 2)
            row["max_PR"] = round(ex["max_PR"], 2)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_degradation(df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar: one cluster per variant kind, one bar per model."""
    deg_cols = [c for c in df.columns if c.startswith("degradation_pts_")]
    if not deg_cols:
        print("  (no degradation columns yet — skipping degradation plot)")
        return
    kinds = [c[len("degradation_pts_"):] for c in deg_cols]
    models = df["model"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(kinds) + 2), 4.5))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)
    x = list(range(len(kinds)))

    for i, m in enumerate(models):
        vals = [df.loc[df["model"] == m, c].iloc[0] for c in deg_cols]
        vals = [0.0 if pd.isna(v) else v for v in vals]
        offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=20, ha="right")
    ax.set_ylabel("degradation vs baseline (percentage points)")
    ax.set_title("Robustness: accuracy drop by surface transform")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(title="model", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_latent(df: pd.DataFrame, out_path: str) -> None:
    """Scatter: baseline accuracy vs min-entropy floor, one labelled point per model."""
    if "baseline_accuracy" not in df.columns or "min_entropy" not in df.columns:
        print("  (need both accuracy and latent results — skipping scatter)")
        return
    sub = df.dropna(subset=["baseline_accuracy", "min_entropy"])
    if sub.empty:
        print("  (no model has both tracks yet — skipping scatter)")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(sub["baseline_accuracy"], sub["min_entropy"], s=60)
    for _, r in sub.iterrows():
        ax.annotate(r["model"],
                    (r["baseline_accuracy"], r["min_entropy"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("baseline accuracy")
    ax.set_ylabel("uncertainty floor (min next-token entropy, nats)")
    ax.set_title("Latent profile vs task performance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    results_dir = config.RESULTS_DIR
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    acc = load_accuracy(results_dir)
    lat = load_latent(results_dir)
    if not acc and not lat:
        print(f"No result files in {results_dir}. Run run_experiments.py or "
              f"run_latent.py first.")
        return 1

    df = build_table(acc, lat)

    csv_path = os.path.join(results_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    # console: keep it readable even with many columns
    with pd.option_context("display.max_columns", None,
                           "display.width", 200):
        print(df.to_string(index=False))
    print(f"\nWrote {csv_path}")

    deg_png = os.path.join(plots_dir, "degradation_by_transform.png")
    plot_degradation(df, deg_png)
    if os.path.exists(deg_png):
        print(f"Wrote {deg_png}")

    scatter_png = os.path.join(plots_dir, "accuracy_vs_latent.png")
    plot_accuracy_vs_latent(df, scatter_png)
    if os.path.exists(scatter_png):
        print(f"Wrote {scatter_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
