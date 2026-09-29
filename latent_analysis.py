#!/usr/bin/env python3
"""
Post-hoc analysis of the latent (intrinsic) results — LPP paper items #2, #7,
#8, #9, #10, #4.

Pure post-processing over the JSONL files already written by run_latent.py,
run_latent_rolling.py, and run_calibration.py. Nothing here runs a model, so it
is fast and safe to re-run.

Produces (only for the data actually present):
  results/plots/layerwise_ER.png            #2  ER vs normalized layer depth
  results/plots/layerwise_PR.png            #2  PR vs normalized layer depth
  results/hourglass.csv                     #2  per-model hourglass scores
  results/plots/sensitivity_prefix.png      #7  metric vs prefix length
  results/plots/sensitivity_context.png     #8  metric vs context length
  results/plots/sensitivity_sample_size.png #9  metric vs sample size
  results/plots/sensitivity_dataset.png     #10 metric per dataset
  results/aggregation_invariance.csv        #4  ranking stability across aggregators

Run from the project root, after at least one latent / calibration run::

    python latent_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import config


def _get_plt():
    """Import matplotlib lazily so the non-plotting helpers (loaders,
    hourglass_score, aggregation_invariance) work without matplotlib installed."""
    import matplotlib
    matplotlib.use("Agg")   # save PNGs without a display
    import matplotlib.pyplot as plt
    return plt


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


def load_layer_records(results_dir: str) -> dict:
    """
    model -> list[record] for every file carrying per-layer ER/PR profiles:
    latent_<model>.jsonl (excluding latent_rolling_*) and calibration_*.jsonl.
    """
    out: dict[str, list] = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "latent_*.jsonl"))):
        base = os.path.basename(p)
        if base.startswith("latent_rolling_"):
            continue
        model = base[len("latent_"):-len(".jsonl")]
        recs = [r for r in _read_jsonl(p) if r.get("ER_by_layer")]
        if recs:
            out.setdefault(model, []).extend(recs)
    for p in sorted(glob.glob(os.path.join(results_dir, "calibration_*.jsonl"))):
        _, model, _ = _parse_calibration_name(p)
        recs = [r for r in _read_jsonl(p) if r.get("ER_by_layer")]
        if recs:
            out.setdefault(model, []).extend(recs)
    return out


def _parse_calibration_name(path: str) -> tuple[str, str, int | None]:
    """
    calibration_<dataset>_<model>[_ctx<L>].jsonl -> (dataset, model, ctx|None).
    Relies on dataset names and model labels containing no underscore.
    """
    base = os.path.basename(path)
    stem = base[len("calibration_"):-len(".jsonl")]
    parts = stem.split("_")
    dataset, model = parts[0], parts[1]
    ctx = None
    if len(parts) >= 3 and parts[2].startswith("ctx"):
        ctx = int(parts[2][len("ctx"):])
    return dataset, model, ctx


def load_rolling_records(results_dir: str) -> dict:
    """
    model -> list[record] carrying entropy_by_prefix, from
    latent_rolling_<model>.jsonl and any calibration_*.jsonl.
    """
    out: dict[str, list] = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "latent_rolling_*.jsonl"))):
        model = os.path.basename(p)[len("latent_rolling_"):-len(".jsonl")]
        recs = [r for r in _read_jsonl(p) if r.get("entropy_by_prefix")]
        if recs:
            out.setdefault(model, []).extend(recs)
    for p in sorted(glob.glob(os.path.join(results_dir, "calibration_*.jsonl"))):
        _, model, _ = _parse_calibration_name(p)
        recs = [r for r in _read_jsonl(p) if r.get("entropy_by_prefix")]
        if recs:
            out.setdefault(model, []).extend(recs)
    return out


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# #2 · Layerwise ER/PR + hourglass detector
# ---------------------------------------------------------------------------
def hourglass_score(vals: list[float]) -> float:
    """
    Positive when the endpoints are higher than the middle third — i.e. an
    hourglass dip. LPP paper Supplement §2.2, Figure 4 (page 20).
    """
    n = len(vals)
    if n == 0:
        return 0.0
    endpoints_mean = (vals[0] + vals[-1]) / 2
    mid = vals[n // 3: 2 * n // 3]
    middle_min = min(mid) if mid else vals[n // 2]
    return endpoints_mean - middle_min


def _mean_layer_curve(records: list, key: str) -> list[float]:
    """Elementwise mean of `key` (a per-layer list) over records of modal length."""
    lengths = [len(r[key]) for r in records if r.get(key)]
    if not lengths:
        return []
    modal_len = statistics.mode(lengths)
    curves = [r[key] for r in records if len(r.get(key, [])) == modal_len]
    return [ _mean(c[i] for c in curves) for i in range(modal_len) ]


def analyze_layerwise(layer_by_model: dict, results_dir: str, plots_dir: str) -> None:
    if not layer_by_model:
        print("  (no per-layer records — skipping layerwise/hourglass)")
        return

    curves = {}   # model -> (er_curve, pr_curve)
    for model, recs in sorted(layer_by_model.items()):
        er = _mean_layer_curve(recs, "ER_by_layer")
        pr = _mean_layer_curve(recs, "PR_by_layer")
        if er and pr:
            curves[model] = (er, pr)

    if not curves:
        print("  (per-layer records had no usable curves — skipping)")
        return

    plt = _get_plt()
    for idx, (metric, title, fname) in enumerate([
        ("ER", "Effective rank vs normalized layer depth", "layerwise_ER.png"),
        ("PR", "Participation ratio vs normalized layer depth", "layerwise_PR.png"),
    ]):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for model, (er, pr) in curves.items():
            vals = er if metric == "ER" else pr
            n = len(vals)
            depth = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
            ax.plot(depth, vals, marker=".", label=model)
        ax.set_xlabel("normalized layer depth (0 = embeddings, 1 = final)")
        ax.set_ylabel(f"{metric} (mean over prompts)")
        ax.set_title(title)
        ax.legend(title="model", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Wrote {os.path.join(plots_dir, fname)}")

    csv_path = os.path.join(results_dir, "hourglass.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,hourglass_ER,hourglass_PR\n")
        for model, (er, pr) in curves.items():
            f.write(f"{model},{hourglass_score(er):.4f},{hourglass_score(pr):.4f}\n")
    print(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# #7 · Prefix-length sensitivity
# ---------------------------------------------------------------------------
def _mean_by_prefix(records: list, key: str) -> dict[int, float]:
    """Average a {L: value} dict `key` across records -> {L: mean}."""
    acc: dict[int, list[float]] = {}
    for r in records:
        for k, v in r.get(key, {}).items():
            acc.setdefault(int(k), []).append(v)
    return {L: _mean(vs) for L, vs in sorted(acc.items())}


def plot_prefix_sensitivity(rolling_by_model: dict, plots_dir: str) -> None:
    """Figure 4A analogue: metric vs prefix length, one line per model. LPP §Datasets."""
    if not rolling_by_model:
        print("  (no rolling records — skipping prefix sensitivity)")
        return

    plt = _get_plt()
    panels = [
        ("entropy_by_prefix", "next-token entropy (nats)"),
        ("max_ER_by_prefix", "max effective rank"),
        ("max_PR_by_prefix", "max participation ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    drew = False
    for ax, (key, ylabel) in zip(axes, panels):
        for model, recs in sorted(rolling_by_model.items()):
            series = _mean_by_prefix(recs, key)
            if series:
                ax.plot(list(series.keys()), list(series.values()), marker="o", label=model)
                drew = True
        ax.set_xlabel("prefix length (tokens)")
        ax.set_ylabel(ylabel)
        ax.legend(title="model", frameon=False)
    if not drew:
        plt.close(fig)
        print("  (rolling records had no per-prefix series — skipping prefix plot)")
        return
    fig.suptitle("Prefix-length sensitivity (rolling schedule)")
    fig.tight_layout()
    out = os.path.join(plots_dir, "sensitivity_prefix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# #8 · Context-length sensitivity
# ---------------------------------------------------------------------------
def plot_context_sensitivity(results_dir: str, plots_dir: str) -> None:
    """Figure 4B analogue: metric vs context length. Reads calibration_*_ctx<L>.jsonl."""
    # model -> {ctx: mean_min_entropy}
    by_model: dict[str, dict[int, float]] = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "calibration_*_ctx*.jsonl"))):
        _, model, ctx = _parse_calibration_name(p)
        if ctx is None:
            continue
        recs = _read_jsonl(p)
        if recs:
            by_model.setdefault(model, {})[ctx] = _mean(r["min_entropy"] for r in recs)

    if not by_model:
        print("  (no calibration_*_ctx*.jsonl — skipping context sensitivity; run "
              "run_calibration.py --context-length {50,100,200,500})")
        return

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model, series in sorted(by_model.items()):
        xs = sorted(series)
        ax.plot(xs, [series[x] for x in xs], marker="o", label=model)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("uncertainty floor (mean min entropy, nats)")
    ax.set_title("Context-length sensitivity")
    ax.legend(title="model", frameon=False)
    fig.tight_layout()
    out = os.path.join(plots_dir, "sensitivity_context.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# #9 · Sample-size sensitivity
# ---------------------------------------------------------------------------
def plot_sample_size_sensitivity(results_dir: str, plots_dir: str,
                                 sizes=(10, 100, 500, 1000)) -> None:
    """
    Supplement §2.1, Figure 2: metric estimate as a function of sample size.
    Subsamples the canonical calibration files post-hoc.
    """
    # model -> records (largest canonical alpaca calibration file available)
    by_model: dict[str, list] = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "calibration_*.jsonl"))):
        _, model, ctx = _parse_calibration_name(p)
        if ctx is not None:
            continue  # only the canonical (non-ctx-suffixed) files
        recs = _read_jsonl(p)
        if len(recs) > len(by_model.get(model, [])):
            by_model[model] = recs

    if not by_model:
        print("  (no canonical calibration files — skipping sample-size sensitivity)")
        return

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    drew = False
    for model, recs in sorted(by_model.items()):
        avail = [s for s in sizes if s <= len(recs)]
        if not avail:
            avail = [len(recs)]
        xs, ys = [], []
        for s in avail:
            xs.append(s)
            ys.append(_mean(r["min_entropy"] for r in recs[:s]))
        ax.plot(xs, ys, marker="o", label=model)
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("sample size (number of prompts)")
    ax.set_ylabel("uncertainty floor (mean min entropy, nats)")
    ax.set_title("Sample-size sensitivity")
    ax.legend(title="model", frameon=False)
    fig.tight_layout()
    out = os.path.join(plots_dir, "sensitivity_sample_size.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# #10 · Dataset sensitivity
# ---------------------------------------------------------------------------
def plot_dataset_sensitivity(results_dir: str, plots_dir: str) -> None:
    """Supplement §2.1, Figure 3: metric per dataset per model (grouped bars)."""
    # dataset -> model -> mean_min_entropy (canonical files only)
    data: dict[str, dict[str, float]] = {}
    models: set[str] = set()
    for p in sorted(glob.glob(os.path.join(results_dir, "calibration_*.jsonl"))):
        dataset, model, ctx = _parse_calibration_name(p)
        if ctx is not None:
            continue
        recs = _read_jsonl(p)
        if recs:
            data.setdefault(dataset, {})[model] = _mean(r["min_entropy"] for r in recs)
            models.add(model)

    if len(data) < 2:
        print("  (need >=2 datasets in calibration files — skipping dataset sensitivity)")
        return

    datasets = sorted(data)
    models = sorted(models)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(datasets) + 2), 4.5))
    width = 0.8 / max(len(models), 1)
    x = list(range(len(datasets)))
    for i, m in enumerate(models):
        vals = [data[d].get(m, 0.0) for d in datasets]
        offsets = [xi + (i - (len(models) - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("uncertainty floor (mean min entropy, nats)")
    ax.set_title("Dataset sensitivity")
    ax.legend(title="model", frameon=False)
    fig.tight_layout()
    out = os.path.join(plots_dir, "sensitivity_dataset.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# #4 · Aggregation-invariance check
# ---------------------------------------------------------------------------
def aggregation_invariance(layer_by_model: dict, results_dir: str) -> None:
    """
    For each metric in {entropy, ER, PR} and aggregation in {min, mean, median,
    max}, rank the models, then compare rankings across aggregators with
    Kendall-tau. Paper claim (Supplement §2.3, Figure 5): tau ~ 1.0 within a
    metric. Needs >=3 models to be meaningful.
    """
    if len(layer_by_model) < 3:
        print(f"  (only {len(layer_by_model)} model(s); need >=3 for aggregation "
              f"invariance — skipping)")
        return
    try:
        from scipy.stats import kendalltau
    except ImportError:
        print("  (scipy not installed — skipping aggregation invariance; "
              "pip install scipy)")
        return

    # per-record scalar for each metric
    metric_keys = {"entropy": "min_entropy", "ER": "max_ER", "PR": "max_PR"}
    aggregators = {
        "min": min, "mean": _mean, "median": statistics.median, "max": max,
    }

    models = sorted(layer_by_model)
    rows = []
    for metric, rec_key in metric_keys.items():
        # model -> aggregator -> aggregated value
        agg_vals: dict[str, dict[str, float]] = {}
        for agg_name, agg_fn in aggregators.items():
            for m in models:
                vals = [r[rec_key] for r in layer_by_model[m] if rec_key in r]
                if vals:
                    agg_vals.setdefault(agg_name, {})[m] = agg_fn(vals)

        # rank models under each aggregator (ascending), compare all pairs
        rankings = {}
        for agg_name, mv in agg_vals.items():
            ordered = sorted(mv, key=lambda mm: mv[mm])
            rankings[agg_name] = {m: i for i, m in enumerate(ordered)}

        names = list(rankings)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                na, nb = names[a], names[b]
                common = [m for m in models if m in rankings[na] and m in rankings[nb]]
                if len(common) < 3:
                    continue
                ra = [rankings[na][m] for m in common]
                rb = [rankings[nb][m] for m in common]
                tau, _ = kendalltau(ra, rb)
                rows.append((metric, na, nb, tau))

    if not rows:
        print("  (not enough overlapping data for aggregation invariance)")
        return

    csv_path = os.path.join(results_dir, "aggregation_invariance.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,aggregator_a,aggregator_b,kendall_tau\n")
        for metric, na, nb, tau in rows:
            f.write(f"{metric},{na},{nb},{tau:.4f}\n")
    print(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    results_dir = config.RESULTS_DIR
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    layer_by_model = load_layer_records(results_dir)
    rolling_by_model = load_rolling_records(results_dir)

    if not layer_by_model and not rolling_by_model:
        print(f"No latent/calibration result files in {results_dir}. Run "
              f"run_latent.py, run_latent_rolling.py, or run_calibration.py first.")
        return 1

    print("#2  layerwise ER/PR + hourglass:")
    analyze_layerwise(layer_by_model, results_dir, plots_dir)
    print("#7  prefix-length sensitivity:")
    plot_prefix_sensitivity(rolling_by_model, plots_dir)
    print("#8  context-length sensitivity:")
    plot_context_sensitivity(results_dir, plots_dir)
    print("#9  sample-size sensitivity:")
    plot_sample_size_sensitivity(results_dir, plots_dir)
    print("#10 dataset sensitivity:")
    plot_dataset_sensitivity(results_dir, plots_dir)
    print("#4  aggregation invariance:")
    aggregation_invariance(layer_by_model, results_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
