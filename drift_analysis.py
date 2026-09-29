#!/usr/bin/env python3
"""
Analysis + plots for the drift-monitoring experiment (proposed LPP extension).

Reads results/drift_<model>_<kind>.jsonl (written by run_drift.py) and, per
(model, kind):
  - computes the detection level of each metric (first drift level whose paired
    bootstrap CI separates from the in-distribution baseline), and the "lead" of
    each latent metric over the accuracy drop;
  - writes results/drift_summary.csv;
  - plots results/plots/drift_<model>_<kind>.png: accuracy decline (left axis)
    overlaid with baseline-normalized entropy / ER / PR rise (right axis), with
    detection severities marked, so a latent metric that moves before accuracy
    is visible at a glance.

Pure post-processing; safe to re-run. Run from the project root after run_drift.

    python drift_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import config
from src.drift import aligned_by_level, detection_level


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read_jsonl(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


LATENT = [("min_entropy", "entropy floor"), ("max_ER", "max ER"), ("max_PR", "max PR")]


def analyze_one(records: list) -> dict:
    """Detection levels, severities and leads for one (model, kind) group."""
    severities = {r["level"]: r["severity"] for r in records}

    correct_rows = [r for r in records if r.get("correct_at_baseline")]
    _, acc = aligned_by_level(correct_rows, "is_correct")
    latent_series = {key: aligned_by_level(records, key)[1] for key, _ in LATENT}

    acc_lvl = detection_level(acc, "down") if acc else None
    det = {key: (detection_level(series, "up") if series else None)
           for key, series in latent_series.items()}

    def sev(lvl):
        return severities.get(lvl) if lvl is not None else None

    result = {
        "severities": severities, "acc": acc, "latent_series": latent_series,
        "acc_detect_level": acc_lvl, "acc_detect_severity": sev(acc_lvl),
        "detect": det,
        "detect_severity": {k: sev(v) for k, v in det.items()},
        "lead_levels": {k: (acc_lvl - v if (acc_lvl is not None and v is not None) else None)
                        for k, v in det.items()},
    }
    return result


def plot_one(model: str, kind: str, res: dict, plots_dir: str) -> None:
    plt = _get_plt()
    severities = res["severities"]
    levels = sorted(severities)
    xs = [severities[l] for l in levels]

    fig, ax_acc = plt.subplots(figsize=(8, 5))
    ax_lat = ax_acc.twinx()

    # accuracy on the left axis
    acc = res["acc"]
    if acc:
        acc_means = [_mean(acc.get(l, [])) for l in levels]
        ax_acc.plot(xs, acc_means, color="black", marker="o", linewidth=2, label="accuracy")
    ax_acc.set_ylabel("task accuracy (samples correct at baseline)")
    ax_acc.set_xlabel(f"drift severity ({kind})")
    ax_acc.set_ylim(0, 1)

    # baseline-normalized latent metrics on the right axis (z vs level-0 spread)
    colors = {"min_entropy": "tab:red", "max_ER": "tab:blue", "max_PR": "tab:green"}
    for key, label in LATENT:
        series = res["latent_series"][key]
        if not series or 0 not in series:
            continue
        base = series[0]
        mu0 = statistics.fmean(base)
        sd0 = statistics.pstdev(base) or 1.0
        zs = [(_mean(series.get(l, [])) - mu0) / sd0 for l in levels]
        ax_lat.plot(xs, zs, color=colors[key], marker="s", linestyle="--", label=f"{label} (z)")
        dsev = res["detect_severity"][key]
        if dsev is not None:
            ax_lat.axvline(dsev, color=colors[key], alpha=0.35, linewidth=1)
    ax_lat.set_ylabel("latent metric shift vs baseline (z-score)")
    ax_lat.axhline(0, color="grey", linewidth=0.6)

    # accuracy detection marker
    if res["acc_detect_severity"] is not None:
        ax_acc.axvline(res["acc_detect_severity"], color="black", alpha=0.5,
                       linewidth=1.5, linestyle=":")

    lead = res["lead_levels"].get("min_entropy")
    lead_txt = (f"entropy lead = {lead} levels" if lead is not None else "entropy lead = n/a")
    demo = "  [SYNTHETIC DEMO]" if model == "SYNTHDEMO" else ""
    ax_acc.set_title(f"Drift early-warning: {model} / {kind}{demo}\n{lead_txt}")

    # only the metric curves carry real labels; axvline markers do not.
    lines = [ln for ln in ax_acc.get_lines() + ax_lat.get_lines()
             if not ln.get_label().startswith("_")]
    ax_acc.legend(lines, [ln.get_label() for ln in lines], loc="center left", frameon=False)
    fig.tight_layout()
    out = os.path.join(plots_dir, f"drift_{model}_{kind}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    results_dir = config.RESULTS_DIR
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(results_dir, "drift_*.jsonl")))
    if not paths:
        print(f"No drift_*.jsonl in {results_dir}. Run run_drift.py first.")
        return 1

    rows_out = []
    for p in paths:
        recs = _read_jsonl(p)
        if not recs:
            continue
        model = recs[0]["model"]
        kind = recs[0]["kind"]
        res = analyze_one(recs)
        plot_one(model, kind, res, plots_dir)

        row = {
            "model": model, "kind": kind,
            "acc_detect_severity": res["acc_detect_severity"],
            "entropy_detect_severity": res["detect_severity"]["min_entropy"],
            "ER_detect_severity": res["detect_severity"]["max_ER"],
            "PR_detect_severity": res["detect_severity"]["max_PR"],
            "lead_entropy_levels": res["lead_levels"]["min_entropy"],
            "lead_ER_levels": res["lead_levels"]["max_ER"],
            "lead_PR_levels": res["lead_levels"]["max_PR"],
        }
        rows_out.append(row)
        print(f"  {model}/{kind}: entropy@{row['entropy_detect_severity']} "
              f"ER@{row['ER_detect_severity']} PR@{row['PR_detect_severity']} "
              f"acc@{row['acc_detect_severity']} "
              f"(entropy lead {row['lead_entropy_levels']} levels)")

    csv_path = os.path.join(results_dir, "drift_summary.csv")
    cols = ["model", "kind", "acc_detect_severity", "entropy_detect_severity",
            "ER_detect_severity", "PR_detect_severity", "lead_entropy_levels",
            "lead_ER_levels", "lead_PR_levels"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows_out:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
