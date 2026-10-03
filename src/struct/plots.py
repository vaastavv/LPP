"""
Figures for the CodeLLM structural-profiling paper.

Fig 3 -- layer-wise structural representation profiles across normalized depth.
Fig 4 -- relationships between latent structural metrics and downstream SE
         performance (built once downstream scores exist).
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
           "#be185d", "#4b5563"]


def figure3(results, out_path: str) -> str:
    """Three panels (SRS, CFS, DFBS) vs normalized depth, one line per model."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [("SRS", "SRS", lambda r: r.SRS),
              ("CFS", "CFS", lambda r: r.CFS),
              ("DFBS", "DFBS", lambda r: r.DFBS)]
    for ax, (title, ylab, getter) in zip(axes, panels):
        for i, res in enumerate(results):
            arr = getter(res)
            if arr is None or len(arr) < 2:
                continue
            x = np.arange(len(arr)) / max(res.L, 1)
            ax.plot(x, arr, marker="o", ms=3, lw=1.5,
                    color=_COLORS[i % len(_COLORS)], label=res.model)
        ax.set_title(title)
        ax.set_xlabel(r"normalized depth $\lambda = l/L$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Layer-wise structural representation profiles", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure4(metric_by_model: dict, perf_by_model: dict, tasks: list[str],
            out_path: str) -> str:
    """
    Scatter of peak latent metric vs downstream performance.
    metric_by_model: {model: {"SRS":v,"CFS":v,"DFBS":v}}
    perf_by_model:   {model: {task: score}}
    One panel per (metric) with the most associated task highlighted.
    """
    models = sorted(metric_by_model)
    metrics = ["SRS", "CFS", "DFBS"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, met in zip(axes, metrics):
        xs = [metric_by_model[m][met] for m in models]
        for j, task in enumerate(tasks):
            ys = [perf_by_model[m].get(task, np.nan) for m in models]
            ax.scatter(xs, ys, s=28, color=_COLORS[j % len(_COLORS)], label=task)
        ax.set_xlabel(f"peak {met}")
        ax.set_ylabel("task performance")
        ax.set_title(met)
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Latent structural metrics vs downstream SE performance", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
