"""Evaluate candidate intent models, select one on VALIDATION, then report on the held-out TEST set.

Order of operations (enforced in code):
  1. compute validation metrics for every available candidate
  2. select the model from validation results and write models/selected_model.json
  3. only then compute test metrics, confusion matrix, error analysis, calibration and OOD results

Selection rule: highest validation macro-F1; if candidates are within `selection_tolerance` macro-F1 of
each other, prefer the one with lower single-query latency.

Usage: python -m src.intent.evaluate [--models tfidf_logreg bert]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)

from src.config import INTENTS, load_config, path, resolve  # noqa: E402
from src.intent.predictor import load_intent_classifier  # noqa: E402
from src.policy.confidence import expected_calibration_error, softmax  # noqa: E402

SELECTION_TOLERANCE = 0.005


def core_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p, "macro_recall": r,
            "macro_f1": f, "weighted_f1": wf}


def dir_size_mb(p: Path) -> float:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return round(total / 1e6, 2)


def latency_ms(clf, texts: list[str], n: int = 100) -> dict:
    clf.predict(texts[0])  # warm-up
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        clf.predict(texts[i % len(texts)])
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(float(np.mean(times)), 3), "p95_ms": round(float(np.percentile(times, 95)), 3)}


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, label="row-normalised")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_reliability(bins_by_name: dict[str, list[dict]], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    for name, bins in bins_by_name.items():
        xs = [b["confidence"] for b in bins if b["count"]]
        ys = [b["accuracy"] for b in bins if b["count"]]
        ax.plot(xs, ys, "o-", label=name)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_threshold_curve(curve: list[dict], chosen: float, out: Path, title: str) -> None:
    t = [r["threshold"] for r in curve]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(t, [r["in_domain_coverage"] for r in curve], label="in-domain coverage")
    ax.plot(t, [r["selective_accuracy"] for r in curve], label="selective accuracy (accepted)")
    ax.plot(t, [r["ood_false_accept_rate"] for r in curve], label="OOD false-accept rate")
    ax.plot(t, [r["balanced_accuracy"] for r in curve], label="balanced accept/reject acc.")
    ax.axvline(chosen, color="k", ls=":", label=f"chosen = {chosen:.2f}")
    ax.set_xlabel("confidence threshold (validation)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tfidf_logreg", "bert"])
    args = ap.parse_args(argv)
    cfg = load_config()
    ev = path(cfg, "evaluation_dir")
    ev.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(path(cfg, "validation"), keep_default_na=False)
    test = pd.read_csv(path(cfg, "test"), keep_default_na=False)
    ood_val = pd.read_csv(path(cfg, "ood_validation"), keep_default_na=False)
    ood_test = pd.read_csv(path(cfg, "ood_test"), keep_default_na=False)
    train_size = len(pd.read_csv(path(cfg, "train"), keep_default_na=False))

    # ---------- 1. validation for each candidate ----------
    cands = {}
    for kind in args.models:
        try:
            clf = load_intent_classifier(kind, cfg)
        except (FileNotFoundError, OSError) as e:
            print(f"[skip] {kind}: not trained ({e.__class__.__name__})")
            continue
        vp = clf.predict_proba(val["text"].tolist())
        cands[kind] = {
            "clf": clf,
            "validation": core_metrics(val["intent"].tolist(), [clf.labels[i] for i in vp.argmax(1)]),
            "latency": latency_ms(clf, val["text"].tolist()),
            "size_mb": dir_size_mb(path(cfg, "intent_model_dir" if kind == "bert" else "baseline_dir")),
        }
        print(f"[validation] {kind}: macro-F1={cands[kind]['validation']['macro_f1']:.4f} "
              f"latency={cands[kind]['latency']['mean_ms']}ms size={cands[kind]['size_mb']}MB")
    if not cands:
        print("no trained models found")
        return 1

    # ---------- 2. selection on validation only ----------
    best_f1 = max(c["validation"]["macro_f1"] for c in cands.values())
    near = [k for k, c in cands.items() if best_f1 - c["validation"]["macro_f1"] <= SELECTION_TOLERANCE]
    selected = min(near, key=lambda k: cands[k]["latency"]["mean_ms"])
    selection = {
        "selected": selected,
        "rule": f"highest validation macro-F1; within {SELECTION_TOLERANCE} macro-F1, prefer lower latency",
        "validation_macro_f1": {k: round(c["validation"]["macro_f1"], 4) for k, c in cands.items()},
        "decided_before_test_evaluation": True,
    }
    resolve("models/selected_model.json").write_text(json.dumps(selection, indent=2))
    print(f"[selection] {selected}")

    # ---------- 3. test evaluation ----------
    results = {}
    for kind, c in cands.items():
        clf = c["clf"]
        sub = ev / kind
        sub.mkdir(exist_ok=True)
        logits = clf.logits(test["text"].tolist())
        probs = softmax(logits, clf.temperature)
        pred = [clf.labels[i] for i in probs.argmax(1)]
        conf = probs.max(1)
        y = np.array([clf.labels.index(l) for l in test["intent"]])
        m = core_metrics(test["intent"].tolist(), pred)
        report = classification_report(test["intent"], pred, labels=INTENTS, output_dict=True, zero_division=0)
        cm = confusion_matrix(test["intent"], pred, labels=INTENTS)
        plot_confusion(cm, INTENTS, f"{kind} - test confusion matrix", sub / "confusion_matrix.png")
        (sub / "classification_report.json").write_text(json.dumps(report, indent=2))

        pairs = Counter((t, p) for t, p in zip(test["intent"], pred) if t != p)
        confused = [{"true": t, "predicted": p, "count": n} for (t, p), n in pairs.most_common(10)]
        # symmetric view: both directions summed
        sym = Counter()
        for (t, p), n in pairs.items():
            sym[tuple(sorted((t, p)))] += n
        confused_sym = [{"pair": list(k), "count": n} for k, n in sym.most_common(10)]

        accepted = conf >= clf.threshold
        err = test.assign(predicted=pred, confidence=conf.round(4),
                          status=np.where(accepted, "confident", "uncertain"))
        err = err[err["intent"] != err["predicted"]].rename(columns={"intent": "true_intent", "text": "query"})
        err = err[["true_intent", "predicted", "confidence", "status", "query"]].sort_values(
            "confidence", ascending=False)
        err.to_csv(sub / "error_analysis.csv", index=False)

        ece_raw, bins_raw = expected_calibration_error(softmax(logits), y, cfg["confidence"]["ece_bins"])
        ece_cal, bins_cal = expected_calibration_error(probs, y, cfg["confidence"]["ece_bins"])
        plot_reliability({"uncalibrated (T=1)": bins_raw, f"temperature-scaled (T={clf.temperature:.2f})": bins_cal},
                         sub / "reliability_diagram.png", f"{kind} - test reliability")
        curves = json.loads((path(cfg, "intent_model_dir" if kind == "bert" else "baseline_dir")
                             / "calibration_curves.json").read_text())
        plot_threshold_curve(curves["curve"], clf.threshold, sub / "threshold_curve.png",
                             f"{kind} - threshold selection (validation + OOD-val)")

        ood_conf = clf.predict_proba(ood_test["text"].tolist()).max(1)
        p_em = probs[:, clf.labels.index("emergency_assistance")]
        is_em = test["intent"].to_numpy() == "emergency_assistance"
        ood_pe = clf.predict_proba(ood_test["text"].tolist())[:, clf.labels.index("emergency_assistance")]
        selective = {
            "threshold": clf.threshold,
            "in_domain_coverage": float(accepted.mean()),
            "selective_accuracy": float((np.array(pred) == test["intent"].to_numpy())[accepted].mean())
            if accepted.any() else None,
            "confident_errors": int(((np.array(pred) != test["intent"].to_numpy()) & accepted).sum()),
            "errors_total": int((np.array(pred) != test["intent"].to_numpy()).sum()),
            "ood_test_size": len(ood_test),
            "ood_false_accept_rate": float((ood_conf >= clf.threshold).mean()),
            "ood_false_accepts": ood_test["text"][ood_conf >= clf.threshold].tolist(),
        }
        emergency = {
            "threshold": clf.emergency_threshold,
            "recall_on_test": float((p_em[is_em] >= clf.emergency_threshold).mean()),
            "false_emergency_rate_in_domain": float((p_em[~is_em] >= clf.emergency_threshold).mean()),
            "false_emergency_rate_ood": float((ood_pe >= clf.emergency_threshold).mean()),
            "argmax_recall_on_test": report["emergency_assistance"]["recall"],
        }
        lowest_recall = sorted(((k, v["recall"]) for k, v in report.items() if k in INTENTS), key=lambda x: x[1])[:5]
        results[kind] = {
            "model": kind,
            "validation": c["validation"],
            "test": m,
            "lowest_recall_intents": [{"intent": k, "recall": r} for k, r in lowest_recall],
            "most_confused_pairs_directed": confused,
            "most_confused_pairs_symmetric": confused_sym,
            "calibration_test": {"temperature": clf.temperature, "ece_uncalibrated": ece_raw, "ece_calibrated": ece_cal},
            "selective_prediction_test": selective,
            "emergency_routing_test": emergency,
            "latency_single_query": c["latency"],
            "model_size_mb": c["size_mb"],
        }
        print(f"[test] {kind}: acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
              f"weighted-F1={m['weighted_f1']:.4f} ECE {ece_raw:.3f}->{ece_cal:.3f} "
              f"coverage={selective['in_domain_coverage']:.3f} OOD-FA={selective['ood_false_accept_rate']:.3f}")

    # ---------- artefacts for the selected model at the top level ----------
    import shutil
    for f in ("classification_report.json", "confusion_matrix.png", "error_analysis.csv",
              "reliability_diagram.png", "threshold_curve.png"):
        shutil.copy(ev / selected / f, ev / f)
    comparison = {
        "note": "All numbers computed by src/intent/evaluate.py on the same held-out test split.",
        "selection": selection,
        "models": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    }
    (ev / "baseline_results.json").write_text(json.dumps(comparison, indent=2, default=float))

    # ---------- experiment log ----------
    icfg_file = path(cfg, "intent_model_dir") / "training_config.json"
    bcfg_file = path(cfg, "baseline_dir") / "training_config.json"
    exp = []
    for kind, r in results.items():
        tc = json.loads((icfg_file if kind == "bert" else bcfg_file).read_text())
        exp.append({
            "model": tc.get("model_name", tc.get("model")), "kind": kind,
            "dataset_version": cfg["dataset_version"], "train_size": train_size,
            "validation_size": len(val), "test_size": len(test),
            "learning_rate": tc.get("learning_rate"), "batch_size": tc.get("batch_size"),
            "epochs": tc.get("epochs"), "best_epoch": tc.get("best_epoch"), "C": tc.get("selected_C"),
            **{k: round(v, 4) for k, v in r["test"].items()},
            "selected": kind == selected,
        })
    (ev / "experiments.json").write_text(json.dumps(exp, indent=2))
    lines = ["| model | dataset | train/val/test | lr | batch | epochs (best) | accuracy | macro P | macro R | macro F1 | weighted F1 | selected |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in exp:
        ep = f"{e['epochs']} ({e['best_epoch']})" if e["epochs"] else "-"
        lines.append(f"| {e['model']} | {e['dataset_version']} | {e['train_size']}/{e['validation_size']}/{e['test_size']} "
                     f"| {e['learning_rate'] or '-'} | {e['batch_size'] or '-'} | {ep} | {e['accuracy']:.4f} | "
                     f"{e['macro_precision']:.4f} | {e['macro_recall']:.4f} | {e['macro_f1']:.4f} | "
                     f"{e['weighted_f1']:.4f} | {'yes' if e['selected'] else ''} |")
    (ev / "experiments.md").write_text("\n".join(lines) + "\n")
    print((ev / "experiments.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
