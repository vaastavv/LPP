"""TF-IDF + Logistic Regression baseline.

C is selected on validation macro-F1 (never on test). The vectorizer/classifier are refit on train only
with the chosen C, then temperature + thresholds are calibrated on validation + OOD-validation.

Usage: python -m src.intent.baseline
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import load_config, path, set_seed
from src.policy.confidence import calibrate
from src.text import normalize


def make(cfg: dict, C: float):
    b = cfg["baseline"]
    vec = TfidfVectorizer(ngram_range=tuple(b["ngram_range"]), sublinear_tf=b["sublinear_tf"],
                          min_df=b["min_df"], lowercase=True)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=cfg["seed"])
    return vec, clf


def main() -> int:
    cfg = load_config()
    set_seed(cfg["seed"])
    train = pd.read_csv(path(cfg, "train"), keep_default_na=False)
    val = pd.read_csv(path(cfg, "validation"), keep_default_na=False)
    ood_val = pd.read_csv(path(cfg, "ood_validation"), keep_default_na=False)
    Xtr, Xva = train["text"].map(normalize), val["text"].map(normalize)

    grid = []
    for C in cfg["baseline"]["C_grid"]:
        vec, clf = make(cfg, C)
        clf.fit(vec.fit_transform(Xtr), train["intent"])
        f1 = f1_score(val["intent"], clf.predict(vec.transform(Xva)), average="macro")
        grid.append({"C": C, "val_macro_f1": float(f1)})
        print(f"C={C:<6} val macro-F1={f1:.4f}")
    best = max(grid, key=lambda r: (r["val_macro_f1"], -r["C"]))
    vec, clf = make(cfg, best["C"])
    t0 = time.time()
    clf.fit(vec.fit_transform(Xtr), train["intent"])
    train_seconds = time.time() - t0
    labels = list(clf.classes_)

    val_logits = clf.decision_function(vec.transform(Xva))
    ood_logits = clf.decision_function(vec.transform(ood_val["text"].map(normalize)))
    y_val = np.array([labels.index(l) for l in val["intent"]])
    cal, curves = calibrate(val_logits, y_val, ood_logits, labels, cfg)

    out = path(cfg, "baseline_dir")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(out / "logistic_regression.pkl", "wb") as f:
        pickle.dump(clf, f)
    (out / "calibration.json").write_text(json.dumps(cal.to_dict(), indent=2))
    (out / "training_config.json").write_text(json.dumps({
        "model": "TF-IDF + LogisticRegression", "C_grid_validation": grid, "selected_C": best["C"],
        **{k: v for k, v in cfg["baseline"].items() if k != "C_grid"}, "seed": cfg["seed"],
        "train_size": len(train), "validation_size": len(val), "train_seconds": round(train_seconds, 2),
        "dataset_version": cfg["dataset_version"],
    }, indent=2))
    (out / "calibration_curves.json").write_text(json.dumps(curves, indent=2))
    print(f"selected C={best['C']} | T={cal.temperature:.3f} threshold={cal.threshold:.2f} "
          f"emergency_threshold={cal.emergency_threshold:.2f} | ECE {cal.ece_before:.4f} -> {cal.ece_after:.4f}")
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
