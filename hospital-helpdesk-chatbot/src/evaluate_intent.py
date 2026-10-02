"""Evaluate a trained intent classifier on any labelled CSV (default: test split).

Usage::

    python -m src.evaluate_intent
    python -m src.evaluate_intent --data data/val.csv --prefix val
    python -m src.evaluate_intent --data my_real_queries.csv --model-dir models/intent_classifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.intent_predictor import IntentPredictor
from src.train_intent import MetricComputer, save_evaluation_artifacts
from src.utils import INTENT_MODEL_DIR, REPORTS_DIR, TEST_CSV, get_logger, save_json

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(TEST_CSV))
    p.add_argument("--model-dir", default=str(INTENT_MODEL_DIR))
    p.add_argument("--out-dir", default=str(REPORTS_DIR))
    p.add_argument("--prefix", default="eval")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    predictor = IntentPredictor(args.model_dir)
    labels = predictor.labels
    label2id = {label: i for i, label in enumerate(labels)}

    df = pd.read_csv(args.data).dropna(subset=["text", "intent"])
    unknown = set(df["intent"]) - set(label2id)
    if unknown:
        raise ValueError(f"Evaluation data contains intents unknown to the model: {unknown}")

    texts = df["text"].tolist()
    preds, confs = [], []
    for i in range(0, len(texts), args.batch_size):
        for pred in predictor.predict_batch(texts[i : i + args.batch_size], k=1):
            preds.append(label2id[pred.intent])
            confs.append(pred.confidence)

    y_true = df["intent"].map(label2id).to_numpy()
    y_pred = np.array(preds)
    metrics = MetricComputer()(y_pred, y_true)
    metrics["mean_confidence"] = float(np.mean(confs))
    metrics["mean_confidence_errors"] = float(np.mean([c for c, a, b in zip(confs, y_true, y_pred) if a != b] or [0.0]))
    metrics["n_samples"] = len(df)

    out_dir = Path(args.out_dir)
    save_evaluation_artifacts(y_true, y_pred, labels, out_dir, prefix=args.prefix, texts=texts)
    save_json(metrics, out_dir / f"{args.prefix}_metrics.json")
    logger.info("Metrics: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return metrics


if __name__ == "__main__":
    main()
