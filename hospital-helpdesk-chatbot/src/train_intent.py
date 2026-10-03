"""Fine-tune ``distilbert-base-uncased`` for multi-class intent classification.

Pipeline
--------
* Load stratified splits (``data/train.csv``, ``val.csv``, ``test.csv``).
* Tokenise (truncation, attention masks) - padding is *dynamic* per batch via
  ``DataCollatorWithPadding``.
* Train with AdamW + linear warmup/decay LR schedule, weight decay, epoch-level
  evaluation and checkpointing, early stopping on validation macro-F1, and
  automatic reload of the best checkpoint.
* Evaluate on the held-out test set: accuracy, precision, recall, F1 (macro and
  weighted), classification report and confusion matrix.
* Save the best model + tokenizer + label maps to ``models/intent_classifier/``
  and metrics/plots to ``reports/``.

Usage::

    python -m src.train_intent                       # sensible defaults
    python -m src.train_intent --epochs 15 --lr 3e-5 --batch-size 32
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    INTENT_MODEL_DIR,
    INTENTS,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    get_logger,
    label_maps,
    save_json,
    set_seed,
)

logger = get_logger(__name__)

BASE_MODEL = "distilbert-base-uncased"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class MetricComputer:
    """Accuracy / precision / recall / F1 via Hugging Face ``evaluate``.

    Falls back to scikit-learn if the metric scripts cannot be fetched (e.g. an
    offline machine) - the numbers are identical.
    """

    def __init__(self) -> None:
        self._metrics = None
        try:
            import evaluate

            self._metrics = {
                "accuracy": evaluate.load("accuracy"),
                "precision": evaluate.load("precision"),
                "recall": evaluate.load("recall"),
                "f1": evaluate.load("f1"),
            }
        except Exception as exc:  # pragma: no cover - depends on network
            logger.warning("Could not load `evaluate` metrics (%s); using scikit-learn", exc)

    def __call__(self, predictions: np.ndarray, references: np.ndarray) -> dict[str, float]:
        if self._metrics:
            m = self._metrics
            out = {"accuracy": m["accuracy"].compute(predictions=predictions, references=references)["accuracy"]}
            for avg in ("macro", "weighted"):
                for name in ("precision", "recall", "f1"):
                    kwargs = {"average": avg}
                    if name != "f1":
                        kwargs["zero_division"] = 0
                    out[f"{name}_{avg}"] = m[name].compute(
                        predictions=predictions, references=references, **kwargs
                    )[name]
            return {k: float(v) for k, v in out.items()}

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        out = {"accuracy": accuracy_score(references, predictions)}
        for avg in ("macro", "weighted"):
            p, r, f, _ = precision_recall_fscore_support(references, predictions, average=avg, zero_division=0)
            out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
        return {k: float(v) for k, v in out.items()}


def save_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_dir: Path,
    prefix: str = "test",
    texts: list[str] | None = None,
) -> dict:
    """Write classification report, confusion matrix (CSV + PNG) and errors."""
    from sklearn.metrics import classification_report, confusion_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(labels)))

    report_txt = classification_report(y_true, y_pred, labels=label_ids, target_names=labels, digits=4, zero_division=0)
    report_dict = classification_report(
        y_true, y_pred, labels=label_ids, target_names=labels, output_dict=True, zero_division=0
    )
    (out_dir / f"{prefix}_classification_report.txt").write_text(report_txt, encoding="utf-8")
    save_json(report_dict, out_dir / f"{prefix}_classification_report.json")
    logger.info("Classification report (%s):\n%s", prefix, report_txt)

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"{prefix}_confusion_matrix.csv")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(11, 9))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=60, colorbar=False)
        ax.set_title(f"Intent classifier - {prefix} confusion matrix")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=130)
        plt.close(fig)
    except ImportError:
        logger.info("matplotlib not installed; skipping confusion matrix plot")

    if texts is not None:
        errors = [
            {"text": t, "true": labels[a], "pred": labels[b]}
            for t, a, b in zip(texts, y_true, y_pred)
            if a != b
        ]
        pd.DataFrame(errors, columns=["text", "true", "pred"]).to_csv(out_dir / f"{prefix}_errors.csv", index=False)
    return report_dict


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_splits(label2id: dict[str, int]):
    from datasets import Dataset, DatasetDict

    splits = {}
    for name, path in (("train", TRAIN_CSV), ("validation", VAL_CSV), ("test", TEST_CSV)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Run: python -m scripts.validate_dataset")
        df = pd.read_csv(path).dropna(subset=["text", "intent"])
        unknown = set(df["intent"]) - set(label2id)
        if unknown:
            raise ValueError(f"{path.name} contains unknown intents: {unknown}")
        df["label"] = df["intent"].map(label2id).astype(int)
        splits[name] = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        logger.info("Loaded %-10s %5d rows", name, len(df))
    return DatasetDict(splits)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default=BASE_MODEL)
    p.add_argument("--output-dir", default=str(INTENT_MODEL_DIR))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--scheduler", default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial"])
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    p.add_argument("--metric", default="f1_macro", help="Validation metric used to pick the best model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Mixed precision (CUDA only)")
    p.add_argument("--keep-checkpoints", action="store_true", help="Keep intermediate checkpoints after training")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    args = parse_args(argv)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_dir = MODELS_DIR / "checkpoints" / "intent_classifier"

    label2id, id2label = label_maps(INTENTS)
    raw = load_splits(label2id)

    # ---- tokenisation (attention masks; padding left to the collator) ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)  # dynamic padding

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(INTENTS), id2label=id2label, label2id=label2id
    )

    metric_computer = MetricComputer()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric_computer(preds, labels)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric,
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    logger.info("Starting training: %s", vars(args))
    train_result = trainer.train()
    logger.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)

    # ---- save best model ----
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_json({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, output_dir / "label_map.json")
    save_json(vars(args), output_dir / "training_args.json")

    # ---- evaluation ----
    val_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="val")
    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = test_output.label_ids

    report = save_evaluation_artifacts(
        y_true, y_pred, INTENTS, REPORTS_DIR, prefix="test", texts=list(raw["test"]["text"])
    )

    metrics = {
        "base_model": args.model_name,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "epochs_trained": round(float(trainer.state.epoch or 0), 2),
        "train": {k: float(v) for k, v in train_result.metrics.items()},
        "validation": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_output.metrics.items()},
        "test_per_class_f1": {label: round(report[label]["f1-score"], 4) for label in INTENTS},
        "log_history": trainer.state.log_history,
    }
    save_json(metrics, REPORTS_DIR / "intent_metrics.json")
    save_json({k: v for k, v in metrics.items() if k != "log_history"}, output_dir / "metrics.json")

    logger.info(
        "TEST accuracy=%.4f  macro-F1=%.4f  weighted-F1=%.4f",
        metrics["test"]["test_accuracy"], metrics["test"]["test_f1_macro"], metrics["test"]["test_f1_weighted"],
    )
    logger.info("Model saved to %s ; reports in %s", output_dir, REPORTS_DIR)

    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return metrics


if __name__ == "__main__":
    main()
