"""Fine-tune a pretrained compact BERT-family encoder (default DistilBERT) for 20-way intent classification.

    raw text -> tokenizer -> encoder -> [CLS] -> dropout -> linear -> 20 logits -> softmax

Model selection uses validation macro-F1 with early stopping; the test set is not read here.
After training, temperature + confidence/emergency thresholds are calibrated on validation + OOD-validation.

Usage: python -m src.intent.train [--config configs/training.yaml] [--epochs N] [--model-name NAME]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import INTENTS, load_config, path, set_seed
from src.intent.dataset import IntentDataset, collate_fn, label_maps
from src.policy.confidence import calibrate
from src.text import normalize


@torch.inference_mode()
def predict_logits(model, tokenizer, texts: list[str], max_length: int, device: str, bs: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(texts), bs):
        enc = tokenizer([normalize(t) for t in texts[i:i + bs]], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        outs.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(outs)


def class_weights(train: pd.DataFrame, label2id: dict, icfg: dict) -> tuple[torch.Tensor | None, dict]:
    counts = Counter(train["intent"])
    ratio = max(counts.values()) / min(counts.values())
    mode = icfg.get("class_weighting", "auto")
    enabled = mode is True or (mode == "auto" and ratio > icfg.get("imbalance_ratio_threshold", 1.5))
    info = {"max_min_ratio": round(ratio, 3), "mode": mode, "enabled": bool(enabled)}
    if not enabled:
        return None, info
    n = sum(counts.values())
    w = torch.tensor([n / (len(counts) * counts[l]) for l in label2id], dtype=torch.float)
    return w, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--model-name")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    icfg = dict(cfg["intent_model"])
    if args.epochs:
        icfg["epochs"] = args.epochs
    if args.model_name:
        icfg["model_name"] = args.model_name
    seed = cfg["seed"]
    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = pd.read_csv(path(cfg, "train"), keep_default_na=False)
    val = pd.read_csv(path(cfg, "validation"), keep_default_na=False)
    ood_val = pd.read_csv(path(cfg, "ood_validation"), keep_default_na=False)
    label2id, id2label = label_maps(INTENTS)

    tokenizer = AutoTokenizer.from_pretrained(icfg["model_name"])
    mcfg = AutoConfig.from_pretrained(icfg["model_name"], num_labels=len(INTENTS),
                                      id2label=id2label, label2id=label2id)
    # Classifier-head dropout (attribute name differs across BERT-family configs).
    for attr in ("seq_classif_dropout", "classifier_dropout"):
        if hasattr(mcfg, attr):
            setattr(mcfg, attr, icfg["dropout"])
    model = AutoModelForSequenceClassification.from_pretrained(icfg["model_name"], config=mcfg).to(device)

    train_ds = IntentDataset(train, tokenizer, label2id, icfg["max_length"])
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=icfg["batch_size"], shuffle=True, generator=g,
                        collate_fn=collate_fn(tokenizer))
    weights, cw_info = class_weights(train, label2id, icfg)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": icfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=icfg["learning_rate"])
    total = len(loader) * icfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(total * icfg["warmup_ratio"]), total)

    y_val = np.array([label2id[l] for l in val["intent"]])
    history, best_state, best_f1, bad_epochs = [], None, -1.0, 0
    t0 = time.time()
    for epoch in range(1, icfg["epochs"] + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        val_logits = predict_logits(model, tokenizer, val["text"].tolist(), icfg["max_length"], device)
        pred = val_logits.argmax(1)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_accuracy": float(accuracy_score(y_val, pred)),
               "val_macro_f1": float(f1_score(y_val, pred, average="macro")),
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))
        if rec["val_macro_f1"] > best_f1 + 1e-6:
            best_f1, bad_epochs = rec["val_macro_f1"], 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= icfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)

    val_logits = predict_logits(model, tokenizer, val["text"].tolist(), icfg["max_length"], device)
    ood_logits = predict_logits(model, tokenizer, ood_val["text"].tolist(), icfg["max_length"], device)
    cal, curves = calibrate(val_logits, y_val, ood_logits, INTENTS, cfg)

    out = path(cfg, "intent_model_dir")
    (out / "model").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "tokenizer")
    (out / "label_mapping.json").write_text(json.dumps(
        {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, indent=2))
    (out / "calibration.json").write_text(json.dumps(cal.to_dict(), indent=2))
    (out / "calibration_curves.json").write_text(json.dumps(curves, indent=2))
    (out / "training_config.json").write_text(json.dumps({
        **icfg, "seed": seed, "device": device, "dataset_version": cfg["dataset_version"],
        "train_size": len(train), "validation_size": len(val), "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1, "class_weighting": cw_info, "history": history,
        "train_seconds": round(time.time() - t0, 1), "torch": torch.__version__,
    }, indent=2))
    print(f"best epoch {best_epoch} val macro-F1 {best_f1:.4f} | T={cal.temperature:.3f} "
          f"threshold={cal.threshold:.2f} emergency_threshold={cal.emergency_threshold:.2f} "
          f"ECE {cal.ece_before:.4f} -> {cal.ece_after:.4f}")
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
