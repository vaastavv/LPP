"""Intent predictors with a common interface, independent of any UI.

    clf = load_intent_classifier()            # uses models/selected_model.json
    clf.predict("Can I book a cardiology appointment?")
    -> {"intent": "appointment_booking", "confidence": 0.96, "status": "confident", ...}

For low-confidence input `intent` is None and `status` is "uncertain"; `predicted_intent` still carries
the arg-max label for logging/clarification.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.config import load_config, path, resolve
from src.policy.confidence import softmax
from src.text import normalize


class BaseIntentClassifier:
    name = "base"

    def __init__(self, labels: list[str], calibration: dict | None):
        self.labels = labels
        cal = calibration or {}
        self.temperature = float(cal.get("temperature", 1.0))
        self.threshold = float(cal.get("threshold", 0.0))
        self.emergency_threshold = float(cal.get("emergency_threshold", self.threshold))
        self._emergency_idx = labels.index("emergency_assistance") if "emergency_assistance" in labels else None

    def logits(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return softmax(self.logits([normalize(t) for t in texts]), self.temperature)

    def predict_batch(self, texts: list[str], top_k: int = 3) -> list[dict]:
        probs = self.predict_proba(texts)
        out = []
        for p in probs:
            order = np.argsort(-p)
            best = int(order[0])
            conf = float(p[best])
            confident = conf >= self.threshold
            out.append({
                "intent": self.labels[best] if confident else None,
                "confidence": round(conf, 4),
                "status": "confident" if confident else "uncertain",
                "predicted_intent": self.labels[best],
                "emergency_probability": round(float(p[self._emergency_idx]), 4)
                if self._emergency_idx is not None else 0.0,
                "top_k": [{"intent": self.labels[i], "confidence": round(float(p[i]), 4)} for i in order[:top_k]],
            })
        return out

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]


class BertIntentClassifier(BaseIntentClassifier):
    name = "bert"

    def __init__(self, model_dir: str | Path, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        mapping = json.loads((model_dir / "label_mapping.json").read_text())
        labels = [mapping["id2label"][str(i)] for i in range(len(mapping["id2label"]))]
        cal_file = model_dir / "calibration.json"
        super().__init__(labels, json.loads(cal_file.read_text()) if cal_file.exists() else None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model")
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        cfg_file = model_dir / "training_config.json"
        self.max_length = json.loads(cfg_file.read_text()).get("max_length", 64) if cfg_file.exists() else 64
        model_labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        if model_labels != labels:
            raise ValueError("label_mapping.json does not match the model config id2label")
        self._torch = torch

    def logits(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outs = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                outs.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, len(self.labels)))


class BaselineIntentClassifier(BaseIntentClassifier):
    name = "tfidf_logreg"

    def __init__(self, baseline_dir: str | Path):
        baseline_dir = Path(baseline_dir)
        with open(baseline_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(baseline_dir / "logistic_regression.pkl", "rb") as f:
            self.clf = pickle.load(f)
        cal_file = baseline_dir / "calibration.json"
        super().__init__(list(self.clf.classes_), json.loads(cal_file.read_text()) if cal_file.exists() else None)

    def logits(self, texts: list[str]) -> np.ndarray:
        # LR decision_function values are the multinomial logits.
        return self.clf.decision_function(self.vectorizer.transform(texts))


def load_intent_classifier(kind: str | None = None, cfg: dict | None = None) -> BaseIntentClassifier:
    """kind: 'bert', 'tfidf_logreg', or None to use the model recorded in models/selected_model.json."""
    cfg = cfg or load_config()
    if kind is None:
        sel = resolve("models/selected_model.json")
        kind = json.loads(sel.read_text())["selected"] if sel.exists() else "bert"
    if kind == "bert":
        return BertIntentClassifier(path(cfg, "intent_model_dir"))
    if kind == "tfidf_logreg":
        return BaselineIntentClassifier(path(cfg, "baseline_dir"))
    raise ValueError(f"unknown intent model kind: {kind}")
