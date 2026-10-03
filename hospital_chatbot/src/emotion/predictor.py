"""Emotion classifier: pretrained RoBERTa-base fine-tuned on GoEmotions (not trained here).

The label inventory is read from the model's own config.id2label - class ids are never assumed - and
checked against the 28 GoEmotions labels the response templates are keyed by.

The default checkpoint (SamLowe/roberta-base-go_emotions) is multi-label (sigmoid per label). We take the
highest-scoring label; if its score is below `min_confidence` we fall back to "neutral".

    clf = EmotionClassifier.from_config()
    clf.predict("I'm terrified, my father has chest pain")
    -> {"emotion": "fear", "confidence": 0.88, "status": "confident", "top_k": [...]}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.config import GOEMOTIONS_LABELS, load_config, path
from src.text import normalize


class EmotionClassifier:
    def __init__(self, model_name_or_dir: str | Path, min_confidence: float = 0.5, max_length: int = 128,
                 device: str | None = None, strict_labels: bool = True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_dir)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.multi_label = self.model.config.problem_type == "multi_label_classification"
        self.min_confidence = min_confidence
        self.max_length = max_length
        missing = sorted(set(GOEMOTIONS_LABELS) - set(self.labels))
        extra = sorted(set(self.labels) - set(GOEMOTIONS_LABELS))
        if strict_labels and (missing or extra):
            raise ValueError(f"emotion model labels do not match GoEmotions: missing={missing} extra={extra}")

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "EmotionClassifier":
        cfg = cfg or load_config()
        ecfg = cfg["emotion_model"]
        local = path(cfg, "emotion_model_dir")
        src = local if (local / "config.json").exists() else ecfg["model_name"]
        return cls(src, ecfg["min_confidence"], ecfg["max_length"])

    def scores(self, texts: list[str]) -> np.ndarray:
        with self._torch.inference_mode():
            enc = self.tokenizer([normalize(t) for t in texts], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float()
            p = self._torch.sigmoid(logits) if self.multi_label else self._torch.softmax(logits, -1)
        return p.cpu().numpy()

    def predict_batch(self, texts: list[str], top_k: int = 3) -> list[dict]:
        out = []
        for s in self.scores(texts):
            order = np.argsort(-s)
            best = self.labels[int(order[0])]
            conf = float(s[order[0]])
            confident = conf >= self.min_confidence
            out.append({
                "emotion": best if confident else "neutral",
                "confidence": round(conf, 4),
                "status": "confident" if confident else "fallback_neutral",
                "raw_emotion": best,
                "top_k": [{"emotion": self.labels[i], "score": round(float(s[i]), 4)} for i in order[:top_k]],
            })
        return out

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]


def download(cfg: dict | None = None) -> Path:
    """Save the pretrained emotion checkpoint under models/emotion/ for offline inference."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    cfg = cfg or load_config()
    name = cfg["emotion_model"]["model_name"]
    out = path(cfg, "emotion_model_dir")
    out.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(name).save_pretrained(out)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.save_pretrained(out, safe_serialization=True)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"saved {name} -> {out}\nproblem_type={model.config.problem_type} num_labels={len(labels)}")
    print("id2label:", dict(enumerate(labels)))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download()
    else:
        clf = EmotionClassifier.from_config()
        for t in sys.argv[1:] or ["I'm really scared about my surgery tomorrow"]:
            print(t, "->", clf.predict(t))
