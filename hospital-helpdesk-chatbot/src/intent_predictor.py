"""Inference wrapper for the fine-tuned DistilBERT intent classifier."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from src.utils import INTENT_MODEL_DIR, get_logger, normalize_text

logger = get_logger(__name__)


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    top_k: list[tuple[str, float]]


class IntentPredictor:
    def __init__(
        self,
        model_dir: str | Path = INTENT_MODEL_DIR,
        device: str | None = None,
        max_length: int = 64,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No trained intent model found in '{model_dir}'. "
                "Train one first with: python -m src.train_intent"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.id2label: dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}
        logger.info("Loaded intent model from %s (%d labels) on %s", model_dir, len(self.id2label), self.device)

    @property
    def labels(self) -> list[str]:
        return [self.id2label[i] for i in sorted(self.id2label)]

    @torch.inference_mode()
    def predict_proba(self, texts: list[str]) -> torch.Tensor:
        texts = [normalize_text(t) for t in texts]
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu()

    def predict_batch(self, texts: list[str], k: int = 3) -> list[IntentPrediction]:
        if not texts:
            return []
        probs = self.predict_proba(texts)
        k = min(k, probs.shape[-1])
        top_p, top_i = probs.topk(k, dim=-1)
        results = []
        for ps, ids in zip(top_p.tolist(), top_i.tolist()):
            top = [(self.id2label[i], round(p, 4)) for p, i in zip(ps, ids)]
            results.append(IntentPrediction(intent=top[0][0], confidence=top[0][1], top_k=top))
        return results

    def predict(self, text: str, k: int = 3) -> IntentPrediction:
        return self.predict_batch([text], k=k)[0]


@lru_cache(maxsize=1)
def get_intent_predictor(model_dir: str = str(INTENT_MODEL_DIR)) -> IntentPredictor:
    return IntentPredictor(model_dir)


if __name__ == "__main__":
    import sys

    predictor = IntentPredictor()
    for query in sys.argv[1:] or ["I want to see a cardiologist tomorrow", "where is my blood report??", "bye"]:
        pred = predictor.predict(query)
        print(f"{pred.intent:26s} {pred.confidence:.3f}  <- {query}")
