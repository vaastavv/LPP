import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.config import GOEMOTIONS_LABELS, INTENTS, load_config, path  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset(cfg):
    return pd.read_csv(path(cfg, "processed_dataset"), keep_default_na=False)


@pytest.fixture(scope="session")
def templates(cfg):
    return json.loads(path(cfg, "response_templates").read_text())


def trained_bert_available(cfg) -> bool:
    d = path(cfg, "intent_model_dir")
    return (d / "model" / "config.json").exists() and (d / "tokenizer").exists()


def emotion_model_available(cfg) -> bool:
    return (path(cfg, "emotion_model_dir") / "config.json").exists()


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory, dataset):
    """A tiny, randomly initialised BERT saved in the same layout as models/intent/ (no network)."""
    from tiny_models import build_tiny_bert
    root = tmp_path_factory.mktemp("tiny_intent")
    build_tiny_bert(dataset["text"].tolist(), root / "hub", INTENTS)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    AutoTokenizer.from_pretrained(root / "hub").save_pretrained(root / "tokenizer")
    AutoModelForSequenceClassification.from_pretrained(root / "hub").save_pretrained(root / "model")
    (root / "label_mapping.json").write_text(json.dumps(
        {"label2id": {l: i for i, l in enumerate(INTENTS)}, "id2label": {str(i): l for i, l in enumerate(INTENTS)}}))
    # Threshold above any 20-way probability a random model produces -> exercises the uncertain path.
    (root / "calibration.json").write_text(json.dumps({"temperature": 1.0, "threshold": 0.9, "emergency_threshold": 0.9}))
    (root / "training_config.json").write_text(json.dumps({"max_length": 32}))
    return root


@pytest.fixture(scope="session")
def tiny_emotion_dir(tmp_path_factory, dataset):
    """Tiny multi-label model with the GoEmotions label set, to test the emotion predictor offline."""
    from tiny_models import build_tokenizer
    from transformers import RobertaConfig, RobertaForSequenceClassification
    root = tmp_path_factory.mktemp("tiny_emotion")
    tok = build_tokenizer(dataset["text"].tolist(), root)
    cfg = RobertaConfig(vocab_size=len(tok) + 2, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=140, pad_token_id=tok.pad_token_id,
                        num_labels=len(GOEMOTIONS_LABELS), problem_type="multi_label_classification",
                        id2label=dict(enumerate(GOEMOTIONS_LABELS)),
                        label2id={l: i for i, l in enumerate(GOEMOTIONS_LABELS)})
    RobertaForSequenceClassification(cfg).save_pretrained(root)
    return root


class StubEmotion:
    """Deterministic stand-in so pipeline tests are independent of the emotion checkpoint."""

    def __init__(self, emotion="neutral", confidence=0.9):
        self.emotion, self.confidence = emotion, confidence

    def predict(self, text):
        return {"emotion": self.emotion, "confidence": self.confidence, "status": "confident",
                "raw_emotion": self.emotion, "top_k": []}
