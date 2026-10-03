import numpy as np
import pytest

from conftest import trained_bert_available
from src.config import INTENTS
from src.intent.predictor import BaselineIntentClassifier, BertIntentClassifier
from src.policy.confidence import (expected_calibration_error, fit_temperature, select_threshold, softmax,
                                   threshold_curve)


# ---- BERT predictor interface (offline, tiny random checkpoint) ----
def test_tokenizer_and_model_load(tiny_bert_dir):
    clf = BertIntentClassifier(tiny_bert_dir)
    assert clf.labels == INTENTS
    assert clf.tokenizer("hello")["input_ids"]


def test_prediction_returns_confidence(tiny_bert_dir):
    r = BertIntentClassifier(tiny_bert_dir).predict("Can I book a cardiology appointment?")
    assert set(r) >= {"intent", "confidence", "status", "predicted_intent", "top_k", "emergency_probability"}
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["predicted_intent"] in INTENTS


def test_unknown_fallback(tiny_bert_dir):
    r = BertIntentClassifier(tiny_bert_dir).predict("Tell me a joke")
    assert r["status"] == "uncertain" and r["intent"] is None


# ---- baseline (trained artefact committed in models/baseline) ----
def test_baseline_predicts(cfg):
    from src.config import path
    clf = BaselineIntentClassifier(path(cfg, "baseline_dir"))
    r = clf.predict("I want to cancel my appointment")
    assert r["predicted_intent"] == "appointment_cancellation"
    assert 0 < clf.threshold < 1


# ---- calibration utilities ----
def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    logits = rng.normal(size=(500, 5))
    logits[np.arange(500), y] += 1.0
    logits *= 6  # overconfident
    T = fit_temperature(logits, y)
    assert T > 1
    assert expected_calibration_error(softmax(logits, T), y)[0] < expected_calibration_error(softmax(logits), y)[0]


def test_threshold_selection_respects_coverage():
    conf_in = np.linspace(0.3, 1.0, 100)
    correct = conf_in > 0.5
    conf_ood = np.linspace(0.1, 0.6, 50)
    sel = select_threshold(threshold_curve(conf_in, correct, conf_ood), min_coverage=0.6)
    assert sel["in_domain_coverage"] >= 0.6 and 0.3 < sel["threshold"] < 0.7


# ---- trained model checks (run only when models/intent has been trained) ----
needs_bert = pytest.mark.skipif("not trained_bert_available(__import__('src.config').config.load_config())",
                                reason="fine-tuned BERT not present in models/intent")


@needs_bert
def test_trained_bert_loads_and_is_calibrated(cfg):
    from src.config import path
    clf = BertIntentClassifier(path(cfg, "intent_model_dir"))
    assert 0 < clf.threshold < 1 and clf.temperature > 0
    r = clf.predict("Can I book a cardiology appointment?")
    assert r["predicted_intent"] == "appointment_booking"
