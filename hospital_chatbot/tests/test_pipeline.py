"""End-to-end: query -> intent -> emotion -> policy -> response.

Uses the selected intent model (BERT if trained, else the committed baseline) and a stub emotion model
unless the real GoEmotions checkpoint is present.
"""
import json

import pytest

from conftest import StubEmotion, emotion_model_available, trained_bert_available
from src.config import resolve
from src.intent.predictor import load_intent_classifier
from src.pipeline import HospitalChatbot
from src.policy.router import decide
from src.response.selector import ResponseSelector


@pytest.fixture(scope="module")
def selector(templates):
    return ResponseSelector(templates)


@pytest.fixture(scope="module")
def intent_clf(cfg):
    kind = None if resolve("models/selected_model.json").exists() else "tfidf_logreg"
    if kind is None and json.loads(resolve("models/selected_model.json").read_text())["selected"] == "bert" \
            and not trained_bert_available(cfg):
        kind = "tfidf_logreg"
    return load_intent_classifier(kind, cfg)


@pytest.fixture(scope="module")
def bot(intent_clf, selector, cfg):
    if emotion_model_available(cfg):
        from src.emotion.predictor import EmotionClassifier
        emo = EmotionClassifier.from_config(cfg)
    else:
        emo = StubEmotion("fear")
    return HospitalChatbot(intent_clf, emo, selector)


def test_case1_doctor_search(bot):
    r = bot.respond("I need to see a cardiologist tomorrow.")
    assert r["route"] == "intent" and r["final_intent"] == "doctor_search", r


def test_case2_rescheduling(bot):
    r = bot.respond("I already have an appointment but need to move it.")
    assert r["route"] == "intent" and r["final_intent"] == "appointment_rescheduling", r


def test_case3_cancellation(bot):
    r = bot.respond("I want to cancel my appointment.")
    assert r["route"] == "intent" and r["final_intent"] == "appointment_cancellation", r


def test_case4_emergency_takes_priority(bot, templates):
    r = bot.respond("My father is having severe chest pain and I'm terrified.")
    assert r["route"] == "emergency" and r["final_intent"] == "emergency_assistance", r
    assert r["response"] in templates["emergency_assistance"].values()
    assert "local emergency number" in r["response"]


def test_case5_out_of_domain_not_confident(bot):
    r = bot.respond("Tell me a joke.")
    assert r["route"] == "clarification", r
    assert r["final_intent"] is None
    assert not (r["intent"]["status"] == "confident" and r["intent"]["intent"] == "hospital_timings")


def test_emotion_changes_tone_not_intent(intent_clf, selector):
    for emo in ("joy", "fear"):
        bot = HospitalChatbot(intent_clf, StubEmotion(emo), selector)
        r = bot.respond("I want to cancel my appointment.")
        assert r["final_intent"] == "appointment_cancellation"
        assert r["template_key"] == f"appointment_cancellation/{emo}"


def test_invalid_input(bot):
    for bad in ["", "   ", "???", None]:
        r = bot.respond(bad)
        assert r["route"] == "invalid_input" and r["valid"] is False


def test_self_harm_routes_to_crisis(bot):
    r = bot.respond("I want to kill myself")
    assert r["route"] == "crisis" and "local emergency number" in r["response"]


def test_red_flag_overrides_low_confidence(selector):
    low = {"intent": None, "confidence": 0.2, "status": "uncertain", "predicted_intent": "greeting",
           "emergency_probability": 0.0, "top_k": []}
    d = decide("he is not breathing", low, {"emotion": "fear"}, selector, emergency_threshold=0.5)
    assert d.route == "emergency"


def test_non_emergency_negation(selector):
    low = {"intent": "billing_query", "confidence": 0.95, "status": "confident", "predicted_intent": "billing_query",
           "emergency_probability": 0.0, "top_k": []}
    d = decide("This is not an emergency, I just have a question about my bill", low, {"emotion": "neutral"},
               selector, emergency_threshold=0.5)
    assert d.route == "intent"
