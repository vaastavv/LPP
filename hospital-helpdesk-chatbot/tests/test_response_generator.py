import json
import re

import pytest

from src.response_generator import ResponseGenerator
from src.utils import EMOTION_GROUPS, INTENTS, RESPONSE_TEMPLATES_JSON


@pytest.fixture(scope="module")
def generator() -> ResponseGenerator:
    return ResponseGenerator(min_intent_confidence=0.5)


def test_templates_complete():
    templates = json.loads(RESPONSE_TEMPLATES_JSON.read_text(encoding="utf-8"))
    assert set(templates) == set(INTENTS)
    for intent, variants in templates.items():
        assert set(variants) == set(EMOTION_GROUPS), intent
        assert all(v.strip() for v in variants.values())
    assert sum(len(v) for v in templates.values()) == 70


@pytest.mark.parametrize("intent", INTENTS)
@pytest.mark.parametrize("emotion", EMOTION_GROUPS)
def test_all_placeholders_resolved(generator, intent, emotion):
    out = generator.generate(intent, emotion, 0.99)
    assert out.fallback_reason is None
    assert not re.search(r"\{\w+\}", out.text), out.text


def test_emotion_specific_template(generator):
    assert generator.generate("billing_query", "anger", 0.9).template_emotion == "anger"


def test_unknown_emotion_falls_back_to_neutral(generator):
    out = generator.generate("billing_query", "bored", 0.9)
    assert out.template_emotion == "neutral"


def test_low_confidence_asks_for_clarification(generator):
    out = generator.generate("billing_query", "fear", 0.2)
    assert out.fallback_reason == "low_confidence"
    assert "emergency" in out.text.lower()


def test_unknown_intent_generic_fallback(generator):
    out = generator.generate("parking_query", "neutral", 0.95)
    assert out.fallback_reason == "unknown_intent"
    assert "helpline" in out.text.lower()


def test_missing_emotion_variant(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"greetings": {"neutral": "Hello!"}}))
    gen = ResponseGenerator(templates_path=path, hospital_info_path=None)
    out = gen.generate("greetings", "anger", 0.9)
    assert out.text == "Hello!"
    assert out.fallback_reason == "missing_emotion_template"
