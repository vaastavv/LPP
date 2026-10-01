import re

import pytest

from src.config import GOEMOTIONS_LABELS, INTENTS
from src.data.validate import validate_templates
from src.response.selector import ResponseSelector


def test_template_matrix_complete(templates):
    errors, warnings = validate_templates(templates)
    assert errors == []
    assert warnings == []  # no reused strings
    assert sorted(templates) == sorted(INTENTS)
    assert sum(len(v) for v in templates.values()) == len(INTENTS) * len(GOEMOTIONS_LABELS) == 560


@pytest.mark.parametrize("intent", INTENTS)
def test_every_emotion_present_and_non_empty(templates, intent):
    assert set(templates[intent]) == set(GOEMOTIONS_LABELS)
    assert all(r.strip() for r in templates[intent].values())


def test_validator_catches_missing(templates):
    broken = {k: dict(v) for k, v in templates.items()}
    del broken["billing_query"]["fear"]
    broken["greeting"]["joy"] = " "
    errors, _ = validate_templates(broken)
    assert any("billing_query" in e for e in errors) and any("greeting/joy" in e for e in errors)


def test_no_fabricated_contact_details(templates):
    for intent, block in templates.items():
        for emo, r in block.items():
            assert not re.search(r"\d{3,}", r), f"{intent}/{emo} contains a number: {r}"
            assert "@" not in r and "http" not in r


def test_emergency_templates_point_to_local_services(templates):
    for emo, r in templates["emergency_assistance"].items():
        assert "local emergency number" in r and "emergency department" in r, emo


def test_selector_lookup_and_fallback(templates):
    s = ResponseSelector(templates)
    r = s.select("appointment_booking", "fear")
    assert r["response"] == templates["appointment_booking"]["fear"] and not r["emotion_fallback"]
    r = s.select("appointment_booking", "not_an_emotion")
    assert r["response"] == templates["appointment_booking"]["neutral"] and r["emotion_fallback"]
    with pytest.raises(KeyError):
        s.select("not_an_intent", "joy")


def test_intent_content_stable_across_emotions(templates):
    """Emotion changes tone, not meaning: every response for an intent shares the same action body."""
    for intent, block in templates.items():
        neutral = block["neutral"]
        body = neutral.split(". ", 1)[-1] if intent in ("greeting", "thank_you", "goodbye", "emergency_assistance") else neutral
        for emo, r in block.items():
            assert r.endswith(body), (intent, emo)
