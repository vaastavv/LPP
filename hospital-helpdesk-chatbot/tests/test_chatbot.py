"""Pipeline tests with stubbed models (no downloads, no network)."""

import pytest

from src import chatbot as chatbot_module
from src.chatbot import HelpdeskChatbot
from src.emotion_detector import EmotionResult
from src.intent_predictor import IntentPrediction
from src.translator import detect_language


class StubIntent:
    def __init__(self, intent: str, conf: float):
        self.intent, self.conf = intent, conf

    def predict(self, text: str) -> IntentPrediction:
        return IntentPrediction(self.intent, self.conf, [(self.intent, self.conf)])


class StubEmotion:
    def __init__(self, group: str):
        self.group = group

    def detect(self, text: str) -> EmotionResult:
        return EmotionResult(self.group, 0.9, self.group, {self.group: 0.9}, {self.group: 0.9})


def make_bot(intent="hospital_timings", conf=0.95, emotion="neutral") -> HelpdeskChatbot:
    return HelpdeskChatbot(StubIntent(intent, conf), StubEmotion(emotion), translate=False)


def test_basic_response():
    r = make_bot().respond("when is opd open")
    assert r.intent == "hospital_timings"
    assert r.fallback_reason is None
    assert "OPD" in r.response


def test_emergency_keyword_overrides_classifier():
    r = make_bot(intent="appointment_booking", emotion="fear").respond("my father has chest pain")
    assert r.intent == "emergency_assistance"
    assert r.safety_override
    assert "108" in r.response


def test_low_confidence_clarifies():
    r = make_bot(conf=0.2).respond("blah")
    assert r.fallback_reason == "low_confidence"


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        make_bot().respond("   ")


def test_hindi_round_trip(monkeypatch):
    monkeypatch.setattr(chatbot_module, "to_english", lambda text, src: ("when is opd open", "hi"))
    monkeypatch.setattr(chatbot_module, "from_english", lambda text, lang: f"[{lang}] {text}")
    bot = HelpdeskChatbot(StubIntent("hospital_timings", 0.9), StubEmotion("neutral"), translate=True)
    r = bot.respond("ओपीडी कब खुलती है?")
    assert r.language == "hi"
    assert r.response.startswith("[hi] ")
    assert r.english_text == "when is opd open"


@pytest.mark.parametrize(
    "text,lang",
    [("When is the OPD open?", "en"), ("ओपीडी कब खुलती है?", "hi"), ("मुझे Dr. Sharma से मिलना है", "hi"), ("123", "en")],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang
