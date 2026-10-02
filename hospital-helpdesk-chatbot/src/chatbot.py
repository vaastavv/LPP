"""End-to-end helpdesk pipeline: translate -> intent + emotion -> template -> translate back.

The UI (``app.py``) and any other front-end (CLI, REST API) only talk to
``HelpdeskChatbot.respond``; models are injected so they can be mocked in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.emotion_detector import EmotionDetector, EmotionResult
from src.intent_predictor import IntentPredictor
from src.response_generator import ResponseGenerator
from src.translator import from_english, to_english
from src.utils import contains_emergency_keyword, get_logger, normalize_text

logger = get_logger(__name__)


@dataclass
class ChatResponse:
    user_text: str
    english_text: str
    language: str
    intent: str
    intent_confidence: float
    top_intents: list[tuple[str, float]]
    emotion: str
    emotion_confidence: float
    emotion_top_label: str
    emotion_scores: dict[str, float]
    response_en: str
    response: str
    fallback_reason: str | None = None
    safety_override: bool = False
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class HelpdeskChatbot:
    def __init__(
        self,
        intent_predictor: IntentPredictor,
        emotion_detector: EmotionDetector,
        response_generator: ResponseGenerator | None = None,
        min_intent_confidence: float = 0.5,
        translate: bool = True,
    ) -> None:
        self.intent_predictor = intent_predictor
        self.emotion_detector = emotion_detector
        self.response_generator = response_generator or ResponseGenerator(min_intent_confidence=min_intent_confidence)
        self.translate = translate

    @property
    def min_intent_confidence(self) -> float:
        return self.response_generator.min_intent_confidence

    @min_intent_confidence.setter
    def min_intent_confidence(self, value: float) -> None:
        self.response_generator.min_intent_confidence = value

    def respond(self, text: str, language: str | None = None) -> ChatResponse:
        """Answer a user query.

        Args:
            text: raw user message (English or Hindi).
            language: ``"en"`` / ``"hi"`` to force, ``None``/``"auto"`` to detect.
        """
        text = normalize_text(text)
        if not text:
            raise ValueError("Empty query")

        source = None if language in (None, "auto") else language
        if self.translate:
            english, lang = to_english(text, source)
        else:
            english, lang = text, source or "en"

        intent_pred = self.intent_predictor.predict(english)
        emotion: EmotionResult = self.emotion_detector.detect(english)

        intent, confidence = intent_pred.intent, intent_pred.confidence
        safety_override = False
        if contains_emergency_keyword(english) and intent != "emergency_assistance":
            logger.warning("Emergency keyword detected; overriding intent '%s' -> emergency_assistance", intent)
            intent, confidence, safety_override = "emergency_assistance", 1.0, True

        generated = self.response_generator.generate(intent, emotion.group, confidence)
        response = from_english(generated.text, lang) if self.translate else generated.text

        return ChatResponse(
            user_text=text,
            english_text=english,
            language=lang,
            intent=intent,
            intent_confidence=float(confidence),
            top_intents=intent_pred.top_k,
            emotion=emotion.group,
            emotion_confidence=emotion.confidence,
            emotion_top_label=emotion.top_label,
            emotion_scores=emotion.group_scores,
            response_en=generated.text,
            response=response,
            fallback_reason=generated.fallback_reason,
            safety_override=safety_override,
        )


def build_default_chatbot(min_intent_confidence: float = 0.5) -> HelpdeskChatbot:
    return HelpdeskChatbot(
        IntentPredictor(), EmotionDetector(), min_intent_confidence=min_intent_confidence
    )


if __name__ == "__main__":
    bot = build_default_chatbot()
    print("Hospital helpdesk (type 'quit' to exit)")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        r = bot.respond(query)
        print(f"[{r.intent} {r.intent_confidence:.2f} | {r.emotion} {r.emotion_confidence:.2f} | {r.language}]")
        print(r.response)
