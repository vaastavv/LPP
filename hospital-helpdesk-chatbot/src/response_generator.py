"""Intent + emotion aware response generation from ``data/response_templates.json``.

Fallback chain (first match wins):

1. Intent missing / confidence below ``min_intent_confidence`` -> clarification response.
2. Intent not present in the templates -> generic fallback response.
3. Emotion not present for the intent -> ``neutral`` variant of that intent.
4. ``neutral`` also missing -> first available variant of that intent.

Templates may contain ``{placeholders}`` (e.g. ``{helpline}``) which are filled
from ``config/hospital_info.json``. Unknown placeholders are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils import (
    EMOTION_GROUPS,
    HOSPITAL_INFO_JSON,
    RESPONSE_TEMPLATES_JSON,
    SafeFormatDict,
    get_logger,
    load_json,
)

logger = get_logger(__name__)

DEFAULT_EMOTION = "neutral"

CLARIFICATION_RESPONSES: dict[str, str] = {
    "positive": "I'd love to help! Could you tell me a little more? For example, you can ask me to book an appointment, find a doctor, check timings, lab reports, billing or insurance.",
    "neutral": "I'm sorry, I didn't quite understand that. Could you rephrase? I can help with appointments, doctors, hospital timings, lab reports, billing, insurance, pharmacy and contact details.",
    "sadness": "I'm sorry, I want to make sure I help you properly. Could you tell me a bit more about what you need? I can help with appointments, doctors, reports, billing and more.",
    "fear": "I want to make sure you get the right help. Could you tell me a little more about what you need? If this is a medical emergency, please call {emergency_number} immediately.",
    "anger": "I'm sorry for the trouble. I want to get this right for you - could you tell me a little more about the issue? You can also call our helpline {helpline} to speak with a person.",
}

GENERIC_FALLBACK = (
    "I'm sorry, I can't help with that here. Please call our helpline {helpline} or email {email}, "
    "and our team will assist you. For emergencies, call {emergency_number}."
)


@dataclass
class GeneratedResponse:
    text: str
    intent: str | None
    emotion: str
    template_intent: str | None
    template_emotion: str | None
    fallback_reason: str | None = None


class ResponseGenerator:
    def __init__(
        self,
        templates_path: str | Path = RESPONSE_TEMPLATES_JSON,
        hospital_info_path: str | Path | None = HOSPITAL_INFO_JSON,
        min_intent_confidence: float = 0.5,
    ) -> None:
        self.templates: dict[str, dict[str, str]] = load_json(templates_path)
        self.context: dict[str, str] = {}
        if hospital_info_path and Path(hospital_info_path).exists():
            self.context = load_json(hospital_info_path)
        self.min_intent_confidence = min_intent_confidence
        self._validate()

    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        for intent, variants in self.templates.items():
            missing = [e for e in EMOTION_GROUPS if e not in variants]
            if missing:
                logger.warning("Intent '%s' has no template for emotions %s (fallback will be used)", intent, missing)

    def _render(self, template: str) -> str:
        return template.format_map(SafeFormatDict(self.context))

    @property
    def intents(self) -> list[str]:
        return list(self.templates)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        intent: str | None,
        emotion: str | None = None,
        intent_confidence: float | None = None,
    ) -> GeneratedResponse:
        emotion = emotion if emotion in EMOTION_GROUPS else DEFAULT_EMOTION

        # 1) low confidence / missing intent -> ask for clarification
        if not intent or (intent_confidence is not None and intent_confidence < self.min_intent_confidence):
            reason = "missing_intent" if not intent else "low_confidence"
            text = CLARIFICATION_RESPONSES.get(emotion, CLARIFICATION_RESPONSES[DEFAULT_EMOTION])
            return GeneratedResponse(self._render(text), intent, emotion, None, emotion, reason)

        # 2) unknown intent -> generic fallback
        variants = self.templates.get(intent)
        if not variants:
            logger.warning("No templates for intent '%s'; using generic fallback", intent)
            return GeneratedResponse(self._render(GENERIC_FALLBACK), intent, emotion, None, None, "unknown_intent")

        # 3/4) emotion fallback inside the intent
        if emotion in variants:
            return GeneratedResponse(self._render(variants[emotion]), intent, emotion, intent, emotion)
        fallback_emotion = DEFAULT_EMOTION if DEFAULT_EMOTION in variants else next(iter(variants))
        return GeneratedResponse(
            self._render(variants[fallback_emotion]), intent, emotion, intent, fallback_emotion, "missing_emotion_template"
        )

    def __call__(self, intent: str | None, emotion: str | None = None, intent_confidence: float | None = None) -> str:
        return self.generate(intent, emotion, intent_confidence).text
