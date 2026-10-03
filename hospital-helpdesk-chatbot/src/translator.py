"""English <-> Hindi translation with deep-translator (Google Translate backend).

The intent and emotion models are English-only, so incoming Hindi queries are
translated to English, and the final English response is translated back into
the user's language.

Language detection is done locally (Devanagari script check) to avoid an extra
network round-trip; Whisper's detected language or an explicit user choice in
the UI can override it. Translation failures never break the chat: the original
text is returned and the failure is logged.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.utils import get_logger

logger = get_logger(__name__)

SUPPORTED_LANGUAGES: dict[str, str] = {"en": "English", "hi": "Hindi"}
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_MAX_CHARS = 4800  # Google Translate limit is 5000 characters per request


def detect_language(text: str, threshold: float = 0.3) -> str:
    """Return ``'hi'`` if a meaningful share of letters are Devanagari, else ``'en'``."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    devanagari = sum(1 for c in letters if _DEVANAGARI_RE.match(c))
    return "hi" if devanagari / len(letters) >= threshold else "en"


@lru_cache(maxsize=8)
def _translator(source: str, target: str):
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=source, target=target)


def _chunks(text: str, size: int = _MAX_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > size and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def translate(text: str, source: str = "auto", target: str = "en") -> str:
    """Translate ``text``; returns the input unchanged on failure or no-op."""
    if not text or not text.strip() or source == target:
        return text
    if target not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported target language '{target}'. Supported: {list(SUPPORTED_LANGUAGES)}")
    try:
        translator = _translator(source, target)
        return " ".join(translator.translate(chunk) or chunk for chunk in _chunks(text))
    except Exception as exc:  # network errors, rate limits, API changes
        logger.warning("Translation %s->%s failed (%s); returning original text", source, target, exc)
        return text


def to_english(text: str, source_lang: str | None = None) -> tuple[str, str]:
    """Translate an incoming query to English.

    Returns ``(english_text, detected_language)``.
    """
    lang = source_lang if source_lang in SUPPORTED_LANGUAGES else detect_language(text)
    if lang == "en":
        return text, "en"
    return translate(text, source=lang, target="en"), lang


def from_english(text: str, target_lang: str) -> str:
    """Translate an English response back into the user's language."""
    if target_lang == "en" or target_lang not in SUPPORTED_LANGUAGES:
        return text
    return translate(text, source="en", target=target_lang)
