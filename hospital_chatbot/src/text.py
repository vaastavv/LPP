"""Input validation and minimal text normalization shared by training and inference.

Normalization is deliberately minimal: Transformer tokenizers handle casing/punctuation themselves, and
aggressive cleanup (stemming, stop-word removal) destroys signal such as "not" or "already".
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_CHARS = 1000
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_WS = re.compile(r"\s+")


@dataclass
class ValidationResult:
    ok: bool
    text: str
    reason: str | None = None


def normalize(text: str) -> str:
    """Unicode NFKC, strip control chars, unify quotes, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", text).strip()


def validate_input(text: object) -> ValidationResult:
    if not isinstance(text, str):
        return ValidationResult(False, "", "input must be a string")
    norm = normalize(text)
    if not norm:
        return ValidationResult(False, "", "empty input")
    if not re.search(r"[^\W_]", norm):
        return ValidationResult(False, norm, "input has no letters or digits")
    if len(norm) > MAX_CHARS:
        # Truncate rather than reject: a long emergency message must still be processed.
        norm = norm[:MAX_CHARS]
    return ValidationResult(True, norm)


def dedup_key(text: str) -> str:
    """Key under which trivially different strings collide (case, punctuation, whitespace)."""
    t = normalize(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return _WS.sub(" ", t).strip()
