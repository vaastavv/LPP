"""
Turn a raw model response into a comparable answer, and grade it.

Each problem declares an answer_type:
  - "numeric" : the answer is a number (arithmetic / word problems)
  - "mcq"     : the answer is a choice letter, e.g. A or B
  - "exact"   : the answer is a short string matched after normalisation

Extraction is deliberately forgiving: models phrase answers in many ways, so we
first look for an explicit "Answer: X" marker, then fall back to a type-specific
heuristic. Grading noise is a known limitation (see the report), so keep the
raw output too and spot-check.
"""
from __future__ import annotations
import re

_ANSWER_MARKER = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_LETTER = re.compile(r"\b([A-D])\b")


def _last_number(text: str):
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract(raw: str, answer_type: str):
    """Return a normalised answer string, or None if nothing could be extracted."""
    if raw is None:
        return None
    raw = raw.strip()

    # 1) explicit marker wins, when present
    marker = _ANSWER_MARKER.search(raw)
    region = marker.group(1).strip() if marker else raw

    if answer_type == "numeric":
        return _last_number(region) or _last_number(raw)

    if answer_type == "mcq":
        # prefer a letter inside the marker region, else anywhere in the reply
        m = _LETTER.search(region.upper()) or _LETTER.search(raw.upper())
        return m.group(1) if m else None

    if answer_type == "exact":
        # first line of the marker region, normalised
        line = region.splitlines()[0] if region else raw
        return _normalise(line)

    raise ValueError(f"unknown answer_type: {answer_type}")


def _normalise(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)      # drop punctuation
    s = re.sub(r"\s+", " ", s)
    return s


def grade(extracted, gold, answer_type: str) -> bool:
    if extracted is None:
        return False
    if answer_type == "numeric":
        try:
            return abs(float(extracted) - float(gold)) < 1e-6
        except (TypeError, ValueError):
            return False
    if answer_type == "mcq":
        return str(extracted).strip().upper() == str(gold).strip().upper()
    if answer_type == "exact":
        return _normalise(str(extracted)) == _normalise(str(gold))
    raise ValueError(f"unknown answer_type: {answer_type}")
