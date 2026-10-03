"""Emergency handling.

Two independent signals route a message to the emergency handler:
  * the intent model: P(emergency_assistance) >= a validation-selected emergency threshold
  * a conservative red-flag phrase detector (safety net that does not depend on the model)

The handler never diagnoses, never gives treatment instructions beyond "get emergency help now", and
never invents phone numbers or locations: it tells the user to contact *their local* emergency number.
"""
from __future__ import annotations

import re

# High-severity phrases only. Generic words such as "ambulance" or "emergency department" alone are NOT
# here, because they appear in ordinary questions ("What's the number for the emergency department?").
RED_FLAG_PATTERNS = [
    r"\bchest pain", r"\bchest (?:is )?(?:hurting|tight)", r"\bheart attack",
    r"\b(?:not|isn'?t|stopped|can'?t|cannot|cant|trouble|difficulty|struggling to) breath",
    r"\bnot respon", r"\bunconscious", r"\bcollaps", r"\bpassed out", r"\bfainted", r"\bwon'?t wake",
    r"\b(?:severe|heavy|heavily|a lot of|lot of|won'?t stop|uncontrolled) bleed", r"\bbleeding (?:heavily|a lot|badly)",
    r"\bstroke\b(?! (?:unit|centre|center|clinic|department|ward|rehab))", r"\bface (?:is )?droop", r"\bseizure", r"\bfits?\b(?= and| right now)", r"\bconvuls",
    r"\boverdos", r"\bpoison", r"\bswallowed (?:a |some |my |his |her )?(?:battery|pills|poison|coin|bleach)",
    r"\bchoking", r"\bturning blue", r"\blips (?:are )?(?:turning )?blue", r"\bdrown", r"\belectrocut",
    r"\bstabbed", r"\bshot\b", r"\bsnake ?bite", r"\bbitten by a snake", r"\banaphyla", r"\bthroat (?:is )?swelling",
    r"\bsend (?:an |the )?ambulance", r"\bneed (?:an |the )?ambulance", r"\bcall (?:an |the )?ambulance",
    r"\bthis is an emergency", r"^\s*emergency\b", r"\bemergency!", r"\burgent(?:ly)? (?:help|medical)",
    r"\bi think i'?m dying", r"\bcoughing (?:up )?blood", r"\bvomiting blood",
]
SELF_HARM_PATTERNS = [
    r"\bkill (?:my ?self|myself)", r"\bend(?:ing)? (?:my|his|her) life", r"\bsuicid",
    r"\b(?:hurt|harm) (?:my ?self|myself)", r"\bself[- ]harm", r"\bdon'?t want to live",
]
NEGATIONS = [r"\bnot an emergency", r"\bnon[- ]emergency", r"\bno emergency", r"\bisn'?t an emergency"]

_RF = [re.compile(p, re.I) for p in RED_FLAG_PATTERNS]
_SH = [re.compile(p, re.I) for p in SELF_HARM_PATTERNS]
_NEG = [re.compile(p, re.I) for p in NEGATIONS]

CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way, and I'm glad you reached out. If you might act on these "
    "thoughts or are in danger, please call your local emergency number now or go to the nearest emergency "
    "department. You can also contact a local crisis helpline or someone you trust and tell them how you "
    "feel. You don't have to go through this alone."
)

FALLBACK_EMERGENCY_RESPONSE = (
    "This sounds urgent. If this is a medical emergency, call your local emergency number or go to the "
    "nearest emergency department now. Do not wait for a reply here."
)


def red_flags(text: str) -> list[str]:
    if any(n.search(text) for n in _NEG):
        return []
    return [p.pattern for p in _RF if p.search(text)]


def self_harm(text: str) -> bool:
    return any(p.search(text) for p in _SH)


def emergency_response(templates: dict, emotion: str | None) -> str:
    block = templates.get("emergency_assistance", {})
    return block.get(emotion or "neutral") or block.get("neutral") or FALLBACK_EMERGENCY_RESPONSE
