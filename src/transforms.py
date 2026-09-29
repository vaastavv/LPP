"""
Utilities for generating meaning-preserving surface variants of a base prompt.

These implement the "change the surface, keep the meaning" idea. Number/name
swaps that would change the ANSWER are NOT auto-generated here (that needs a
solver); for those, author the variant and its correct answer directly in
data/problems.jsonl. The helpers below only produce variants whose correct
answer is unchanged, so they are safe to auto-expand.
"""
from __future__ import annotations
import random


def swap_names(text: str, mapping: dict) -> str:
    """Replace whole-word names using mapping, e.g. {'Alice': 'Ravi'}."""
    out = text
    for a, b in mapping.items():
        out = _replace_word(out, a, b)
    return out


def _replace_word(text, a, b):
    import re
    return re.sub(rf"\b{re.escape(a)}\b", b, text)


def inject_noop(text: str, distractor: str) -> str:
    """
    Append a relevant-looking but inconsequential sentence (GSM-NoOp style).
    The answer must not depend on `distractor`.
    """
    text = text.rstrip()
    sep = " " if text.endswith(('.', '?', '!')) else ". "
    return f"{text}{sep}{distractor}"


def reorder_sentences(text: str, seed: int = 42) -> str:
    """Shuffle the order of information-bearing sentences (keeps content)."""
    import re
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    if len(parts) < 3:
        return text  # too short to meaningfully reorder without changing a question
    body, tail = parts[:-1], parts[-1]     # keep the final question in place
    rng = random.Random(seed)
    rng.shuffle(body)
    return " ".join(body + [tail])


def auto_variants(base_prompt: str, spec: dict | None) -> list:
    """
    Build answer-preserving variants from an optional 'auto' spec on a family:
      spec = {
        "names":      {"Alice": "Ravi"},          # -> a paraphrase-ish rename
        "noop":       "The room was painted blue.",# -> a NoOp variant
        "reorder":    true                          # -> a reordered variant
      }
    Returns a list of {"kind":..., "prompt":...} (answer stays the family's answer).
    """
    variants = []
    if not spec:
        return variants
    if "names" in spec:
        variants.append({"kind": "rename", "prompt": swap_names(base_prompt, spec["names"])})
    if "noop" in spec:
        variants.append({"kind": "noop", "prompt": inject_noop(base_prompt, spec["noop"])})
    if spec.get("reorder"):
        variants.append({"kind": "reorder", "prompt": reorder_sentences(base_prompt)})
    return variants
