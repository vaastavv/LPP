"""
Ambiguous Reasoning (AR) task — LPP paper item #12.

Paper: Results §"LPP-informed tasks" (page 6); Supplement §1.1.1 (page 18) and
Algorithm 1; Table 2 (page 9). Grading: mean of (i) ambiguity-flag accuracy and
(ii) answer-choice accuracy. 100 items, 10 in-context examples, max_new_tokens=16.

The model is shown an ambiguous (or, as a control, unambiguous) prefix and two
candidate senses. It must first judge the PREFIX ALONE as AMBIGUOUS or NOT
AMBIGUOUS, then — after reading a disambiguating hint — choose the correct sense
A or B.

Scope note: the paper's Algorithm 1 disambiguates prefixes procedurally from a
per-prefix entropy scan over a natural corpus, but the paper does NOT release its
ambiguity data. This module uses our own small hand-written banks instead: an
AMBIGUITY_BANK of genuinely two-sense prefixes and a CONTROL_BANK of clearly
unambiguous prefixes (so the AMBIGUOUS / NOT AMBIGUOUS flag is a non-trivial
judgement). Items are sampled with randomized A/B ordering and alternating
correct sense so position and label carry no signal.
"""
from __future__ import annotations

import random

# Genuinely ambiguous prefixes: each has two plausible senses (a / b) and a hint
# that resolves it to exactly one. Format: {prefix, a, b, hint_a, hint_b}.
AMBIGUITY_BANK = [
    {"prefix": "She deposited money at the bank",
     "a": "a financial institution", "b": "the side of a river",
     "hint_a": "The teller printed a receipt for the transaction.",
     "hint_b": "The muddy shore was slippery after the heavy rain."},
    {"prefix": "The bark was loud that night",
     "a": "the sound a dog makes", "b": "the outer covering of a tree",
     "hint_a": "The neighbours complained about the restless puppy.",
     "hint_b": "A storm had split the old oak down the middle."},
    {"prefix": "He picked up the bat",
     "a": "a piece of sports equipment", "b": "a flying nocturnal animal",
     "hint_a": "He stepped up to the plate for his turn to hit.",
     "hint_b": "It had been roosting in the attic all winter."},
    {"prefix": "They watched the crane by the water",
     "a": "a tall wading bird", "b": "a construction machine",
     "hint_a": "It folded its long neck and speared a fish.",
     "hint_b": "The operator lifted a steel beam onto the pier."},
    {"prefix": "The pitcher was on the table",
     "a": "a container for liquid", "b": "a baseball player",
     "hint_a": "She filled it with lemonade for the guests.",
     "hint_b": "The coach benched him after a rough inning."},
    {"prefix": "I need a new mouse",
     "a": "a computer pointing device", "b": "a small rodent",
     "hint_a": "The old one's scroll wheel finally stopped working.",
     "hint_b": "The cat has been chasing it around the kitchen."},
    {"prefix": "She adjusted the spring",
     "a": "a coiled metal component", "b": "a natural water source",
     "hint_a": "The mattress had lost its bounce over the years.",
     "hint_b": "Fresh water bubbled up from between the rocks."},
    {"prefix": "The letter was very moving",
     "a": "emotionally touching", "b": "physically in motion",
     "hint_a": "She wept as she read her grandmother's words.",
     "hint_b": "The magnet dragged it slowly across the desk."},
    {"prefix": "He left the club early",
     "a": "a nightlife venue", "b": "an organized association",
     "hint_a": "The music was too loud and the drinks too expensive.",
     "hint_b": "The chess members voted him out as treasurer."},
    {"prefix": "The plant needs attention",
     "a": "a living green organism", "b": "a manufacturing facility",
     "hint_a": "Its leaves had started to yellow and droop.",
     "hint_b": "The assembly line halted for the third time this week."},
    {"prefix": "Watch the pupil closely",
     "a": "a student", "b": "part of the eye",
     "hint_a": "The teacher suspected the child was cheating.",
     "hint_b": "The doctor checked how it dilated under the light."},
    {"prefix": "The seal was broken",
     "a": "a wax or plastic closure", "b": "a marine mammal",
     "hint_a": "Someone had already opened the sealed envelope.",
     "hint_b": "It slid off the ice floe into the freezing sea."},
    {"prefix": "She could not find the ruler",
     "a": "a measuring instrument", "b": "a sovereign leader",
     "hint_a": "She needed to draw a straight line on the page.",
     "hint_b": "The kingdom had been without a monarch for a year."},
    {"prefix": "They admired the palm",
     "a": "a type of tree", "b": "the inner surface of the hand",
     "hint_a": "It swayed gently along the tropical beach.",
     "hint_b": "The fortune teller traced the deep lines on it."},
    {"prefix": "The band was too tight",
     "a": "an elastic loop", "b": "a group of musicians",
     "hint_a": "It left a red mark around her wrist.",
     "hint_b": "The drummer and the singer kept arguing on tour."},
    {"prefix": "He studied the character",
     "a": "a person in a story", "b": "a written symbol",
     "hint_a": "The novelist gave her a tragic backstory.",
     "hint_b": "Each Chinese glyph took hours to memorize."},
]

# Clearly unambiguous control prefixes: still two options, one obviously right
# per the hint, but the prefix itself is NOT ambiguous.
CONTROL_BANK = [
    {"prefix": "The sun rose over the mountains",
     "a": "a description of dawn", "b": "a recipe for soup",
     "hint_a": "The morning light spread across the valley.",
     "hint_b": ""},
    {"prefix": "She solved the quadratic equation",
     "a": "a mathematics task", "b": "a gardening chore",
     "hint_a": "She applied the standard formula for the two roots.",
     "hint_b": ""},
    {"prefix": "The chef seasoned the soup",
     "a": "cooking a meal", "b": "repairing a car",
     "hint_a": "He added a pinch of salt and fresh basil.",
     "hint_b": ""},
    {"prefix": "The astronaut floated in the cabin",
     "a": "being in space", "b": "swimming in a lake",
     "hint_a": "Outside the window the blue Earth curved away.",
     "hint_b": ""},
    {"prefix": "The children built a sandcastle",
     "a": "playing at the beach", "b": "writing a report",
     "hint_a": "The tide would wash it away by evening.",
     "hint_b": ""},
    {"prefix": "He tightened the last bolt",
     "a": "assembling something", "b": "reading a poem",
     "hint_a": "The bookshelf was finally sturdy enough to fill.",
     "hint_b": ""},
    {"prefix": "The orchestra tuned their instruments",
     "a": "preparing to perform music", "b": "planting vegetables",
     "hint_a": "The conductor raised the baton for the first movement.",
     "hint_b": ""},
    {"prefix": "The river froze solid in winter",
     "a": "cold-weather scene", "b": "a baking instruction",
     "hint_a": "Children skated across the thick clear ice.",
     "hint_b": ""},
]

AMBIGUOUS = "AMBIGUOUS"
NOT_AMBIGUOUS = "NOT_AMBIGUOUS"


def generate_ar(n: int = 100, seed: int = 42, ambiguous_fraction: float = 0.75) -> list[dict]:
    """
    Sample n AR items from the banks with randomized A/B ordering and alternating
    correct sense. Roughly `ambiguous_fraction` of items are genuinely ambiguous
    (gold status AMBIGUOUS); the rest are controls (NOT_AMBIGUOUS).

    Returns items: {'prefix', 'option_a', 'option_b', 'hint', 'gold_status',
    'gold_answer', 'source'}.
    """
    rng = random.Random(seed)
    items = []
    for i in range(n):
        is_amb = rng.random() < ambiguous_fraction
        entry = rng.choice(AMBIGUITY_BANK if is_amb else CONTROL_BANK)

        # Ambiguous items alternate which sense is correct so the hint (and 'A')
        # carries no positional signal. Control items have only a sensible sense
        # 'a' (hint_b is empty), so their correct sense is always 'a'.
        correct_is_a = (i % 2 == 0) if is_amb else True
        correct_sense = entry["a"] if correct_is_a else entry["b"]
        hint = entry["hint_a"] if correct_is_a else entry["hint_b"]

        # Randomize display order of the two options.
        if rng.random() < 0.5:
            option_a, option_b = entry["a"], entry["b"]
        else:
            option_a, option_b = entry["b"], entry["a"]
        gold_answer = "A" if option_a == correct_sense else "B"

        items.append({
            "prefix": entry["prefix"],
            "option_a": option_a,
            "option_b": option_b,
            "hint": hint,
            "gold_status": AMBIGUOUS if is_amb else NOT_AMBIGUOUS,
            "gold_answer": gold_answer,
            "source": "ambiguity" if is_amb else "control",
        })
    return items


def prompt_template(item: dict, in_context: list[dict]) -> str:
    """
    Build the AR prompt (Table 2 format) with `in_context` (paper: 10) examples,
    each shown with its expected two-line response.
    """
    header = (
        "Consider the ambiguous prefix and two possible senses. First judge the "
        "prefix alone as AMBIGUOUS or NOT AMBIGUOUS. Then, after reading the hint, "
        "choose the correct option A or B. Respond strictly as:\n"
        "status=AMBIGUOUS or NOT AMBIGUOUS\n"
        "answer=A or B\n"
    )

    def _block(it: dict, with_answer: bool) -> str:
        status_word = "AMBIGUOUS" if it["gold_status"] == AMBIGUOUS else "NOT AMBIGUOUS"
        body = (
            f"Prefix: {it['prefix']}. "
            f"Options: A. {it['option_a']} or B. {it['option_b']}. "
            f"Hint: {it['hint']}. Your response:"
        )
        if with_answer:
            body += f"\nstatus={status_word}\nanswer={it['gold_answer']}\n"
        return body

    shots = "\n\n".join(_block(ex, with_answer=True) for ex in in_context)
    tail = _block(item, with_answer=False)
    return f"{header}\n{shots}\n\n{tail}"


def parse_ar_response(text: str) -> dict:
    """
    Extract status in {AMBIGUOUS, NOT_AMBIGUOUS} and answer in {A, B} from the
    model output by regex. Returns {'status': ..., 'answer': ...}; a field is
    None when it cannot be found.
    """
    import re

    up = text.upper()
    status = None
    # Check "NOT AMBIGUOUS" first, since "AMBIGUOUS" is a substring of it.
    if re.search(r"NOT[\s_]*AMBIGUOUS", up):
        status = NOT_AMBIGUOUS
    elif "AMBIGUOUS" in up:
        status = AMBIGUOUS

    answer = None
    m = re.search(r"ANSWER\s*=\s*([AB])", up)
    if m:
        answer = m.group(1)
    else:
        m = re.search(r"\b([AB])\b", up)
        if m:
            answer = m.group(1)

    return {"status": status, "answer": answer}


def grade_ar(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "status_accuracy": 0.0, "answer_accuracy": 0.0, "mean_accuracy": 0.0}
    n = len(records)
    status_acc = sum(r["status_correct"] for r in records) / n
    answer_acc = sum(r["answer_correct"] for r in records) / n
    mean_acc = sum((r["status_correct"] + r["answer_correct"]) / 2 for r in records) / n
    return {
        "n": n,
        "status_accuracy": status_acc,
        "answer_accuracy": answer_acc,
        "mean_accuracy": mean_acc,
    }
