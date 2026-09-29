"""
Offline tests for the LPP task/analysis additions that need no model download:
SPC generation + grading, AR generation + parsing, and the hourglass detector.

Run:  python -m pytest tests/ -q      (or)      python tests/test_tasks.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spc_task import (
    generate_spc, prompt_template as spc_prompt, char_f1, parse_prediction,
    grade_spc, ALPHABET, SEQ_LEN, MASK_LEN,
)
from src.ar_task import (
    generate_ar, prompt_template as ar_prompt, parse_ar_response, grade_ar,
    AMBIGUOUS, NOT_AMBIGUOUS,
)
from latent_analysis import hourglass_score


# ---------------------------------------------------------------------------
# SPC
# ---------------------------------------------------------------------------
def test_spc_generation_shapes():
    items = generate_spc(20, seed=42)
    assert len(items) == 20
    for it in items:
        assert len(it["prompt_prefix"]) == SEQ_LEN - MASK_LEN
        assert len(it["gold"]) == MASK_LEN
        assert set(it["prompt_prefix"] + it["gold"]) <= set(ALPHABET)


def test_spc_deterministic():
    assert generate_spc(10, seed=7) == generate_spc(10, seed=7)
    assert generate_spc(10, seed=7) != generate_spc(10, seed=8)


def test_spc_alternation_gold():
    # first alternation item (rule cycles: index 0 -> 'alternation')
    it = generate_spc(1, seed=42)[0]
    assert it["rule"] == "alternation"
    seq = it["prompt_prefix"] + it["gold"]
    # strict two-symbol alternation
    assert all(seq[i] == seq[0] for i in range(0, SEQ_LEN, 2))
    assert all(seq[i] == seq[1] for i in range(1, SEQ_LEN, 2))


def test_char_f1():
    assert char_f1("ABC", "ABC") == 1.0
    assert char_f1("", "ABC") == 0.0
    assert char_f1("XYZ", "ABC") == 0.0
    # multiset overlap: pred {A,B}, gold {A,B,C} -> P=1, R=2/3, F1=0.8
    assert abs(char_f1("AB", "ABC") - 0.8) < 1e-9


def test_parse_prediction():
    assert parse_prediction("ABC") == "ABC"          # bare answer
    assert parse_prediction("ABA\n") == "ABA"        # trailing newline
    assert parse_prediction("ABCDEF") == "ABC"       # capped at MASK_LEN
    assert parse_prediction(" abc ") == "ABC"         # lowercased + padded
    assert parse_prediction(":ABC") == "ABC"         # leading punctuation skipped
    assert parse_prediction("xyz") == ""             # no in-alphabet symbols
    assert parse_prediction("123 %%%") == ""         # no letters at all


def test_grade_spc():
    recs = [{"char_f1": 1.0}, {"char_f1": 0.0}]
    s = grade_spc(recs)
    assert s["n"] == 2 and abs(s["mean_char_f1"] - 0.5) < 1e-9


def test_spc_prompt_has_shots():
    shots = [("ABAB", "ABA"), ("CDCD", "CDC")]
    p = spc_prompt("ABABABAB", shots)
    assert p.count("Sequence:") == 3      # 2 shots + the query
    assert p.rstrip().endswith("Answer:")


# ---------------------------------------------------------------------------
# AR
# ---------------------------------------------------------------------------
def test_ar_generation_shapes():
    items = generate_ar(30, seed=42)
    assert len(items) == 30
    for it in items:
        assert it["gold_status"] in (AMBIGUOUS, NOT_AMBIGUOUS)
        assert it["gold_answer"] in ("A", "B")
        # gold answer must point at a real option
        assert it["option_a"] and it["option_b"]


def test_ar_deterministic():
    assert generate_ar(20, seed=1) == generate_ar(20, seed=1)


def test_ar_has_both_statuses():
    statuses = {it["gold_status"] for it in generate_ar(50, seed=42)}
    assert statuses == {AMBIGUOUS, NOT_AMBIGUOUS}


def test_parse_ar_response():
    r = parse_ar_response("status=AMBIGUOUS\nanswer=B")
    assert r["status"] == AMBIGUOUS and r["answer"] == "B"
    r = parse_ar_response("I think status = NOT AMBIGUOUS and answer=A")
    assert r["status"] == NOT_AMBIGUOUS and r["answer"] == "A"
    r = parse_ar_response("completely unparseable")
    assert r["status"] is None and r["answer"] is None


def test_grade_ar():
    recs = [
        {"status_correct": 1, "answer_correct": 1},
        {"status_correct": 1, "answer_correct": 0},
        {"status_correct": 0, "answer_correct": 0},
    ]
    s = grade_ar(recs)
    assert abs(s["status_accuracy"] - 2 / 3) < 1e-9
    assert abs(s["answer_accuracy"] - 1 / 3) < 1e-9
    assert abs(s["mean_accuracy"] - 0.5) < 1e-9


def test_ar_prompt_has_shots():
    items = generate_ar(11, seed=42)
    p = ar_prompt(items[-1], items[:10])
    assert p.count("Prefix:") == 11      # 10 shots + the query
    assert "status=" in p and "answer=" in p


# ---------------------------------------------------------------------------
# hourglass detector (#2)
# ---------------------------------------------------------------------------
def test_hourglass_score():
    # high ends, low middle -> positive (an hourglass)
    assert hourglass_score([10, 5, 1, 5, 10]) > 0
    # flat -> ~0
    assert abs(hourglass_score([5, 5, 5, 5, 5])) < 1e-9
    # bump in the middle -> negative
    assert hourglass_score([1, 5, 10, 5, 1]) < 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("\nAll task tests passed.")
