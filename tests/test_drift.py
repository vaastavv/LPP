"""
Offline tests for the drift-monitoring experiment (src/drift.py). No model or
matplotlib needed: perturbations and detection statistics only.

Run:  python tests/test_drift.py
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift import (
    KINDS, DEFAULT_SEVERITIES, perturb, char_noise, word_dropout,
    paired_bootstrap_detect, detection_level, aligned_by_level,
)


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------
def test_severity_zero_is_identity():
    text = "A basket has 12 apples. How many now?"
    for kind in KINDS:
        assert perturb(text, kind, 0.0, seed=1) == text


def test_perturb_deterministic():
    text = "A printer prints 5 pages per minute."
    assert perturb(text, "char_noise", 0.3, seed=7) == perturb(text, "char_noise", 0.3, seed=7)
    assert perturb(text, "char_noise", 0.3, seed=7) != perturb(text, "char_noise", 0.3, seed=8)


def test_char_noise_monotonic_severity():
    text = "the quick brown fox jumps over the lazy dog " * 3

    def n_changed(sev):
        # average edit count over several seeds
        total = 0
        for s in range(8):
            c = char_noise(text, sev, random.Random(s))
            total += sum(1 for a, b in zip(text, c) if a != b)
        return total / 8

    assert n_changed(0.05) < n_changed(0.2) < n_changed(0.5)


def test_word_dropout_never_empty():
    text = "one two three four five"
    for s in range(20):
        out = word_dropout(text, 1.0, random.Random(s))
        assert out.strip() != ""


# ---------------------------------------------------------------------------
# Detection statistics
# ---------------------------------------------------------------------------
def test_bootstrap_detect_up_and_down():
    base = [0.0] * 60
    higher = [1.0] * 60
    lower = [-1.0] * 60
    assert paired_bootstrap_detect(base, higher, "up")["detected"]
    assert not paired_bootstrap_detect(base, higher, "down")["detected"]
    assert paired_bootstrap_detect(base, lower, "down")["detected"]


def test_bootstrap_no_detect_when_equal():
    rng = random.Random(0)
    base = [rng.gauss(0, 1) for _ in range(80)]
    same = list(base)
    assert not paired_bootstrap_detect(base, same, "up")["detected"]
    assert not paired_bootstrap_detect(base, same, "down")["detected"]


def _make_rows():
    """Synthetic drift data: entropy rises at level 2, accuracy drops at level 4."""
    rng = random.Random(0)
    rows = []
    for lvl in range(6):
        ent_shift = 0.0 if lvl < 2 else 0.8 * (lvl - 1)
        acc_prob = 0.9 if lvl < 4 else 0.2
        for i in range(40):
            rows.append({
                "level": lvl, "sample_id": [i, 0],
                "min_entropy": 1.0 + ent_shift + rng.gauss(0, 0.05),
                "is_correct": int(rng.random() < acc_prob),
                "correct_at_baseline": True,
            })
    return rows


def test_aligned_and_detection_lead():
    rows = _make_rows()
    _, ent = aligned_by_level(rows, "min_entropy")
    _, acc = aligned_by_level(rows, "is_correct")
    ent_lvl = detection_level(ent, "up")
    acc_lvl = detection_level(acc, "down")
    assert ent_lvl == 2, ent_lvl
    assert acc_lvl == 4, acc_lvl
    # entropy leads accuracy by 2 levels
    assert acc_lvl - ent_lvl == 2


def test_aligned_uses_common_ids():
    rows = [
        {"level": 0, "sample_id": [0, 0], "v": 1.0},
        {"level": 0, "sample_id": [1, 0], "v": 2.0},
        {"level": 1, "sample_id": [0, 0], "v": 3.0},   # id [1,0] missing at level 1
    ]
    levels, per = aligned_by_level(rows, "v")
    assert levels == [0, 1]
    assert per[0] == [1.0] and per[1] == [3.0]   # only common id [0,0] kept


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("\nAll drift tests passed.")
