"""
Offline tests that do NOT need a model download. They check the parts most
likely to have bugs: answer extraction/grading, metric aggregation, transforms,
and the latent-metric math (on synthetic tensors).

Run:  python -m pytest tests/ -q      (or)      python tests/test_offline.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.answer_extract import extract, grade
from src.metrics import summarize
from src.transforms import inject_noop, swap_names, auto_variants


def test_extract_numeric():
    assert extract("The total is 14 apples. Answer: 14", "numeric") == "14"
    assert extract("so it comes to 1,250 in the end", "numeric") == "1250"
    assert grade(extract("Answer: 40", "numeric"), "40", "numeric")
    assert not grade(extract("Answer: 41", "numeric"), "40", "numeric")


def test_extract_mcq():
    assert extract("I think the answer=B because of the river", "mcq") == "B"
    assert extract("Answer: A", "mcq") == "A"
    assert grade(extract("The correct one is B.", "mcq"), "B", "mcq")


def test_extract_exact():
    assert grade(extract("Answer: Green.", "exact"), "green", "exact")


def test_transforms():
    base = "Alice has 3 pens. How many pens does Alice have?"
    assert "Ravi" in swap_names(base, {"Alice": "Ravi"})
    assert inject_noop("She has 3 pens.", "The pens are red.").endswith("The pens are red.")
    v = auto_variants(base, {"names": {"Alice": "Ravi"}, "noop": "It is Tuesday.", "reorder": False})
    kinds = {x["kind"] for x in v}
    assert "rename" in kinds and "noop" in kinds


def test_summarize():
    recs = [
        {"model": "m", "family_id": "f1", "kind": "baseline", "extracted": "14", "is_correct": True},
        {"model": "m", "family_id": "f1", "kind": "numeric",  "extracted": "22", "is_correct": True},
        {"model": "m", "family_id": "f1", "kind": "noop",     "extracted": "18", "is_correct": False},
    ]
    s = summarize(recs)["m"]
    assert s["baseline_accuracy"] == 1.0
    assert s["accuracy_by_kind"]["noop"] == 0.0
    assert s["degradation_points_by_kind"]["noop"] == 100.0
    # 3 distinct answers over 3 variants -> consistency 1/3
    assert abs(s["answer_consistency"] - 0.333) < 1e-3  # rounded to 3 dp in summary


def test_latent_math():
    import torch
    from src.latent_metrics import effective_rank, participation_ratio, token_entropy
    # Rank-1 data: all variance in one direction -> ER ~ 1, PR ~ 1
    T, d = 50, 8
    direction = torch.randn(d)
    H = torch.outer(torch.randn(T), direction)
    assert effective_rank(H) < 1.5
    assert participation_ratio(H) < 1.5
    # Isotropic data: variance spread across all d dims -> ER and PR ~ d
    H2 = torch.randn(2000, d)
    assert effective_rank(H2) > 0.6 * d
    assert participation_ratio(H2) > 0.6 * d
    # Uniform distribution has the maximum entropy = log(V)
    V = 100
    logits = torch.zeros(1, V)
    assert abs(float(token_entropy(logits)[0]) - math.log(V)) < 1e-4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("\nAll offline tests passed.")
