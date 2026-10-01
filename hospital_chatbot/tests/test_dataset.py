from collections import Counter

import pandas as pd

from src.config import INTENTS, path
from src.data.deduplicate import near_duplicate_pairs
from src.data.validate import validate_dataset
from src.text import dedup_key


def test_schema(dataset):
    assert list(dataset.columns) == ["text", "intent"]


def test_all_20_intents(dataset):
    assert sorted(dataset["intent"].unique()) == sorted(INTENTS)
    assert len(INTENTS) == 20


def test_examples_per_intent(dataset, cfg):
    counts = Counter(dataset["intent"])
    for intent in INTENTS:
        assert cfg["dataset"]["min_per_intent"] <= counts[intent] <= cfg["dataset"]["max_per_intent"], intent


def test_no_exact_or_trivial_duplicates(dataset):
    assert not dataset["text"].duplicated().any()
    assert not dataset["text"].map(dedup_key).duplicated().any()


def test_no_near_duplicates(dataset, cfg):
    pairs = near_duplicate_pairs(dataset["text"].tolist(), cfg["dataset"]["near_duplicate_threshold"])
    # Only pairs where *both* items are protected originals may remain.
    prov = pd.read_csv(path(cfg, "provenance"), keep_default_na=False).set_index("text")["source"]
    texts = dataset["text"].tolist()
    for p in pairs:
        assert prov[texts[p.i]] == "original" and prov[texts[p.j]] == "original", (texts[p.i], texts[p.j])


def test_no_empty_text(dataset):
    assert (dataset["text"].str.strip() != "").all()


def test_validator_passes(dataset, cfg):
    assert validate_dataset(dataset, cfg["dataset"]["min_per_intent"], cfg["dataset"]["max_per_intent"]) == []


def test_validator_catches_problems():
    bad = pd.DataFrame({"text": ["Hi", "hi!", "", "x"], "intent": ["greeting", "greeting", "greeting", "made_up"]})
    errs = " ".join(validate_dataset(bad))
    assert "unknown intents" in errs and "empty text" in errs and "duplicates" in errs and "missing intents" in errs


def test_original_examples_preserved_or_logged(cfg, dataset):
    import json
    raw = pd.read_csv(path(cfg, "raw_dataset"), keep_default_na=False)
    log = json.loads(path(cfg, "changelog").read_text())
    removed = {i["text"] for s in log["steps"] if "removed" in s["step"] for i in s["items"]
               if "text" in i} | {i.get("dropped") for s in log["steps"] for i in s["items"] if isinstance(i, dict)}
    kept = set(dataset["text"])
    for t in raw["text"]:
        assert t in kept or t in removed, f"original example silently dropped: {t}"


def test_splits_disjoint_and_stratified(cfg):
    parts = {s: pd.read_csv(path(cfg, s), keep_default_na=False) for s in ("train", "validation", "test")}
    sets = {k: set(v["text"]) for k, v in parts.items()}
    assert not sets["train"] & sets["test"]
    assert not sets["train"] & sets["validation"]
    assert not sets["validation"] & sets["test"]
    total = sum(len(v) for v in parts.values())
    assert abs(len(parts["train"]) / total - 0.70) < 0.02
    for s in ("validation", "test"):
        assert set(parts[s]["intent"]) == set(INTENTS)


def test_no_near_duplicate_leakage(cfg):
    import json
    summary = json.loads((path(cfg, "train").parent / "split_summary.json").read_text())
    thr = cfg["dataset"]["leakage_group_threshold"]
    for s in ("validation", "test"):
        assert summary["leakage"][s]["max_similarity_to_train_same_intent"] < thr
