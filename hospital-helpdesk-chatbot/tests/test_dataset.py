import pandas as pd
import pytest

from scripts.generate_dataset import TEMPLATES, generate_dataset
from scripts.validate_dataset import check_missing_values, remove_duplicates, stratified_split, check_leakage
from src.utils import INTENTS, dedup_key


def test_templates_cover_all_intents():
    assert set(TEMPLATES) == set(INTENTS)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(per_intent=150, seed=7)


def test_generated_dataset_is_balanced_and_unique(dataset):
    counts = dataset["intent"].value_counts()
    assert set(counts.index) == set(INTENTS)
    assert (counts == 150).all()
    assert dataset["text"].map(dedup_key).is_unique


def test_generation_is_deterministic():
    a = generate_dataset(per_intent=20, seed=1)
    b = generate_dataset(per_intent=20, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_drops_missing_and_duplicates():
    df = pd.DataFrame(
        {
            "text": ["Book a doctor", "book a doctor!", None, "Hi", "hi", "  Bye  "],
            "intent": ["appointment_booking", "appointment_booking", "greetings", "greetings", "goodbye", "goodbye"],
        }
    )
    df, info = check_missing_values(df)
    assert info["rows_dropped_missing"] == 1
    df, info = remove_duplicates(df)
    # "Hi"/"hi" carry conflicting labels -> both dropped; "Book a doctor" near-dup removed
    assert info["label_conflict_rows"] == 2
    assert sorted(df["text"]) == ["Book a doctor", "Bye"]


def test_stratified_split_has_no_leakage(dataset):
    train, val, test = stratified_split(dataset, 0.15, 0.15, seed=0)
    assert len(train) + len(val) + len(test) == len(dataset)
    assert set(val["intent"]) == set(INTENTS) == set(test["intent"])
    assert check_leakage(train, val, test) == {"train_val": 0, "train_test": 0, "val_test": 0}
