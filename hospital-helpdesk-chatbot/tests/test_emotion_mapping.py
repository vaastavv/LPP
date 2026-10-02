import pytest

from src.emotion_detector import GOEMOTIONS_LABELS, EmotionConfig, aggregate_scores


@pytest.fixture(scope="module")
def config() -> EmotionConfig:
    return EmotionConfig.from_json()


def scores(**overrides: float) -> dict[str, float]:
    base = {label: 0.01 for label in GOEMOTIONS_LABELS}
    base.update(overrides)
    return base


def test_every_goemotions_label_is_mapped(config):
    assert set(config.label_to_group) == set(GOEMOTIONS_LABELS)


def test_fear_maps_to_fear(config):
    assert aggregate_scores(scores(fear=0.8, nervousness=0.4), config).group == "fear"


def test_gratitude_maps_to_positive(config):
    assert aggregate_scores(scores(gratitude=0.95), config).group == "positive"


def test_annoyance_maps_to_anger(config):
    assert aggregate_scores(scores(annoyance=0.6, neutral=0.2), config).group == "anger"


def test_neutral_override(config):
    # neutral wins on raw score, but a strong-enough negative signal takes priority
    res = aggregate_scores(scores(neutral=0.7, sadness=0.35), config)
    assert res.group == "sadness"
    res = aggregate_scores(scores(neutral=0.7, sadness=0.05), config)
    assert res.group == "neutral"


def test_sum_aggregation():
    cfg = EmotionConfig(**{**EmotionConfig.from_json().__dict__, "aggregation": "sum"})
    res = aggregate_scores(scores(neutral=0.5, annoyance=0.3, anger=0.3), cfg)
    assert res.group == "anger"
    assert res.group_scores["anger"] == pytest.approx(0.3 + 0.3 + 0.01 + 0.01)


def test_invalid_mapping_rejected():
    with pytest.raises(ValueError):
        EmotionConfig(groups={"joyful": ["joy"]})
    with pytest.raises(ValueError):
        EmotionConfig(groups={"positive": ["joy"], "neutral": ["joy"]})
