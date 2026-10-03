import pytest

from conftest import emotion_model_available
from src.config import GOEMOTIONS_LABELS
from src.emotion.predictor import EmotionClassifier


def test_label_set_read_from_model_config(tiny_emotion_dir):
    clf = EmotionClassifier(tiny_emotion_dir, min_confidence=0.5)
    assert clf.labels == GOEMOTIONS_LABELS and clf.multi_label


def test_prediction_shape_and_neutral_fallback(tiny_emotion_dir):
    clf = EmotionClassifier(tiny_emotion_dir, min_confidence=1.01)  # impossible -> always fallback
    r = clf.predict("I'm scared")
    assert r["emotion"] == "neutral" and r["status"] == "fallback_neutral"
    assert r["raw_emotion"] in GOEMOTIONS_LABELS and 0 <= r["confidence"] <= 1


def test_label_mismatch_rejected(tmp_path, tiny_emotion_dir):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(tiny_emotion_dir)
    m.config.id2label = {**m.config.id2label, 0: "boredom"}
    m.save_pretrained(tmp_path)
    for f in tiny_emotion_dir.iterdir():
        if "token" in f.name or f.suffix in (".txt",):
            (tmp_path / f.name).write_bytes(f.read_bytes())
    with pytest.raises(ValueError):
        EmotionClassifier(tmp_path)


@pytest.mark.skipif("not emotion_model_available(__import__('src.config').config.load_config())",
                    reason="pretrained GoEmotions checkpoint not downloaded to models/emotion")
def test_real_emotion_model(cfg):
    clf = EmotionClassifier.from_config(cfg)
    assert sorted(clf.labels) == sorted(GOEMOTIONS_LABELS)
    r = clf.predict("Thank you so much, you have been incredibly helpful!")
    assert r["emotion"] in ("gratitude", "admiration", "joy")
