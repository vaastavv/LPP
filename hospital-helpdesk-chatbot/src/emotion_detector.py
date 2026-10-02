"""Emotion detection with the pretrained GoEmotions model (no training).

``SamLowe/roberta-base-go_emotions`` is a multi-label classifier over the 28
GoEmotions labels (27 emotions + neutral) with independent sigmoid outputs. We
score all labels and collapse them into five helpdesk-relevant groups:

    positive | neutral | sadness | fear | anger

The label -> group mapping and the aggregation strategy live in
``config/emotion_mapping.json`` so they can be tuned without code changes.

Aggregation
-----------
* ``max``  - group score = highest member label score (default; robust to
  groups of different sizes).
* ``sum``  - group score = sum of member label scores (clipped to 1.0).

Because GoEmotions predicts ``neutral`` very often for short service requests,
a non-neutral group whose score is at least ``neutral_override_threshold``
wins over ``neutral``: for an empathetic helpdesk, a missed "fear" is worse
than an unnecessary empathetic sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.utils import EMOTION_GROUPS, EMOTION_MAPPING_JSON, get_logger, load_json

logger = get_logger(__name__)

GOEMOTIONS_LABELS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]


@dataclass
class EmotionConfig:
    model_name: str = "SamLowe/roberta-base-go_emotions"
    groups: dict[str, list[str]] = field(default_factory=dict)
    aggregation: str = "max"
    neutral_override_threshold: float = 0.25
    min_confidence: float = 0.10
    default_group: str = "neutral"

    @classmethod
    def from_json(cls, path: str | Path = EMOTION_MAPPING_JSON) -> "EmotionConfig":
        return cls(**load_json(path))

    def __post_init__(self) -> None:
        if self.aggregation not in {"max", "sum"}:
            raise ValueError(f"aggregation must be 'max' or 'sum', got {self.aggregation!r}")
        unknown_groups = set(self.groups) - set(EMOTION_GROUPS)
        if unknown_groups:
            raise ValueError(f"Unknown emotion groups in mapping: {unknown_groups}")
        mapped = [label for labels in self.groups.values() for label in labels]
        dupes = {label for label in mapped if mapped.count(label) > 1}
        if dupes:
            raise ValueError(f"Labels mapped to more than one group: {dupes}")
        unmapped = set(GOEMOTIONS_LABELS) - set(mapped)
        if unmapped:
            logger.warning("GoEmotions labels not mapped to any group (ignored): %s", sorted(unmapped))

    @property
    def label_to_group(self) -> dict[str, str]:
        return {label: group for group, labels in self.groups.items() for label in labels}


@dataclass
class EmotionResult:
    group: str                       # one of EMOTION_GROUPS
    confidence: float                # score of the winning group
    top_label: str                   # highest scoring raw GoEmotions label
    group_scores: dict[str, float]   # aggregated score per group
    label_scores: dict[str, float]   # raw score for every GoEmotions label


def aggregate_scores(label_scores: dict[str, float], config: EmotionConfig) -> EmotionResult:
    """Pure function: collapse raw GoEmotions scores into a grouped EmotionResult."""
    label_to_group = config.label_to_group
    group_scores = {g: 0.0 for g in EMOTION_GROUPS}
    for label, score in label_scores.items():
        group = label_to_group.get(label)
        if group is None:
            continue
        if config.aggregation == "max":
            group_scores[group] = max(group_scores[group], score)
        else:
            group_scores[group] = min(1.0, group_scores[group] + score)

    best_group = max(group_scores, key=group_scores.get)
    if best_group == "neutral":
        non_neutral = {g: s for g, s in group_scores.items() if g != "neutral"}
        runner_up = max(non_neutral, key=non_neutral.get)
        if non_neutral[runner_up] >= config.neutral_override_threshold:
            best_group = runner_up
    if group_scores[best_group] < config.min_confidence:
        best_group = config.default_group

    top_label = max(label_scores, key=label_scores.get) if label_scores else "neutral"
    return EmotionResult(
        group=best_group,
        confidence=round(group_scores[best_group], 4),
        top_label=top_label,
        group_scores={g: round(s, 4) for g, s in group_scores.items()},
        label_scores={k: round(v, 4) for k, v in sorted(label_scores.items(), key=lambda kv: -kv[1])},
    )


class EmotionDetector:
    """Thin wrapper around the Hugging Face text-classification pipeline."""

    def __init__(self, config: EmotionConfig | None = None, device: int | str | None = None) -> None:
        from transformers import pipeline

        self.config = config or EmotionConfig.from_json()
        if device is None:
            import torch

            device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading emotion model %s", self.config.model_name)
        self._pipe = pipeline(
            "text-classification",
            model=self.config.model_name,
            top_k=None,          # return scores for all 28 labels
            truncation=True,
            device=device,
        )

    def raw_scores(self, texts: list[str]) -> list[dict[str, float]]:
        outputs = self._pipe(texts, batch_size=16)
        return [{item["label"]: float(item["score"]) for item in out} for out in outputs]

    def detect(self, text: str) -> EmotionResult:
        return self.detect_batch([text])[0]

    def detect_batch(self, texts: list[str]) -> list[EmotionResult]:
        if not texts:
            return []
        return [aggregate_scores(scores, self.config) for scores in self.raw_scores(texts)]


@lru_cache(maxsize=1)
def get_emotion_detector() -> EmotionDetector:
    return EmotionDetector()


def detect_emotion(text: str) -> EmotionResult:
    """Convenience function using a process-wide cached detector."""
    return get_emotion_detector().detect(text)


if __name__ == "__main__":
    import sys

    detector = EmotionDetector()
    for sentence in sys.argv[1:] or [
        "My father is having chest pain, please help!",
        "Thank you so much, you were really helpful",
        "This is the third time my bill is wrong, ridiculous",
        "When is the OPD open?",
        "I just lost my mother and need her medical records",
    ]:
        res = detector.detect(sentence)
        print(f"{res.group:8s} ({res.confidence:.2f}, top={res.top_label:14s}) <- {sentence}")
