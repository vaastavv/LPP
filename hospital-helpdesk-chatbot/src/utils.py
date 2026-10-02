"""Shared utilities: project paths, logging, seeding, JSON helpers, intent labels."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

INTENTS_CSV = DATA_DIR / "intents.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
RESPONSE_TEMPLATES_JSON = DATA_DIR / "response_templates.json"
EMOTION_MAPPING_JSON = CONFIG_DIR / "emotion_mapping.json"
HOSPITAL_INFO_JSON = CONFIG_DIR / "hospital_info.json"
INTENT_MODEL_DIR = MODELS_DIR / "intent_classifier"

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
INTENTS: list[str] = [
    "appointment_booking",
    "appointment_cancellation",
    "appointment_rescheduling",
    "doctor_search",
    "hospital_timings",
    "emergency_assistance",
    "billing_query",
    "insurance_query",
    "lab_reports",
    "pharmacy_query",
    "contact_information",
    "greetings",
    "thank_you",
    "goodbye",
]

EMOTION_GROUPS: list[str] = ["positive", "neutral", "sadness", "fear", "anger"]

# Phrases that must always be routed to emergency handling, regardless of what the
# classifier predicts. A missed emergency is far more costly than a false alarm.
EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "chest pain",
    "heart attack",
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "not breathing",
    "stopped breathing",
    "unconscious",
    "fainted",
    "collapsed",
    "seizure",
    "stroke",
    "bleeding heavily",
    "heavy bleeding",
    "severe bleeding",
    "overdose",
    "poisoning",
    "suicide",
    "kill myself",
    "severe burn",
    "choking",
    "ambulance",
)


# --------------------------------------------------------------------------- #
# Logging / reproducibility
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """Return a module logger with a consistent format (configured once)."""
    level = level or os.getenv("LOG_LEVEL", "INFO")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if available) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")


def label_maps(labels: list[str] | None = None) -> tuple[dict[str, int], dict[int, str]]:
    """Return (label2id, id2label) for the given labels (default: INTENTS)."""
    labels = labels or INTENTS
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Light normalisation used before inference and for duplicate detection.

    Keeps punctuation and casing information out of the dedup key but does *not*
    correct spelling: the model is trained on noisy text on purpose.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _WS_RE.sub(" ", text).strip()
    return text


def dedup_key(text: str) -> str:
    """Case/punctuation-insensitive key for duplicate detection."""
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return _WS_RE.sub(" ", text).strip()


def contains_emergency_keyword(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(kw in lowered for kw in EMERGENCY_KEYWORDS)


class SafeFormatDict(dict):
    """dict for str.format_map that leaves unknown placeholders untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
