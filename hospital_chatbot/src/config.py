"""Configuration loading and project paths."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"

INTENTS = [
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
    "department_information",
    "contact_information",
    "facility_information",
    "admission_query",
    "discharge_query",
    "medical_records",
    "prescription_query",
    "greeting",
    "thank_you",
    "goodbye",
]

# Canonical GoEmotions inventory (27 emotions + neutral). The emotion predictor verifies this against the
# downloaded model's config.id2label at load time; the templates are keyed by label *name*, never by id.
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]

ORIGINAL_EMOTIONS = ["neutral", "joy", "sadness", "fear", "anger", "gratitude"]


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(path or os.environ.get("HOSPITAL_CHATBOT_CONFIG", DEFAULT_CONFIG)) as f:
        return yaml.safe_load(f)


def resolve(rel: str | os.PathLike) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def path(cfg: dict, key: str) -> Path:
    return resolve(cfg["paths"][key])


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
