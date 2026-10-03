"""Controlled response selection: response_templates.json[intent][emotion]."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config, path


class ResponseSelector:
    def __init__(self, templates: dict):
        self.templates = {k: v for k, v in templates.items() if k != "_meta"}

    @classmethod
    def from_file(cls, file: str | Path | None = None) -> "ResponseSelector":
        file = file or path(load_config(), "response_templates")
        return cls(json.loads(Path(file).read_text()))

    def select(self, intent: str, emotion: str | None) -> dict:
        if intent not in self.templates:
            raise KeyError(f"no templates for intent {intent!r}")
        block = self.templates[intent]
        if emotion in block:
            return {"response": block[emotion], "template_key": f"{intent}/{emotion}", "emotion_fallback": False}
        return {"response": block["neutral"], "template_key": f"{intent}/neutral", "emotion_fallback": True}
