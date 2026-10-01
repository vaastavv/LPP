"""End-to-end chatbot: validation -> normalization -> intent || emotion -> policy -> response.

    bot = HospitalChatbot.load()
    bot.respond("Can I see a heart specialist tomorrow?")

CLI: python -m src.pipeline "your message"   (or no args for an interactive prompt)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

from src.config import load_config
from src.intent.predictor import BaseIntentClassifier, load_intent_classifier
from src.policy.router import decide
from src.response.selector import ResponseSelector
from src.text import validate_input


class HospitalChatbot:
    def __init__(self, intent_classifier: BaseIntentClassifier, emotion_classifier, selector: ResponseSelector):
        self.intent = intent_classifier
        self.emotion = emotion_classifier
        self.selector = selector

    @classmethod
    def load(cls, intent_kind: str | None = None, cfg: dict | None = None) -> "HospitalChatbot":
        from src.emotion.predictor import EmotionClassifier
        cfg = cfg or load_config()
        return cls(load_intent_classifier(intent_kind, cfg), EmotionClassifier.from_config(cfg),
                   ResponseSelector.from_file())

    def respond(self, text: str) -> dict:
        v = validate_input(text)
        if not v.ok:
            d = decide(text if isinstance(text, str) else "", None, None, self.selector,
                       self.intent.emergency_threshold, valid=False)
            return {"input": text, "valid": False, "validation_error": v.reason, "intent": None,
                    "emotion": None, "route": d.route, "response": d.response, "reason": d.reason}
        intent_result = self.intent.predict(v.text)       # the two models are independent
        emotion_result = self.emotion.predict(v.text)
        d = decide(v.text, intent_result, emotion_result, self.selector, self.intent.emergency_threshold)
        return {"input": v.text, "valid": True, "intent": intent_result, "emotion": emotion_result,
                **{k: val for k, val in asdict(d).items() if k != "intent"}, "final_intent": d.intent}


def main() -> None:
    bot = HospitalChatbot.load()
    msgs = sys.argv[1:]
    if msgs:
        for m in msgs:
            print(json.dumps(bot.respond(m), indent=2))
        return
    print("Hospital assistant (Ctrl-D to quit)")
    for line in sys.stdin:
        r = bot.respond(line)
        print(f"[{r['route']}] intent={r['final_intent']} emotion={(r['emotion'] or {}).get('emotion')}\n> {r['response']}")


if __name__ == "__main__":
    main()
