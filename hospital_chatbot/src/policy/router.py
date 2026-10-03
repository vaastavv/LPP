"""Policy layer: decides how the system behaves given intent + emotion predictions.

Priority (first match wins):
  1. invalid input           -> ask the user to type their question
  2. self-harm language      -> crisis handler
  3. emergency               -> emergency handler (model P(emergency) >= emergency threshold OR red flag)
  4. confident intent        -> normal intent response (intent x emotion template)
  5. otherwise               -> clarification / unknown handler
"""
from __future__ import annotations

from dataclasses import dataclass

from src.policy import emergency as em
from src.response.selector import ResponseSelector

INTENT_DESCRIPTIONS = {
    "appointment_booking": "booking a new appointment",
    "appointment_cancellation": "cancelling an appointment",
    "appointment_rescheduling": "changing an existing appointment",
    "doctor_search": "finding a doctor or specialist",
    "hospital_timings": "hospital or service timings",
    "emergency_assistance": "emergency help",
    "billing_query": "bills and payments",
    "insurance_query": "insurance and claims",
    "lab_reports": "test reports",
    "pharmacy_query": "the pharmacy",
    "department_information": "hospital departments",
    "contact_information": "contacting the hospital",
    "facility_information": "hospital facilities",
    "admission_query": "admission",
    "discharge_query": "discharge",
    "medical_records": "medical records",
    "prescription_query": "prescriptions",
    "greeting": "a greeting",
    "thank_you": "saying thanks",
    "goodbye": "ending the chat",
}

INVALID_INPUT_RESPONSE = "Please type your question, for example: \"I'd like to book an appointment.\""
UNKNOWN_RESPONSE = (
    "I'm sorry, I'm not sure I understood. I can help with appointments, finding a doctor, hospital "
    "timings, test reports, billing, insurance, admission, discharge, prescriptions and medical records. "
    "Could you tell me a bit more about what you need?"
)
SOCIAL = {"greeting", "thank_you", "goodbye"}


@dataclass
class Decision:
    route: str            # invalid_input | crisis | emergency | intent | clarification
    intent: str | None
    response: str
    reason: str
    template_key: str | None = None


def clarification(intent_result: dict) -> str:
    cands = [c["intent"] for c in intent_result.get("top_k", [])[:2]
             if c["intent"] not in SOCIAL and c["intent"] != "emergency_assistance"]
    if len(cands) == 2:
        a, b = (INTENT_DESCRIPTIONS[c] for c in cands)
        return (f"I want to make sure I help with the right thing. Is your question about {a} or {b}? "
                "If it's something else, please tell me a little more.")
    return UNKNOWN_RESPONSE


def decide(text: str, intent_result: dict | None, emotion_result: dict | None, selector: ResponseSelector,
           emergency_threshold: float, valid: bool = True) -> Decision:
    if not valid or intent_result is None:
        return Decision("invalid_input", None, INVALID_INPUT_RESPONSE, "input failed validation")
    emotion = (emotion_result or {}).get("emotion", "neutral")
    if em.self_harm(text):
        return Decision("crisis", "emergency_assistance", em.CRISIS_RESPONSE, "self-harm language detected")
    flags = em.red_flags(text)
    p_em = intent_result.get("emergency_probability", 0.0)
    if p_em >= emergency_threshold or flags:
        reason = (f"P(emergency)={p_em:.2f} >= {emergency_threshold:.2f}" if p_em >= emergency_threshold
                  else f"red-flag phrase: {flags[0]}")
        return Decision("emergency", "emergency_assistance",
                        em.emergency_response(selector.templates, emotion), reason,
                        f"emergency_assistance/{emotion}")
    if intent_result["status"] == "confident":
        sel = selector.select(intent_result["intent"], emotion)
        return Decision("intent", intent_result["intent"], sel["response"],
                        f"confidence {intent_result['confidence']:.2f} >= threshold", sel["template_key"])
    return Decision("clarification", None, clarification(intent_result),
                    f"confidence {intent_result['confidence']:.2f} below threshold")
