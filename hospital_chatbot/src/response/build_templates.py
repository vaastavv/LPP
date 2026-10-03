"""Build data/response_templates.json: 20 intents x 28 GoEmotions labels.

Design: intent decides the *content/action* (BODIES), emotion decides the *tone* (openers). Every
response is opener + body, so meaning stays stable across emotions while tone adapts. Bodies never
state availability, results, coverage, prices, phone numbers or locations - they ask for the details
the hospital workflow needs and point to the team that confirms facts.

emergency_assistance uses a dedicated urgent body (the policy layer also routes it through the
emergency handler before template lookup). greeting / thank_you / goodbye use social openers.

Usage:
    python -m src.response.build_templates              # full matrix -> data/response_templates.json
    python -m src.response.build_templates --seed-only  # 6 original emotions -> data/raw/original_response_templates.json
"""
from __future__ import annotations

import argparse
import json

from src.config import GOEMOTIONS_LABELS, INTENTS, ORIGINAL_EMOTIONS, load_config, path

EMERGENCY_BODY = (
    "If this is a medical emergency, call your local emergency number or go to the nearest emergency "
    "department now. Do not wait for a reply here. If someone is unconscious, not breathing, has severe "
    "chest pain or heavy bleeding, get emergency help immediately and stay with them."
)

BODIES = {
    "appointment_booking": (
        "I can help you request a new appointment. Please share the department or doctor you'd like to see, "
        "the patient's name, and your preferred date and time. Availability will be confirmed by the "
        "appointments team."),
    "appointment_cancellation": (
        "I can help you cancel an existing appointment. Please share the patient's name, the booking "
        "reference or registered phone number, and the appointment date so the booking can be found. Any "
        "fee or refund follows the hospital's cancellation policy, which the appointments team will confirm."),
    "appointment_rescheduling": (
        "I can help you move an existing appointment. Please share the patient's name, the booking reference "
        "or registered phone number, the current appointment date, and your preferred new date and time. "
        "The new slot will be confirmed once availability is checked."),
    "doctor_search": (
        "I can help you find a suitable doctor. Please tell me the speciality or the doctor's name you're "
        "looking for, plus any preference such as day, language or the doctor's gender. If you're unsure "
        "which specialist you need, a general physician or the help desk can guide you."),
    "hospital_timings": (
        "I can help with timings. Please tell me which service you need the hours for, such as the OPD, "
        "visiting hours, the lab, the pharmacy or billing, and the day you plan to come. Timings can differ "
        "by department and on holidays, so please confirm before you travel."),
    "emergency_assistance": EMERGENCY_BODY,
    "billing_query": (
        "I can help with your billing question. Please share the patient's name and hospital or bill number, "
        "and tell me what you need: an estimate, an explanation of charges, a payment, a receipt or a refund. "
        "The billing team will confirm the exact amounts."),
    "insurance_query": (
        "I can help with your insurance question. Please share your insurer or TPA name, your policy number, "
        "and the treatment or admission it relates to. Coverage and claim decisions are made by your insurer; "
        "the hospital's insurance desk can check eligibility and the paperwork needed."),
    "lab_reports": (
        "I can help you with your test report. Please share the patient's name, hospital ID or registered "
        "phone number, the test name and the date it was done. Reports are released once the lab completes "
        "them, and your doctor is the right person to explain the results."),
    "pharmacy_query": (
        "I can help with your pharmacy question. Please tell me the medicine or item you need and whether you "
        "want to collect it or ask about delivery. Stock and price are confirmed by the pharmacy, and "
        "prescription medicines need a valid prescription."),
    "department_information": (
        "I can help with information about our departments. Please tell me which department or service you're "
        "interested in, such as cardiology, radiology or maternity, and what you'd like to know, for example "
        "the services offered or how to find it."),
    "contact_information": (
        "I can help you reach the right team. Please tell me who you'd like to contact, such as reception, "
        "appointments, billing, a department or patient relations. Please use the official contact details "
        "on the hospital's website or your appointment documents."),
    "facility_information": (
        "I can help with information about hospital facilities. Please tell me what you're looking for, such "
        "as parking, food, room types, accessibility, Wi-Fi or a place for family members to stay."),
    "admission_query": (
        "I can help with the admission process. Please share the patient's name, the admitting doctor or "
        "department, and the planned date if you have one. The admission desk will confirm bed availability, "
        "the documents required and any deposit."),
    "discharge_query": (
        "I can help with discharge. Please share the patient's name and ward or hospital ID. Discharge timing "
        "depends on the treating doctor's approval, the discharge summary and bill settlement, and the ward "
        "team will go through the discharge instructions with you."),
    "medical_records": (
        "I can help you request medical records. Please share the patient's name, hospital ID, the visit or "
        "period you need, and the purpose. To protect privacy, records are released only after the medical "
        "records department verifies identity and authorisation."),
    "prescription_query": (
        "I can help with your prescription request. Please share the patient's name, the prescribing doctor "
        "and what you need, such as a renewal, a copy or a correction. Only a doctor can issue or change a "
        "prescription, so any change to a medicine or dose will be reviewed by them."),
    "greeting": (
        "I'm the hospital's virtual assistant. I can help with appointments, finding a doctor, timings, "
        "reports, billing, insurance and more. How can I help you today?"),
    "thank_you": "Is there anything else I can help you with?",
    "goodbye": "If you need anything else, I'm here to help.",
}

# Tone openers for task intents. Must read naturally before any task body.
TASK_OPENERS = {
    "admiration": "Thank you for your kind words.",
    "amusement": "Glad to keep things light.",
    "anger": "I'm sorry this has been frustrating, and I want to help put it right.",
    "annoyance": "Sorry for the hassle. Let's sort this out quickly.",
    "approval": "Great.",
    "caring": "It's kind of you to take care of this.",
    "confusion": "No problem, let's take this step by step.",
    "curiosity": "Good question.",
    "desire": "I understand this matters to you.",
    "disappointment": "I'm sorry things haven't gone the way you hoped.",
    "disapproval": "I understand you're unhappy with how this has gone, and your concern is noted.",
    "disgust": "I'm sorry you've had such an unpleasant experience.",
    "embarrassment": "There's nothing to worry about; this is a very common request.",
    "excitement": "Happy to help!",
    "fear": "I understand this can feel stressful, and I'm here to help.",
    "gratitude": "Thank you, I'm glad to help.",
    "grief": "I'm so sorry for what you're going through. I'll keep this as simple as I can.",
    "joy": "Absolutely! I'd be happy to help.",
    "love": "It's clear you care a lot about this.",
    "nervousness": "Please don't worry; we'll take this one step at a time.",
    "optimism": "That's a positive way to look at it.",
    "pride": "That's great to hear.",
    "realization": "Good point.",
    "relief": "I'm glad things feel a little easier.",
    "remorse": "Please don't worry, these things happen.",
    "sadness": "I'm sorry you're going through a difficult time.",
    "surprise": "I understand this may be unexpected.",
    "neutral": "",
}

SOCIAL_OPENERS = {
    "greeting": {
        "admiration": "Hello, and thank you for the kind words!",
        "amusement": "Hello! Nice to start on a cheerful note.",
        "anger": "Hello. I'm sorry if something has upset you; I'll do my best to help.",
        "annoyance": "Hello. Sorry for any hassle so far; let's get you sorted.",
        "approval": "Hello, great to have you here.",
        "caring": "Hello, and thank you for checking in.",
        "confusion": "Hello! Don't worry if you're not sure where to start.",
        "curiosity": "Hello! Happy to answer your questions.",
        "desire": "Hello! Let's find what you need.",
        "disappointment": "Hello. I'm sorry if things haven't gone well so far.",
        "disapproval": "Hello. I hear that you're not happy, and I'd like to help.",
        "disgust": "Hello. I'm sorry if you've had an unpleasant experience.",
        "embarrassment": "Hello! There's no such thing as a silly question here.",
        "excitement": "Hello! Lovely to hear from you.",
        "fear": "Hello. I'm here to help, and we can go at your pace.",
        "gratitude": "Hello, and thank you for reaching out!",
        "grief": "Hello. I'm sorry if you're going through a hard time.",
        "joy": "Hello! It's great to hear from you.",
        "love": "Hello! Warm wishes to you.",
        "nervousness": "Hello. Please don't worry; I'm here to help.",
        "optimism": "Hello! Let's get things moving.",
        "pride": "Hello! Good to hear from you.",
        "realization": "Hello! Good to see you here.",
        "relief": "Hello! Glad you found us.",
        "remorse": "Hello! No need to apologise.",
        "sadness": "Hello. I'm sorry if today has been difficult.",
        "surprise": "Hello! Yes, you've reached the hospital assistant.",
        "neutral": "Hello!",
    },
    "thank_you": {
        "admiration": "Thank you, that's very kind of you to say.",
        "amusement": "Ha, glad I could help!",
        "anger": "You're welcome, and I'm sorry if anything was frustrating today.",
        "annoyance": "You're welcome, and sorry for any hassle along the way.",
        "approval": "Glad that worked for you.",
        "caring": "You're very welcome; take good care.",
        "confusion": "You're welcome. If anything is still unclear, just ask.",
        "curiosity": "You're welcome, and feel free to ask more questions.",
        "desire": "You're welcome; I hope you get what you need.",
        "disappointment": "You're welcome, and I'm sorry it wasn't everything you hoped for.",
        "disapproval": "Thank you, and your feedback is noted.",
        "disgust": "You're welcome, and I'm sorry about the unpleasant experience.",
        "embarrassment": "No trouble at all, you're welcome.",
        "excitement": "You're so welcome!",
        "fear": "You're welcome. I hope things feel a little less worrying now.",
        "gratitude": "You're very welcome!",
        "grief": "You're welcome. I'm so sorry for what you're going through.",
        "joy": "You're welcome, I'm really glad I could help!",
        "love": "That's lovely to hear, you're very welcome.",
        "nervousness": "You're welcome. Please don't hesitate to come back if you're unsure about anything.",
        "optimism": "You're welcome, and I hope everything goes smoothly.",
        "pride": "You're welcome, and well done for getting this sorted.",
        "realization": "You're welcome, glad that cleared things up.",
        "relief": "You're welcome, I'm glad that's a relief.",
        "remorse": "You're welcome, and no need to apologise.",
        "sadness": "You're welcome. I'm sorry things are hard right now.",
        "surprise": "You're welcome!",
        "neutral": "You're welcome.",
    },
    "goodbye": {
        "admiration": "Thank you for the kind words. Goodbye!",
        "amusement": "Bye for now, glad we kept it cheerful!",
        "anger": "Goodbye, and I'm sorry if anything frustrated you today.",
        "annoyance": "Goodbye, and sorry for any hassle.",
        "approval": "Great, goodbye for now.",
        "caring": "Goodbye, and please look after yourself.",
        "confusion": "Goodbye. If anything is still unclear later, just message again.",
        "curiosity": "Goodbye! Come back any time with more questions.",
        "desire": "Goodbye, I hope you get everything you need.",
        "disappointment": "Goodbye, and I'm sorry if this wasn't as helpful as you hoped.",
        "disapproval": "Goodbye. Your feedback is noted.",
        "disgust": "Goodbye, and I'm sorry about the unpleasant experience.",
        "embarrassment": "Goodbye, and no worries at all.",
        "excitement": "Bye! Have a great day!",
        "fear": "Goodbye. Take care, and seek help promptly if anything worries you.",
        "gratitude": "You're welcome, goodbye!",
        "grief": "Goodbye. I'm so sorry for your loss, and please take care.",
        "joy": "Goodbye, it was a pleasure helping you!",
        "love": "Goodbye, and warm wishes to you and your family.",
        "nervousness": "Goodbye. Please don't worry, and reach out any time.",
        "optimism": "Goodbye, and I hope all goes well.",
        "pride": "Goodbye, and well done for getting everything sorted.",
        "realization": "Goodbye, glad things are clearer now.",
        "relief": "Goodbye, I'm glad things feel better.",
        "remorse": "Goodbye, and please don't worry about it.",
        "sadness": "Goodbye. I'm sorry things are difficult; please take care.",
        "surprise": "Goodbye for now!",
        "neutral": "Goodbye, and take care.",
    },
}

# Emergency: calm, short, never playful. One distinct prefix per emotion.
EMERGENCY_OPENERS = {
    "admiration": "Thank you for reaching out. This needs urgent attention.",
    "amusement": "Please treat this seriously.",
    "anger": "I understand, and getting you help fast is what matters now.",
    "annoyance": "I'll keep this brief so you can act quickly.",
    "approval": "Acting quickly is the right call.",
    "caring": "You're doing the right thing by getting help.",
    "confusion": "Here is what to do right now.",
    "curiosity": "If you're unsure whether this is an emergency, treat it as one.",
    "desire": "Getting urgent help is the priority.",
    "disappointment": "Please don't wait any longer.",
    "disapproval": "Your safety comes first.",
    "disgust": "Please get help immediately.",
    "embarrassment": "Never hesitate to ask for emergency help.",
    "excitement": "This needs urgent action.",
    "fear": "I understand you're frightened. Please act now.",
    "gratitude": "Thank you for telling me. Please act now.",
    "grief": "I'm so sorry this is happening.",
    "joy": "Please get help right away.",
    "love": "Getting them help quickly is the most caring thing you can do.",
    "nervousness": "Try to stay as calm as you can.",
    "optimism": "Quick action can make a real difference.",
    "pride": "You've done well to reach out.",
    "realization": "If this is serious, act now.",
    "relief": "Even if things seem to be easing, don't take chances.",
    "remorse": "Don't blame yourself; focus on getting help now.",
    "sadness": "I'm sorry you're dealing with this.",
    "surprise": "Sudden symptoms need urgent attention.",
    "neutral": "This sounds urgent.",
}

SOCIAL = {"greeting", "thank_you", "goodbye"}


def build(emotions: list[str]) -> dict:
    out: dict = {}
    for intent in INTENTS:
        out[intent] = {}
        for emo in emotions:
            if intent == "emergency_assistance":
                text = f"{EMERGENCY_OPENERS[emo]} {EMERGENCY_BODY}"
            elif intent in SOCIAL:
                text = f"{SOCIAL_OPENERS[intent][emo]} {BODIES[intent]}"
            else:
                opener = TASK_OPENERS[emo]
                text = f"{opener} {BODIES[intent]}" if opener else BODIES[intent]
            out[intent][emo] = text.strip()
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.seed_only:
        target, emotions = path(cfg, "raw_response_templates"), ORIGINAL_EMOTIONS
    else:
        target, emotions = path(cfg, "response_templates"), GOEMOTIONS_LABELS
    templates = build(emotions)
    target.write_text(json.dumps(templates, indent=2, ensure_ascii=False) + "\n")
    n = sum(len(v) for v in templates.values())
    print(f"wrote {target}: {len(templates)} intents x {len(emotions)} emotions = {n} templates")


if __name__ == "__main__":
    main()
