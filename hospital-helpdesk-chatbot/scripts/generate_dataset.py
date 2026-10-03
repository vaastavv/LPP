"""Generate the synthetic intent dataset ``data/intents.csv`` (text,intent).

The generator combines hand-written patient queries (with slots such as
``{specialty}`` or ``{day}``) with *style transforms* that mimic how real users
write to a hospital helpdesk:

* formal and informal language
* short and long queries
* spelling mistakes / keyboard typos
* urgent requests
* elderly speaking style
* young-user / texting style

Every intent yields exactly ``--per-intent`` unique examples (default 150).

Usage::

    python -m scripts.generate_dataset --per-intent 150 --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
from collections.abc import Callable

import pandas as pd

from src.utils import INTENTS, INTENTS_CSV, dedup_key, get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Slot values
# --------------------------------------------------------------------------- #
SLOTS: dict[str, list[str]] = {
    "specialty": [
        "cardiologist", "dermatologist", "orthopedic", "neurologist", "pediatrician",
        "gynecologist", "ENT specialist", "eye specialist", "dentist", "psychiatrist",
        "general physician", "diabetologist", "urologist", "oncologist",
        "gastroenterologist", "pulmonologist", "physiotherapist", "kidney specialist",
    ],
    "doctor": [
        "Dr. Sharma", "Dr. Mehta", "Dr. Iyer", "Dr. Khan", "Dr. Gupta", "Dr. Reddy",
        "Dr. Fernandes", "Dr. Singh", "Dr. Banerjee", "Dr. Nair", "Dr. Patel",
        "Dr. Rao", "Dr. Thomas", "Dr. Kapoor",
    ],
    "day": [
        "today", "tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "next week", "this weekend", "the 15th", "day after tomorrow",
    ],
    "time": [
        "morning", "evening", "10 am", "11:30", "afternoon", "after 5 pm",
        "early morning", "2 pm", "lunch time", "9 o'clock",
    ],
    "test": [
        "blood test", "CBC", "thyroid test", "MRI", "CT scan", "X-ray", "ultrasound",
        "urine test", "lipid profile", "sugar test", "HbA1c", "ECG", "liver function test",
        "covid test", "biopsy", "vitamin D test",
    ],
    "medicine": [
        "paracetamol", "insulin", "metformin", "my BP tablets", "amoxicillin",
        "inhaler", "cough syrup", "eye drops", "thyroxine", "the prescribed antibiotics",
        "pain killers", "my heart medicines", "vitamin supplements",
    ],
    "insurer": [
        "Star Health", "HDFC Ergo", "ICICI Lombard", "Niva Bupa", "CGHS", "ECHS",
        "Ayushman Bharat", "my company insurance", "my mediclaim", "Care Health",
        "New India Assurance", "my TPA",
    ],
    "relation": [
        "my mother", "my father", "my wife", "my husband", "my son", "my daughter",
        "my grandmother", "my grandfather", "my brother", "my sister", "my baby",
        "my friend", "my neighbour",
    ],
    "department": [
        "OPD", "cardiology", "radiology", "pathology lab", "emergency ward",
        "pediatrics", "orthopedics", "billing department", "pharmacy", "ICU",
        "maternity ward", "physiotherapy",
    ],
    "symptom": [
        "fever", "back pain", "skin rash", "knee pain", "headache", "stomach ache",
        "cough", "high sugar", "chest congestion", "ear pain", "toothache", "anxiety",
    ],
}

# --------------------------------------------------------------------------- #
# Base templates (one list per intent)
# --------------------------------------------------------------------------- #
TEMPLATES: dict[str, list[str]] = {
    "appointment_booking": [
        "I want to book an appointment with a {specialty}",
        "Book an appointment with {doctor} for {day}",
        "Can I get an appointment for {day} {time}?",
        "I need to see a {specialty} {day}",
        "How do I book a consultation with {doctor}?",
        "Please schedule a visit with the {specialty} for {relation}",
        "I would like to make an appointment for {relation} with {doctor}",
        "Is there any slot available with {doctor} {day}?",
        "need appointment for {symptom}",
        "I have {symptom}, can I book a doctor visit {day}?",
        "Can you fix an appointment in {department} for {day} {time}?",
        "Kindly book an OPD appointment with the {specialty}",
        "I want to consult a {specialty} regarding {symptom}",
        "Are appointments open for {doctor} this week?",
        "Please reserve a slot for me with {doctor} on {day} {time}",
        "I'd like to register for a new appointment",
        "Can I book online consultation with a {specialty}?",
        "First time patient here, how do I get an appointment with {doctor}?",
        "Book me in with the {specialty} at the earliest",
        "I need a follow up appointment with {doctor}",
        "Get me an appointment for {relation}, she has {symptom}",
        "Is it possible to see {doctor} {day} without waiting too long? I want to book",
        "want to meet {doctor} {day} {time}",
        "Appointment booking for {specialty} please",
        "I require an appointment for a general health checkup on {day}",
        "Could you help me schedule a video consultation with {doctor}?",
        "My {symptom} is not going away, I want to book a doctor",
        "Please confirm a booking with {doctor} for {relation}",
    ],
    "appointment_cancellation": [
        "I want to cancel my appointment with {doctor}",
        "Please cancel my booking for {day}",
        "Cancel the appointment I booked with the {specialty}",
        "I won't be able to come {day}, cancel my appointment",
        "How can I cancel my appointment?",
        "Kindly cancel {relation}'s appointment with {doctor}",
        "I need to call off my visit on {day} {time}",
        "Remove my appointment for {day}",
        "I don't need the appointment anymore, please cancel it",
        "Cancel my {test} appointment scheduled for {day}",
        "Can I cancel my consultation and get a refund?",
        "Please drop my slot with {doctor}, I am feeling better now",
        "I booked by mistake, cancel it",
        "Withdraw my appointment for {day} please",
        "I want to cancel the online consultation with {doctor}",
        "Something came up, I have to cancel my {day} appointment",
        "Is there a cancellation fee if I cancel my appointment with {doctor}?",
        "Please cancel all my upcoming appointments",
        "cancel appointment",
        "We are travelling, cancel {relation}'s visit with the {specialty}",
        "Delete my booking with {doctor} on {day}",
        "I would like to cancel my scheduled visit in {department}",
        "My appointment on {day} {time} is not needed, cancel it",
        "Can you cancel the follow up with {doctor}?",
    ],
    "appointment_rescheduling": [
        "I want to reschedule my appointment with {doctor}",
        "Can I move my appointment from {day} to another day?",
        "Please change my appointment to {day} {time}",
        "Shift my booking with the {specialty} to {day}",
        "I can't make it {day}, can we reschedule?",
        "Postpone my appointment with {doctor} to {day}",
        "Is it possible to change the time of my appointment to {time}?",
        "Kindly reschedule {relation}'s visit with {doctor}",
        "I need a different slot for my {test}",
        "Move my consultation to {time} please",
        "Can I prepone my appointment to {day}?",
        "How do I reschedule an appointment?",
        "Change appointment date please",
        "Please give me another date for my appointment with {doctor}",
        "I am stuck at work, can my {day} appointment be shifted to {time}?",
        "Can you push my appointment in {department} to {day}?",
        "Rebook my visit with {doctor} for some other day",
        "Need to change my slot from {time} to evening",
        "Could my follow-up with the {specialty} be moved to {day}?",
        "Swap my appointment to {day} {time} if available",
        "I want to change the doctor and date for my appointment",
        "Reschedule my {test} to {day} please",
        "Can I come {day} instead of the booked date?",
    ],
    "doctor_search": [
        "Which {specialty} is available {day}?",
        "Is {doctor} available {day}?",
        "Who is the best {specialty} in your hospital?",
        "I am looking for a {specialty}",
        "Find me a doctor for {symptom}",
        "Which doctor should I see for {symptom}?",
        "Does {doctor} work on {day}?",
        "Tell me about {doctor}'s qualifications and experience",
        "List of {specialty} doctors in the hospital",
        "Do you have a female {specialty}?",
        "What are {doctor}'s OPD days?",
        "Which doctor treats {symptom} in {relation}?",
        "Is there a senior {specialty} who speaks Hindi?",
        "What is the consultation fee of {doctor}?",
        "Which department handles {symptom}?",
        "Is {doctor} still practicing at your hospital?",
        "Who is the head of {department}?",
        "Suggest a good {specialty} for {relation}",
        "doctor for {symptom}?",
        "When does {doctor} sit in the clinic?",
        "Do you have any {specialty} visiting {day} {time}?",
        "I want details of doctors in {department}",
        "Which {specialty} has the earliest availability?",
    ],
    "hospital_timings": [
        "What are the hospital timings?",
        "What time does the OPD open?",
        "Is the hospital open on {day}?",
        "What are the visiting hours?",
        "When does the {department} close?",
        "Till what time is the {department} open?",
        "Are you open on Sundays and public holidays?",
        "What are the timings of {department}?",
        "When can I visit {relation} in the ward?",
        "What time does the lab start sample collection?",
        "Is OPD open in the {time}?",
        "Opening hours please",
        "What time can I come for my {test}?",
        "When do the doctors start seeing patients in the morning?",
        "Is the hospital open 24 hours?",
        "What are your working hours on {day}?",
        "At what time does registration counter open?",
        "What are ICU visiting hours?",
        "Can relatives visit at night?",
        "When does the billing counter close?",
        "Is the hospital open during Diwali?",
        "timings of {department} on {day}?",
        "What time does the evening OPD start?",
    ],
    "emergency_assistance": [
        "{relation} is having severe chest pain, please help",
        "I need an ambulance immediately",
        "Emergency! {relation} has collapsed",
        "Someone is unconscious, what should I do?",
        "{relation} can't breathe properly, need help now",
        "There has been an accident, we need emergency help",
        "Send an ambulance to my house quickly",
        "{relation} is bleeding heavily",
        "I think {relation} is having a heart attack",
        "My child swallowed something and is choking",
        "{relation} is having a seizure right now",
        "Help, {relation} fell down the stairs and is not responding",
        "Where is the emergency ward?",
        "Is the emergency department open now?",
        "{relation} has very high fever and is shivering badly, it's an emergency",
        "Somebody took too many pills, overdose, please help",
        "I have severe breathing difficulty",
        "Snake bite emergency, what to do?",
        "{relation} has stroke symptoms, face drooping and slurred speech",
        "Urgent help needed, pregnant woman in labour pain",
        "Burn injury from hot oil, need emergency treatment",
        "What is the ambulance number?",
        "{relation} fainted and is not waking up",
        "My sugar level dropped very low and I feel dizzy, need help",
        "Road accident near the hospital, need trauma care",
        "Emergency contact number please, it's urgent",
        "I am having a panic attack and can't breathe",
    ],
    "billing_query": [
        "How much will the {test} cost?",
        "I want a copy of my hospital bill",
        "Why is my bill so high?",
        "Can I pay the bill online?",
        "What are the payment options for the bill?",
        "I was charged twice for my consultation",
        "How do I get a refund for {relation}'s cancelled procedure?",
        "Can I get an itemized bill for my stay?",
        "What is the cost of a room per day?",
        "Do you accept credit cards and UPI?",
        "I need the final bill for discharge",
        "What are the consultation charges for a {specialty}?",
        "Is there an EMI option for surgery payment?",
        "My bill has a wrong charge for {test}",
        "Where is the billing counter?",
        "How much is the deposit for admission?",
        "I paid but didn't receive the receipt",
        "Can I get a GST invoice for the treatment?",
        "What is the package cost for a normal delivery?",
        "Need the bill breakdown for {relation}'s treatment",
        "When will I get my refund?",
        "price of {test}?",
        "Are there any discounts for senior citizens on bills?",
        "How much does a health checkup package cost?",
    ],
    "insurance_query": [
        "Do you accept {insurer}?",
        "Is cashless treatment available with {insurer}?",
        "How do I claim insurance for my hospitalisation?",
        "Is my treatment covered under {insurer}?",
        "What documents are needed for an insurance claim?",
        "My insurance claim was rejected, what should I do?",
        "Is the hospital empanelled with {insurer}?",
        "How long does cashless approval take?",
        "Can I get pre-authorization for surgery under {insurer}?",
        "Will my insurance cover {relation}'s delivery?",
        "Where is the TPA desk?",
        "Does {insurer} cover the cost of {test}?",
        "I want reimbursement documents for my insurance company",
        "Insurance claim status please",
        "Is OPD consultation covered by {insurer}?",
        "My policy is with {insurer}, can I get cashless admission?",
        "What is the process for reimbursement claims?",
        "Do you take government health schemes like CGHS?",
        "Can I use two insurance policies for one bill?",
        "The insurance company is asking for discharge summary, how do I get it?",
        "What part of the bill will insurance not cover?",
        "Is my health card valid at your hospital?",
        "insurance accepted here?",
    ],
    "lab_reports": [
        "When will my {test} report be ready?",
        "How can I download my lab reports?",
        "I want my {test} results",
        "Can you send my reports by email?",
        "My {test} report is delayed",
        "Where do I collect my X-ray films and reports?",
        "Is {relation}'s {test} report available?",
        "Can I see my reports on the patient portal?",
        "I have not received my report yet",
        "What do my {test} results mean?",
        "Please share my blood report on WhatsApp",
        "How long does the {test} report take?",
        "I lost my lab report, can I get a duplicate?",
        "Can the doctor review my {test} report online?",
        "Report status for sample given {day}",
        "Are my test results normal?",
        "How do I access old reports from last year?",
        "Need {test} report urgently for my doctor",
        "Did my biopsy result come?",
        "{test} report ready?",
        "Can someone else collect my reports on my behalf?",
        "Where is the report collection counter?",
        "I got an SMS that my report is ready, how do I view it?",
    ],
    "pharmacy_query": [
        "Is {medicine} available in your pharmacy?",
        "Is the pharmacy open now?",
        "Can I order {medicine} online from the hospital pharmacy?",
        "Do you deliver medicines at home?",
        "I need to refill my prescription for {medicine}",
        "What is the price of {medicine}?",
        "Is there a generic alternative for {medicine}?",
        "Where is the pharmacy located?",
        "Can I buy {medicine} without a prescription?",
        "Is the pharmacy open 24 hours?",
        "I want to return unused medicines",
        "Do you have {medicine} in stock?",
        "Can I get a discount on medicines?",
        "My doctor prescribed {medicine}, where can I get it?",
        "Pharmacy timings on {day}?",
        "Can I pick up {relation}'s medicines?",
        "Is there a chemist inside the hospital?",
        "need {medicine}",
        "The pharmacy gave me the wrong medicine",
        "Can I send my prescription on WhatsApp to the pharmacy?",
        "Do you stock imported medicines?",
        "How should I take {medicine}, can the pharmacist explain?",
    ],
    "contact_information": [
        "What is the hospital phone number?",
        "How can I contact the {department}?",
        "Give me the helpline number",
        "What is your email address?",
        "Where is the hospital located?",
        "What is the hospital address?",
        "How do I reach the hospital from the railway station?",
        "Can I get the contact number of {doctor}'s clinic?",
        "Is there a WhatsApp number for the hospital?",
        "Who do I contact for complaints?",
        "Contact details of the {department} please",
        "I want to talk to a human at the helpdesk",
        "Do you have parking at the hospital?",
        "What is the reception number?",
        "How can I reach the patient relations team?",
        "Send me the hospital location on maps",
        "Which bus goes to your hospital?",
        "Customer care number?",
        "How can I give feedback about my visit?",
        "Whom should I call for {relation}'s admission enquiry?",
        "Phone number of the {department}?",
        "Is there a toll free number?",
        "What's the official website of the hospital?",
    ],
    "greetings": [
        "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
        "Namaste", "Hi there", "Hello, is anyone there?", "Hey, how are you?",
        "Greetings", "Hello helpdesk", "Hi, I need some help", "Hello, good morning",
        "Hey there", "Hii", "Hiiii", "Helo", "Namaskar", "Hi bot", "Yo",
        "Hello sir", "Hello madam", "Hi, can you help me?", "Good morning, hospital",
        "Hey assistant", "Hello, I have a question", "Heyy", "Hi, are you there?",
        "Good day", "Hello, who am I speaking with?", "Hi, how does this work?",
        "Hello, anybody available?", "Morning!", "Evening!",
        "Hey, I am new here", "Hello, nice to meet you", "Hi hospital helpdesk",
        "Hello, can I ask something?", "Hi, good evening",
    ],
    "thank_you": [
        "Thank you", "Thanks", "Thanks a lot", "Thank you so much", "Thank you very much",
        "Thanks for your help", "Many thanks", "That was helpful, thanks",
        "Thank you for the information", "Thanks, that solves my problem",
        "I appreciate your help", "Thanks a ton", "Thank you, you were very helpful",
        "Great, thanks", "Thanks for booking it", "Thank you for your patience",
        "Dhanyavaad", "Shukriya", "Thx", "Ty", "Thanks buddy", "Much appreciated",
        "Thank you for the quick response", "Okay thank you",
        "Thanks for helping {relation}", "Thank you, God bless you",
        "Thank you for your kind support", "Wonderful, thank you",
        "Thanks, that's exactly what I needed", "Thank you doctor",
        "Thanks for the update", "Really grateful for your help",
        "Thank you so much for helping me", "Thanks for the details",
        "Perfect, thanks", "Nice, thank you",
    ],
    "goodbye": [
        "Bye", "Goodbye", "See you", "See you later", "Bye bye", "Take care",
        "That's all, bye", "Ok bye", "Talk to you later", "Have a nice day",
        "Good night", "I'm done, goodbye", "Catch you later", "Bye for now",
        "That's all I needed, bye", "Alvida", "Chalo bye", "See ya", "Cya", "Later",
        "Nothing else, bye", "I'll go now", "Okay, signing off",
        "Have a good day, bye", "Will come back later, bye", "Goodbye and take care",
        "End chat", "Close the chat", "No more questions, bye", "Gotta go",
        "Bye, see you at the hospital", "Okay that's it, goodbye",
        "Bye, have a good evening",
    ],
}

# --------------------------------------------------------------------------- #
# Style transforms
# --------------------------------------------------------------------------- #
COMMON_MISSPELLINGS: dict[str, list[str]] = {
    "appointment": ["appointmnet", "apointment", "appoinment", "appointement", "appt"],
    "doctor": ["docter", "doctr", "dr", "doc"],
    "hospital": ["hospitl", "hospita", "hosptal", "hospitel"],
    "report": ["reprot", "repot", "reoprt"],
    "reports": ["reprots", "reoprts", "reports"],
    "insurance": ["insurence", "insurnce", "insuranse"],
    "pharmacy": ["pharmcy", "pharmasy", "farmacy"],
    "emergency": ["emergancy", "emergncy", "emargency"],
    "cancel": ["cancle", "cancell", "cansel"],
    "reschedule": ["reshedule", "rescedule", "re-schedule"],
    "schedule": ["shedule", "schedual"],
    "available": ["availble", "avilable", "avail"],
    "please": ["pls", "plz", "plese", "pleas"],
    "medicine": ["medecine", "medicin", "medisine"],
    "medicines": ["medecines", "medicins", "meds"],
    "tomorrow": ["tommorow", "tmrw", "tomorow"],
    "bill": ["bil", "billl"],
    "timings": ["timmings", "timing", "timeings"],
    "number": ["numbr", "no.", "num"],
    "thank": ["thnk", "thank", "tank"],
    "thanks": ["thx", "thanx", "thnks", "tnx"],
    "hello": ["helo", "hellow", "hallo"],
    "goodbye": ["gudbye", "good bye", "goodby"],
    "breathe": ["breath", "breth"],
    "ambulance": ["ambulence", "ambulanse", "ambulnce"],
    "available?": ["availble?", "avlbl?"],
    "consultation": ["consultaion", "consultation", "consulation"],
    "prescription": ["priscription", "prescripton"],
}

SLANG: dict[str, str] = {
    "you": "u", "please": "pls", "are": "r", "your": "ur", "tomorrow": "tmrw",
    "appointment": "appt", "doctor": "doc", "because": "bcoz", "want to": "wanna",
    "going to": "gonna", "right now": "rn", "message": "msg", "okay": "ok",
    "thanks": "thx", "information": "info", "as soon as possible": "asap",
}

FORMAL_PREFIXES = [
    "Good morning. ", "Dear Sir/Madam, ", "Respected helpdesk, ", "Hello, ",
    "Greetings. ", "Excuse me, ", "Good afternoon, ",
]
FORMAL_WRAPPERS = [
    "Could you kindly assist: {q}", "I would be grateful if you could help. {q}",
    "May I request your assistance? {q}", "{q} I would appreciate your guidance.",
    "{q} Kindly do the needful.", "I am writing to enquire. {q}",
]
INFORMAL_PREFIXES = ["hey ", "hi ", "umm ", "ok so ", "yo ", "hey there, ", "quick q - "]
INFORMAL_SUFFIXES = [" pls", " thx", " ?", " :)", " lol", " asap", " bro", " yaar"]
ELDERLY_PREFIXES = [
    "Hello dear, ", "Beta, ", "Excuse me young man, ", "Hello, I am a senior citizen and ",
    "My grandson told me to message here. ", "I am 72 years old, ",
    "Sorry, I am not good with phones. ", "Hello madam, I am an old lady and ",
]
ELDERLY_SUFFIXES = [
    " Please explain slowly.", " Please tell me in simple words.",
    " I cannot see the screen very well.", " My hearing is weak so please write it.",
    " Bless you.", " Please help this old man.",
]
URGENT_PREFIXES = ["URGENT: ", "Please hurry! ", "Urgent - ", "Quickly please, ", "ASAP: "]
URGENT_SUFFIXES = [" It's urgent!", " Please reply fast!!", " urgent!!!", " Need help right now.", "!!"]
LONG_CONTEXT = [
    "I was here last month for a checkup and the staff were very kind. ",
    "I have been trying to call the reception since morning but nobody picks up. ",
    "I am coming from another city so I need to plan my travel. ",
    "My family doctor referred me to your hospital. ",
    "I am an existing patient and my UHID is with me. ",
    "I tried the website but could not understand the process. ",
]

# Which styles make sense for which intents (e.g. no 'urgent goodbye').
SOCIAL_INTENTS = {"greetings", "thank_you", "goodbye"}


def _apply_misspellings(text: str, rng: random.Random, prob: float = 0.6) -> str:
    words = text.split(" ")
    out = []
    for word in words:
        key = word.lower().strip(",.!")
        if key in COMMON_MISSPELLINGS and rng.random() < prob:
            repl = rng.choice(COMMON_MISSPELLINGS[key])
            out.append(word.lower().replace(key, repl))
        else:
            out.append(word)
    return " ".join(out)


def _keyboard_typo(text: str, rng: random.Random) -> str:
    """Introduce 1-2 character-level typos (swap, drop, duplicate) in long words."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not candidates:
        return text
    for idx in rng.sample(candidates, k=min(len(candidates), rng.choice([1, 1, 2]))):
        w = words[idx]
        pos = rng.randrange(1, len(w) - 1)
        op = rng.choice(["swap", "drop", "dup"])
        if op == "swap":
            w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
        elif op == "drop":
            w = w[:pos] + w[pos + 1 :]
        else:
            w = w[:pos] + w[pos] + w[pos:]
        words[idx] = w
    return " ".join(words)


def style_plain(q: str, rng: random.Random, intent: str) -> str:
    return q


def style_formal(q: str, rng: random.Random, intent: str) -> str:
    if intent in SOCIAL_INTENTS:
        return rng.choice(["Respected sir, ", "Dear team, ", ""]) + q.rstrip(".!") + "."
    if rng.random() < 0.5:
        return rng.choice(FORMAL_PREFIXES) + q
    return rng.choice(FORMAL_WRAPPERS).format(q=q)


def style_informal(q: str, rng: random.Random, intent: str) -> str:
    q = q.lower().rstrip("?.!")
    if intent in SOCIAL_INTENTS:
        return q + rng.choice(["", " :)", "!!", " ya", " yaar", " 🙏"])
    return rng.choice(INFORMAL_PREFIXES) + q + rng.choice(INFORMAL_SUFFIXES)


def style_young(q: str, rng: random.Random, intent: str) -> str:
    q = q.lower()
    for full, short in SLANG.items():
        q = re.sub(rf"\b{re.escape(full)}\b", short, q)
    q = q.rstrip("?.!")
    return q + rng.choice(["", " ??", " pls", " rn", " 😅", " fr"])


def style_elderly(q: str, rng: random.Random, intent: str) -> str:
    if intent in SOCIAL_INTENTS:
        return q + rng.choice([" dear.", " beta.", ", God bless.", " my child."])
    text = rng.choice(ELDERLY_PREFIXES) + q[0].lower() + q[1:]
    if rng.random() < 0.6:
        text += rng.choice(ELDERLY_SUFFIXES)
    return text


def style_urgent(q: str, rng: random.Random, intent: str) -> str:
    if rng.random() < 0.5:
        return rng.choice(URGENT_PREFIXES) + q
    return q.rstrip(".") + rng.choice(URGENT_SUFFIXES)


def style_long(q: str, rng: random.Random, intent: str) -> str:
    return rng.choice(LONG_CONTEXT) + q + rng.choice(
        ["", " Please let me know the process.", " Thank you in advance.", " What should I do?"]
    )


def style_short(q: str, rng: random.Random, intent: str) -> str:
    """Keep only the informative tail of the query (telegraphic style)."""
    words = q.rstrip("?.!").split()
    if len(words) <= 4:
        return q.lower()
    keep = rng.randint(3, min(6, len(words)))
    return " ".join(words[-keep:]).lower()


def style_typo(q: str, rng: random.Random, intent: str) -> str:
    q = _apply_misspellings(q, rng)
    if rng.random() < 0.6:
        q = _keyboard_typo(q, rng)
    return q


StyleFn = Callable[[str, random.Random, str], str]

# (style_fn, weight) – weights are re-normalised per intent
STYLE_WEIGHTS: list[tuple[StyleFn, float]] = [
    (style_plain, 0.30),
    (style_formal, 0.10),
    (style_informal, 0.10),
    (style_young, 0.08),
    (style_elderly, 0.10),
    (style_urgent, 0.08),
    (style_long, 0.07),
    (style_short, 0.05),
    (style_typo, 0.12),
]


def _styles_for(intent: str) -> list[tuple[StyleFn, float]]:
    styles = STYLE_WEIGHTS
    if intent in SOCIAL_INTENTS:
        styles = [(f, w) for f, w in styles if f not in (style_urgent, style_long, style_short)]
    if intent == "emergency_assistance":
        # Emergencies are urgent far more often than not.
        styles = [(f, w * (3 if f is style_urgent else 1)) for f, w in styles]
    return styles


def fill_slots(template: str, rng: random.Random) -> str:
    def repl(match: re.Match) -> str:
        return rng.choice(SLOTS[match.group(1)])

    text = re.sub(r"\{(\w+)\}", repl, template)
    return text[0].upper() + text[1:] if text else text


def generate_for_intent(intent: str, n: int, rng: random.Random, max_tries: int = 50_000) -> list[str]:
    templates = TEMPLATES[intent]
    styles = _styles_for(intent)
    fns, weights = zip(*styles)

    seen: set[str] = set()
    examples: list[str] = []

    # 1) every base template appears at least once in plain form (coverage)
    for tpl in templates:
        text = fill_slots(tpl, rng)
        key = dedup_key(text)
        if key and key not in seen:
            seen.add(key)
            examples.append(text)

    # 2) sample styled variations until we have n unique examples
    tries = 0
    while len(examples) < n and tries < max_tries:
        tries += 1
        base = fill_slots(rng.choice(templates), rng)
        style = rng.choices(fns, weights=weights, k=1)[0]
        text = style(base, rng, intent).strip()
        # compound styles occasionally (e.g. elderly + typo)
        if rng.random() < 0.15 and style is not style_typo:
            text = style_typo(text, rng, intent)
        key = dedup_key(text)
        if key and key not in seen:
            seen.add(key)
            examples.append(text)

    if len(examples) < n:
        raise RuntimeError(f"Could only generate {len(examples)} unique examples for '{intent}'")

    rng.shuffle(examples)
    return examples[:n]


def generate_dataset(per_intent: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for intent in INTENTS:
        for text in generate_for_intent(intent, per_intent, rng):
            rows.append({"text": text, "intent": intent})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-intent", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(INTENTS_CSV))
    args = parser.parse_args()

    df = generate_dataset(args.per_intent, args.seed)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %d examples (%d intents) to %s", len(df), df["intent"].nunique(), args.output)
    logger.info("Per-intent counts:\n%s", df["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
