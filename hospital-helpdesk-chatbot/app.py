"""Streamlit UI for the multilingual hospital helpdesk chatbot.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html

import streamlit as st

from src.utils import HOSPITAL_INFO_JSON, INTENT_MODEL_DIR, load_json

st.set_page_config(page_title="Hospital Helpdesk Assistant", page_icon="🏥", layout="wide")

HOSPITAL = load_json(HOSPITAL_INFO_JSON)

EMOTION_STYLE = {
    "positive": ("😊", "#16a34a"),
    "neutral": ("😐", "#64748b"),
    "sadness": ("😢", "#2563eb"),
    "fear": ("😟", "#d97706"),
    "anger": ("😠", "#dc2626"),
}
LANGUAGE_OPTIONS = {"Auto-detect": "auto", "English": "en", "हिन्दी (Hindi)": "hi"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1200px;}
      .hero {background: linear-gradient(120deg,#0ea5e9 0%,#2563eb 60%,#1e40af 100%);
             color:#fff;padding:1.2rem 1.5rem;border-radius:16px;margin-bottom:1rem;}
      .hero h1 {margin:0;font-size:1.7rem;color:#fff;}
      .hero p {margin:.25rem 0 0 0;opacity:.92;}
      .badge {display:inline-block;padding:2px 10px;margin:2px 6px 2px 0;border-radius:999px;
              font-size:.78rem;font-weight:600;color:#fff;}
      .meta {font-size:.8rem;color:#64748b;margin-top:.35rem;}
      .emergency {background:#fee2e2;border-left:5px solid #dc2626;padding:.6rem .9rem;
                  border-radius:8px;color:#7f1d1d;font-weight:600;margin-bottom:.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading intent & emotion models...")
def load_chatbot():
    from src.chatbot import build_default_chatbot

    return build_default_chatbot()


@st.cache_resource(show_spinner="Loading Whisper speech model...")
def load_whisper_model(size: str):
    from src.audio import load_whisper

    return load_whisper(size)


@st.cache_data(show_spinner=False, max_entries=256)
def synthesize(text: str, lang: str) -> bytes | None:
    from src.audio import text_to_speech

    try:
        return text_to_speech(text, lang=lang)
    except Exception as exc:  # network issues etc.
        st.toast(f"Text-to-speech unavailable: {exc}", icon="⚠️")
        return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    lang_label = st.selectbox("Conversation language", list(LANGUAGE_OPTIONS), index=0)
    language = LANGUAGE_OPTIONS[lang_label]
    voice_reply = st.toggle("🔊 Speak responses", value=True)
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
    min_conf = st.slider("Min. intent confidence", 0.0, 1.0, 0.5, 0.05,
                         help="Below this, the assistant asks the user to rephrase.")
    show_debug = st.toggle("Show model details", value=True)

    st.divider()
    st.subheader("📞 Quick contacts")
    st.markdown(
        f"**Emergency:** {HOSPITAL['emergency_number']} / {HOSPITAL['hospital_emergency_line']}  \n"
        f"**Helpline:** {HOSPITAL['helpline']}  \n"
        f"**WhatsApp:** {HOSPITAL['whatsapp']}"
    )
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_audio_hash = None
        st.rerun()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    f"""<div class="hero"><h1>🏥 {html.escape(HOSPITAL['hospital_name'])} Helpdesk</h1>
    <p>Ask about appointments, doctors, timings, reports, billing, insurance or pharmacy —
    by typing or speaking, in English or Hindi.</p></div>""",
    unsafe_allow_html=True,
)

if not (INTENT_MODEL_DIR / "config.json").exists():
    st.error(
        "The intent model has not been trained yet. Run:\n\n"
        "```bash\npython -m scripts.generate_dataset\npython -m scripts.validate_dataset\n"
        "python -m src.train_intent\n```"
    )
    st.stop()

bot = load_chatbot()
bot.min_intent_confidence = min_conf

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_audio_hash", None)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{html.escape(text)}</span>'


def render_assistant(msg: dict) -> None:
    meta = msg["meta"]
    if meta["intent"] == "emergency_assistance":
        st.markdown(
            f'<div class="emergency">🚨 If this is life-threatening, call {HOSPITAL["emergency_number"]} now.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(msg["content"])
    if msg.get("audio"):
        st.audio(msg["audio"], format="audio/mp3", autoplay=msg.pop("autoplay", False))
    if show_debug:
        emoji, color = EMOTION_STYLE.get(meta["emotion"], ("", "#64748b"))
        badges = (
            badge(f"intent: {meta['intent']}", "#2563eb")
            + badge(f"confidence: {meta['intent_confidence']:.0%}", "#0891b2")
            + badge(f"{emoji} emotion: {meta['emotion']} ({meta['emotion_confidence']:.0%})", color)
            + badge(f"lang: {meta['language']}", "#7c3aed")
        )
        if meta.get("safety_override"):
            badges += badge("safety override", "#dc2626")
        if meta.get("fallback_reason"):
            badges += badge(f"fallback: {meta['fallback_reason']}", "#9ca3af")
        st.markdown(badges, unsafe_allow_html=True)
        with st.expander("Details"):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Top intents")
                for name, p in meta["top_intents"]:
                    st.progress(float(p), text=f"{name} — {p:.1%}")
            with c2:
                st.caption(f"Emotion groups (top GoEmotions label: {meta['emotion_top_label']})")
                for name, p in sorted(meta["emotion_scores"].items(), key=lambda kv: -kv[1]):
                    st.progress(float(min(p, 1.0)), text=f"{name} — {p:.1%}")
            if meta["language"] != "en":
                st.caption(f"Query (EN): {meta['english_text']}")
                st.caption(f"Response (EN): {meta['response_en']}")


def handle_query(text: str, source: str, lang_hint: str | None) -> None:
    st.session_state.messages.append({"role": "user", "content": text, "source": source})
    with st.spinner("Thinking..."):
        result = bot.respond(text, language=lang_hint)
    audio = synthesize(result.response, result.language) if voice_reply else None
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.response,
        "audio": audio,
        "autoplay": True,
        "meta": result.to_dict(),
    })


# --------------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------------- #
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🏥"):
        st.markdown(
            "Hello! I'm the virtual helpdesk assistant. How can I help you today?  \n"
            "_नमस्ते! आप हिंदी में भी पूछ सकते हैं।_"
        )

for message in st.session_state.messages:
    avatar = "🏥" if message["role"] == "assistant" else ("🎙️" if message.get("source") == "voice" else "🧑")
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            render_assistant(message)
        else:
            st.markdown(message["content"])

# --------------------------------------------------------------------------- #
# Inputs: voice + text
# --------------------------------------------------------------------------- #
voice_col, _ = st.columns([2, 3])
with voice_col:
    recording = st.audio_input("🎙️ Speak your question")

if recording is not None:
    audio_bytes = recording.getvalue()
    audio_hash = hashlib.sha1(audio_bytes).hexdigest()
    if audio_hash != st.session_state.last_audio_hash:  # process each recording once
        st.session_state.last_audio_hash = audio_hash
        from src.audio import speech_to_text

        load_whisper_model(whisper_size)
        with st.spinner("Transcribing..."):
            try:
                whisper_lang = None if language == "auto" else language
                transcript = speech_to_text(audio_bytes, language=whisper_lang, model_size=whisper_size)
            except Exception as exc:
                st.error(f"Could not transcribe audio: {exc}")
                transcript = None
        if transcript and transcript.text:
            hint = language if language != "auto" else (transcript.language if transcript.language in ("en", "hi") else None)
            handle_query(transcript.text, "voice", hint)
            st.rerun()
        elif transcript is not None:
            st.warning("I couldn't hear anything — please try again.")

if prompt := st.chat_input("Type your question (English / हिन्दी)..."):
    handle_query(prompt, "text", language)
    st.rerun()
