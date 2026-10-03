"""Voice I/O: Whisper speech-to-text and gTTS text-to-speech.

* ``speech_to_text`` accepts a file path or raw audio bytes (wav/mp3/webm/...)
  and returns the transcript plus Whisper's detected language. Whisper decodes
  audio with ``ffmpeg``, which must be installed on the system.
* ``text_to_speech`` returns MP3 bytes (and optionally writes them to disk).
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
GTTS_LANGS = {"en": "en", "hi": "hi"}


@dataclass
class Transcription:
    text: str
    language: str


@lru_cache(maxsize=2)
def load_whisper(model_size: str = DEFAULT_WHISPER_MODEL):
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Whisper '%s' on %s", model_size, device)
    return whisper.load_model(model_size, device=device)


def speech_to_text(
    audio: str | Path | bytes,
    language: str | None = None,
    model_size: str = DEFAULT_WHISPER_MODEL,
    suffix: str = ".wav",
) -> Transcription:
    """Transcribe speech with Whisper.

    Args:
        audio: path to an audio file or the raw bytes of one.
        language: ISO code ("en", "hi") to force; ``None`` lets Whisper detect it.
        model_size: tiny | base | small | medium | large.
        suffix: file extension used when ``audio`` is bytes (helps ffmpeg).
    """
    model = load_whisper(model_size)
    tmp_path: str | None = None
    try:
        if isinstance(audio, (bytes, bytearray)):
            if not audio:
                raise ValueError("Empty audio input")
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
            path = tmp_path
        else:
            path = str(audio)
            if not Path(path).exists():
                raise FileNotFoundError(path)

        result = model.transcribe(path, language=language, fp16=model.device.type == "cuda")
        text = (result.get("text") or "").strip()
        lang = result.get("language") or language or "en"
        logger.info("Transcribed %d chars (lang=%s)", len(text), lang)
        return Transcription(text=text, language=lang)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def text_to_speech(text: str, lang: str = "en", output_path: str | Path | None = None, slow: bool = False) -> bytes:
    """Synthesize speech with gTTS and return MP3 bytes.

    Requires network access to Google's TTS endpoint.
    """
    from gtts import gTTS

    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text")
    tts = gTTS(text=text, lang=GTTS_LANGS.get(lang, "en"), slow=slow)
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    audio_bytes = buffer.getvalue()
    if output_path:
        Path(output_path).write_bytes(audio_bytes)
    return audio_bytes
