from __future__ import annotations

import os
from pathlib import Path

from .schemas import SpeechToTextResult, TextToSpeechResult

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024
DEFAULT_AUDIO_DIR = Path("voice/audio")


def audio_directory() -> Path:
    directory = Path(os.getenv("GRIMBOT_AUDIO_DIR", str(DEFAULT_AUDIO_DIR))).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def approved_audio_path(audio_path: str | None, audio_dir: str | Path | None = None) -> Path | None:
    if not audio_path:
        return None

    safe_dir = Path(audio_dir).expanduser() if audio_dir else audio_directory()
    safe_root = safe_dir.resolve()
    path = Path(audio_path).expanduser()

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return None

    if not resolved.is_file():
        return None

    if resolved.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        return None

    try:
        resolved.relative_to(safe_root)
    except ValueError:
        return None

    if resolved.stat().st_size > MAX_AUDIO_BYTES:
        return None

    return resolved


def speech_to_text(mock_transcript: str | None = None, audio_path: str | None = None) -> SpeechToTextResult:
    mock_enabled = os.getenv("GRIMBOT_VOICE_MOCK", "true").lower() != "false"
    if mock_enabled:
        transcript = _safe_transcript(mock_transcript, fallback="input unavailable")
        return SpeechToTextResult(transcript=transcript, mode="mock", source="mock_transcript")

    approved = approved_audio_path(audio_path)
    if not approved:
        transcript = _safe_transcript(mock_transcript, fallback="audio unavailable")
        return SpeechToTextResult(transcript=transcript, mode="mock", source="fallback")

    return SpeechToTextResult(
        transcript="local speech-to-text placeholder",
        mode="local",
        source=str(approved),
    )


def text_to_speech(text: str) -> TextToSpeechResult:
    mock_enabled = os.getenv("GRIMBOT_VOICE_MOCK", "true").lower() != "false"
    clean_text = text.strip() or "No response available."
    if mock_enabled:
        return TextToSpeechResult(text=clean_text, mode="mock", audio_path=None)

    return TextToSpeechResult(text=clean_text, mode="local", audio_path=None)


def _safe_transcript(value: str | None, fallback: str) -> str:
    transcript = (value or "").strip()
    return transcript or fallback
