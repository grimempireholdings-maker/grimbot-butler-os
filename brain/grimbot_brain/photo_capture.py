from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from .conversation_agent import ConversationProvider, run_conversation_agent
from .memory import BrainMemory
from .perception import analyze_user_photo
from .schemas import (
    PhotoAnalysisResult,
    PhotoConversationResponse,
    VoiceConversationRequest,
)
from .vision import MAX_IMAGE_BYTES, image_directory


PHOTO_MEDIA_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


def process_photo_capture(
    data: bytes,
    media_type: str,
    user_prompt: str,
    memory: BrainMemory,
    *,
    analyzer: Callable[[str | Path, str, str], PhotoAnalysisResult] = analyze_user_photo,
    provider: ConversationProvider | None = None,
) -> PhotoConversationResponse:
    """Analyze one user-selected photo and delete its temporary bytes in all outcomes."""
    normalized_type = media_type.split(";", 1)[0].strip().lower()
    suffix = validate_photo_upload(data, normalized_type)
    prompt = " ".join(user_prompt.split())[:1000] or "What do you notice in this photo?"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="user_photo_",
            dir=image_directory(),
            delete=False,
        ) as handle:
            handle.write(data)
            temp_path = Path(handle.name)

        analysis = analyzer(temp_path, normalized_type, prompt)
        memory.log_episode(
            "photo_capture",
            json.dumps(
                {
                    "description": analysis.description,
                    "user_prompt": prompt,
                    "vision_mode": analysis.mode,
                    "vision_model": analysis.model,
                    "media_type": normalized_type,
                    "raw_media_stored": False,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            importance=0.65,
        )
        request = VoiceConversationRequest(push_to_talk=True, mock_transcript=prompt)
        response = run_conversation_agent(
            request=request,
            transcript=prompt,
            memory=memory,
            provider=provider,
            visual_observation=analysis,
        )
        return PhotoConversationResponse(analysis=analysis, agent_response=response)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def validate_photo_upload(data: bytes, media_type: str) -> str:
    if media_type not in PHOTO_MEDIA_SUFFIXES:
        raise ValueError("Unsupported photo type")
    if not data:
        raise ValueError("Photo was empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Photo exceeds the 10 MB limit")
    if not _matches_signature(data, media_type):
        raise ValueError("Photo content did not match its declared type")
    return PHOTO_MEDIA_SUFFIXES[media_type]


def _matches_signature(data: bytes, media_type: str) -> bool:
    if media_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/webp":
        return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    if media_type in {"image/heic", "image/heif"}:
        return len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in {
            b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"
        }
    return False
