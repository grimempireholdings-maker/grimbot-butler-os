from __future__ import annotations

import os
from pathlib import Path

from .schemas import BrainCycleInput, PerceptionResult

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def perceive(cycle_input: BrainCycleInput) -> PerceptionResult:
    """Return a structured perception result from mock data or Gemini."""
    mock_enabled = os.getenv("GRIMBOT_MOCK_PERCEPTION", "true").lower() != "false"
    api_key = os.getenv("GEMINI_API_KEY")

    if mock_enabled or not api_key:
        return _mock_perception(cycle_input)

    return _gemini_perception(cycle_input, api_key)


def _mock_perception(cycle_input: BrainCycleInput) -> PerceptionResult:
    frame_text = (cycle_input.mock_camera_frame or cycle_input.image_path or "mock camera frame").lower()
    obstacle_words = ("obstacle", "wall", "chair", "blocked", "person", "close")
    obstacle_detected = any(word in frame_text for word in obstacle_words) or cycle_input.distance_cm < 50

    if obstacle_detected:
        summary = "Obstacle or nearby object detected in simulated camera input"
    else:
        summary = "Path appears clear in simulated camera input"

    return PerceptionResult(
        mode="mock",
        scene_summary=summary,
        obstacle_detected=obstacle_detected,
        obstacle_distance_cm=cycle_input.distance_cm if obstacle_detected else None,
        confidence=0.75,
    )


def _gemini_perception(cycle_input: BrainCycleInput, api_key: str) -> PerceptionResult:
    try:
        import google.generativeai as genai
    except ImportError:
        return _mock_perception(cycle_input)

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = (
        "Analyze this robot sensor cycle. Return a concise natural-language scene summary only. "
        f"Image path: {cycle_input.image_path or 'none'}; "
        f"mock camera frame: {cycle_input.mock_camera_frame or 'none'}; "
        f"distance_cm: {cycle_input.distance_cm}."
    )
    image_path = _safe_image_path(cycle_input.image_path)
    if image_path:
        uploaded_image = genai.upload_file(str(image_path))
        response = model.generate_content([prompt, uploaded_image])
    else:
        response = model.generate_content(prompt)
    summary = getattr(response, "text", "") or "Gemini perception completed"
    obstacle_detected = cycle_input.distance_cm < 50 or "obstacle" in summary.lower()

    return PerceptionResult(
        mode="gemini",
        scene_summary=summary.strip(),
        obstacle_detected=obstacle_detected,
        obstacle_distance_cm=cycle_input.distance_cm if obstacle_detected else None,
        confidence=0.8,
    )


def _safe_image_path(image_path: str | None) -> Path | None:
    if not image_path:
        return None

    path = Path(image_path).expanduser()
    if not path.exists() or not path.is_file():
        return None

    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return None

    if path.stat().st_size > MAX_IMAGE_BYTES:
        return None

    return path
