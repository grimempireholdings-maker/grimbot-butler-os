from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .schemas import BrainCycleInput, PerceptionResult, PhotoAnalysisResult
from .vision import approved_image_path


def perceive(cycle_input: BrainCycleInput) -> PerceptionResult:
    """Return a structured perception result from mock data or Gemini."""
    mock_enabled = os.getenv("GRIMBOT_MOCK_PERCEPTION", "true").lower() != "false"
    api_key = os.getenv("GEMINI_API_KEY")

    if mock_enabled or not api_key:
        return _mock_perception(cycle_input)

    return _gemini_perception(cycle_input, api_key)


def analyze_user_photo(
    image_path: str | Path,
    media_type: str,
    user_prompt: str = "What do you notice in this photo?",
) -> PhotoAnalysisResult:
    """Analyze one explicitly shared photo with Gemini; never return a mock observation."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not gemini_key and not openrouter_key:
        raise RuntimeError("Gemini vision is not configured")
    approved = approved_image_path(str(image_path))
    if not approved:
        raise ValueError("Photo failed image safety validation")

    direct_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"
    openrouter_model = (
        os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash-lite").strip()
        or "google/gemini-2.5-flash-lite"
    )
    prompt = (
        "You are the visual perception layer for Maya, a personal assistant. Analyze exactly this one "
        "user-initiated photo. Describe what is visibly supported, address the user's question when possible, "
        "and state uncertainty instead of guessing. Do not claim a live feed, continuous vision, or background "
        f"camera access. User question: {user_prompt[:1000]}"
    )
    encoded_image = base64.b64encode(approved.read_bytes()).decode("ascii")
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": media_type,
                    "data": encoded_image,
                }},
            ]
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
    }
    if gemini_key:
        model = direct_model
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    else:
        model = openrouter_model
        openrouter_payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded_image}"}},
                ],
            }],
            "max_tokens": 700,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "GrimBot Butler OS",
        }
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(openrouter_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini vision request failed") from exc

    if gemini_key:
        candidates = result.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        description = " ".join(
            part.get("text", "").strip() for part in parts if isinstance(part, dict) and part.get("text")
        ).strip()
    else:
        choices = result.get("choices", [])
        description = str(choices[0].get("message", {}).get("content", "")).strip() if choices else ""
    if not description:
        raise RuntimeError("Gemini vision returned no description")
    return PhotoAnalysisResult(
        description=description,
        model=model,
        media_type=media_type,
        raw_media_stored=False,
    )


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
    image_path = approved_image_path(cycle_input.image_path)
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
