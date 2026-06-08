from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .memory import BrainMemory
from .robot_memory import RobotMemory
from .schemas import RoomScanRequest, RoomScanResult
from .vision import approved_image_path, capture_webcam_frame


def run_room_scan(request: RoomScanRequest, memory: BrainMemory) -> RoomScanResult:
    image_path = _resolve_scan_image(request)
    mock_enabled = os.getenv("GRIMBOT_MOCK_PERCEPTION", "true").lower() != "false"
    api_key = os.getenv("GEMINI_API_KEY")

    if mock_enabled or not api_key or not image_path:
        result = _mock_room_scan(request, image_path)
    else:
        result = _gemini_room_scan(image_path, api_key)

    memory.log_room_scan(result)
    RobotMemory(memory).ingest_room_scan(result, room_name=request.room_name, zone_name=request.zone_name)
    return result


def _resolve_scan_image(request: RoomScanRequest) -> Path | None:
    if request.capture_webcam:
        return capture_webcam_frame(camera_index=request.camera_index)

    return approved_image_path(request.image_path)


def _mock_room_scan(request: RoomScanRequest, image_path: Path | None) -> RoomScanResult:
    frame_text = (request.mock_camera_frame or "").lower()
    visible_objects = ["floor", "table", "chair"]
    mess_zones = []
    hazards = []

    if "laundry" in frame_text:
        visible_objects.append("laundry")
        mess_zones.append("laundry pile")
    if "dishes" in frame_text:
        visible_objects.append("dishes")
        mess_zones.append("dishes on table")
    if "cord" in frame_text or "cable" in frame_text:
        visible_objects.append("cord")
        hazards.append("loose cord on floor")
    if "notebook" in frame_text:
        visible_objects.append("notebook")
        mess_zones.append("notebooks on desk")
    if "drink" in frame_text or "cup" in frame_text or "container" in frame_text:
        visible_objects.append("drink container")
        mess_zones.append("drink containers on surface")
    if "spill" in frame_text:
        hazards.append("possible liquid spill")
        mess_zones.append("spill area")

    if not mess_zones:
        mess_zones.append("general surfaces")

    cleanup_order = hazards + mess_zones
    if not cleanup_order:
        cleanup_order = ["scan room again from another angle"]

    return RoomScanResult(
        room_summary="Mock room scan completed with simulated visual context.",
        visible_objects=visible_objects,
        mess_zones=mess_zones,
        hazards=hazards,
        suggested_cleanup_order=cleanup_order,
        next_best_action=cleanup_order[0],
        mode="mock",
        image_path=str(image_path) if image_path else None,
    )


def _gemini_room_scan(image_path: Path, api_key: str) -> RoomScanResult:
    try:
        import google.generativeai as genai
    except ImportError:
        return _mock_room_scan(RoomScanRequest(image_path=str(image_path)), image_path)

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)
    uploaded_image = genai.upload_file(str(image_path))
    prompt = (
        "Scan this room image for a robotic butler. Return JSON only with keys: "
        "room_summary, visible_objects, mess_zones, hazards, suggested_cleanup_order, next_best_action. "
        "Use arrays of short strings for list fields. Do not include motor commands."
    )
    response = model.generate_content([prompt, uploaded_image])
    raw_text = getattr(response, "text", "") or "{}"

    try:
        payload = _extract_json_object(raw_text)
        payload["mode"] = "gemini"
        payload["image_path"] = str(image_path)
        return RoomScanResult.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return _unreadable_gemini_result(image_path)


def _unreadable_gemini_result(image_path: Path) -> RoomScanResult:
    return RoomScanResult(
        room_summary="Gemini room scan returned an unreadable response.",
        visible_objects=[],
        mess_zones=[],
        hazards=["vision response could not be parsed"],
        suggested_cleanup_order=["retry room scan"],
        next_best_action="retry room scan",
        mode="gemini",
        image_path=str(image_path),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")

    return json.loads(stripped[start : end + 1])
