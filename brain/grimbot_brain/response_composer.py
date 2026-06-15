from __future__ import annotations

import math
import re

from .persona import directives_for_mode, resolve_permission, verification_phrase
from .schemas import MayaComposeRequest, MayaComposedResponse, RobotCommand

MAX_RESPONSE_PART_LENGTH = 360


def compose_maya_response(request: MayaComposeRequest) -> MayaComposedResponse:
    permission = resolve_permission(request.mode, request.requested_permission, request.verified)
    user_response = _compose_text(request, permission)
    return MayaComposedResponse(
        mode=request.mode,
        permission=permission,
        verified=request.verified,
        directives_applied=directives_for_mode(request.mode),
        machine_output=request.raw_output,
        user_response=user_response,
    )


def _compose_text(request: MayaComposeRequest, permission: str) -> str:
    if request.mode == "quiet_observer":
        return f"{verification_phrase(request.verified)} Observed. No action taken."

    if request.mode == "neutral_robot":
        return _neutral_text(request, permission)

    if request.response_mode == "cleanup_coaching":
        return _cleanup_coaching_text(request, permission)

    return _maya_text(request, permission)


def _neutral_text(request: MayaComposeRequest, permission: str) -> str:
    summary = _safe_part(_extract_summary(request.raw_output), request.verified)
    return f"{verification_phrase(request.verified)} {summary} Permission: {permission}."


def _maya_text(request: MayaComposeRequest, permission: str) -> str:
    summary = _safe_part(_extract_summary(request.raw_output), request.verified)
    state_prefix = _state_prefix(request.adaptive_state)
    if _is_stop_command(request.raw_output):
        return f"{verification_phrase(request.verified)} Safety wins: {summary}"
    if state_prefix:
        summary = f"{state_prefix} {summary}"
    if permission == "ask_approval":
        return f"{verification_phrase(request.verified)} I need approval before acting. Recommendation: {summary}"
    return f"{verification_phrase(request.verified)} Here is the signal: {summary}"


def _cleanup_coaching_text(request: MayaComposeRequest, permission: str) -> str:
    action = _safe_part(_extract_next_action(request.raw_output), request.verified)
    state_prefix = _state_prefix(request.adaptive_state)
    if state_prefix:
        action = f"{state_prefix} {action}"
    if _is_stop_command(request.raw_output):
        return f"{verification_phrase(request.verified)} Stop first. {action}"
    if permission == "ask_approval":
        return f"{verification_phrase(request.verified)} Best move: {action}. I need approval before execution."
    return f"{verification_phrase(request.verified)} First: {action}. Then reassess."


def _extract_summary(raw_output: dict) -> str:
    if "context_summary" in raw_output and raw_output["context_summary"]:
        return str(raw_output["context_summary"])
    if "room_summary" in raw_output:
        return str(raw_output["room_summary"])
    if isinstance(raw_output.get("data"), dict) and "room_summary" in raw_output["data"]:
        return str(raw_output["data"]["room_summary"])
    if "next_best_action" in raw_output:
        return f"Next best action is {raw_output['next_best_action']}"
    if {"action", "reason"}.issubset(raw_output):
        return f"{raw_output['action']}: {raw_output['reason']}"
    return "Structured output received."


def _extract_next_action(raw_output: dict) -> str:
    if "next_best_action" in raw_output:
        return str(raw_output["next_best_action"])
    if "suggested_cleanup_order" in raw_output and raw_output["suggested_cleanup_order"]:
        return str(raw_output["suggested_cleanup_order"][0])
    if "reason" in raw_output:
        return str(raw_output["reason"])
    return "clear the highest-risk item first"


def _is_stop_command(raw_output: dict) -> bool:
    try:
        command = RobotCommand.model_validate(raw_output)
    except ValueError:
        return False
    return command.action == "stop"


def _safe_part(value: str, verified: bool) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not verified:
        text = re.sub(r"\bverified\b\s*[:.-]?\s*", "", text, flags=re.IGNORECASE)
    if len(text) > MAX_RESPONSE_PART_LENGTH:
        return text[: MAX_RESPONSE_PART_LENGTH - 3].rstrip() + "..."
    return text or "Structured output received."


def _state_prefix(adaptive_state: dict[str, float] | None) -> str:
    if not adaptive_state:
        return ""

    urgency = _state_value(adaptive_state, "urgency")
    friction = _state_value(adaptive_state, "friction")
    confidence = _state_value(adaptive_state, "confidence")
    novelty = _state_value(adaptive_state, "novelty")
    curiosity = _state_value(adaptive_state, "curiosity")

    if urgency >= 0.65:
        return "Urgency is elevated, so I will keep this concise."
    if friction >= 0.60:
        return "Pressure is high; one small step is enough."
    if confidence >= 0.70:
        return "Confidence is high; I recommend this clearly."
    if novelty >= 0.60 or curiosity >= 0.65:
        return "New context is showing up; exploration is useful."
    return ""


def _state_value(adaptive_state: dict[str, float], key: str) -> float:
    try:
        value = float(adaptive_state.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
