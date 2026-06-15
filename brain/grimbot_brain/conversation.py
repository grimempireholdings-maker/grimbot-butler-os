from __future__ import annotations

import re

from .identity.context_schemas import ContextSearchRequest
from .identity.context_store import ContextStore
from .maya_core import build_maya_briefing
from .memory import BrainMemory
from .robot_memory import RobotMemory
from .response_composer import compose_maya_response
from .schemas import (
    MayaComposeRequest,
    RelevantMemoryRequest,
    VoiceConversationRequest,
    VoiceConversationResponse,
)
from .voice import speech_to_text, text_to_speech


def run_voice_conversation(request: VoiceConversationRequest, memory: BrainMemory) -> VoiceConversationResponse:
    if not request.push_to_talk:
        raise ValueError("Voice conversation requires push_to_talk=true")

    stt = speech_to_text(mock_transcript=request.mock_transcript, audio_path=request.audio_path)
    robot_memory = RobotMemory(memory)
    memory_context = robot_memory.relevant(
        RelevantMemoryRequest(
            query=stt.transcript,
            room_name=request.room_name,
            zone_name=request.zone_name,
            limit=10,
        )
    )
    context = ContextStore(memory)
    context_result = context.search(ContextSearchRequest(query=stt.transcript, limit=10))
    response_verified = request.verified

    if _is_briefing_request(stt.transcript):
        briefing = build_maya_briefing(
            request=_briefing_request(request),
            memory=robot_memory,
            context=context,
        )
        machine_output = briefing.model_dump()
        machine_output["context_summary"] = _briefing_summary(briefing)
        response_verified = briefing.verified
    elif _mentions_project(stt.transcript, context.projects()):
        machine_output = context_result.model_dump()
        machine_output["context_summary"] = _context_answer(context_result)
        response_verified = request.verified and _context_answer_is_verified(context_result)
    elif _is_physical_request(stt.transcript, request.room_name, request.zone_name):
        machine_output = memory_context.model_dump()
    elif context_result.projects or context_result.entries:
        machine_output = context_result.model_dump()
        machine_output["context_summary"] = _context_answer(context_result)
        response_verified = request.verified and _context_answer_is_verified(context_result)
    else:
        machine_output = context_result.model_dump()
        machine_output["context_summary"] = context_result.clarification_question
        response_verified = False

    maya_response = compose_maya_response(
        MayaComposeRequest(
            raw_output=machine_output,
            mode=request.assistant_mode,
            response_mode=request.response_mode,
            verified=response_verified,
            requested_permission="suggest",
            user_goal=stt.transcript,
        )
    )
    speech_output = text_to_speech(maya_response.user_response)

    return VoiceConversationResponse(
        transcript=stt.transcript,
        memory_context=memory_context,
        maya_response=maya_response,
        speech_output=speech_output,
        machine_output=machine_output,
    )


def _briefing_request(request: VoiceConversationRequest):
    from .schemas import MayaBriefingRequest

    return MayaBriefingRequest(
        room_name=request.room_name,
        zone_name=request.zone_name,
        verified=request.verified,
        mode=request.assistant_mode,
    )


def _is_briefing_request(query: str) -> bool:
    normalized = _normalize(query)
    phrases = (
        "how is my day looking",
        "how s my day looking",
        "hows my day looking",
        "brief me",
        "my briefing",
        "top priorities",
        "my priorities",
        "what should i focus on",
    )
    return any(phrase in normalized for phrase in phrases)


def _is_physical_request(query: str, room_name: str | None, zone_name: str | None) -> bool:
    if room_name or zone_name:
        return True
    normalized = _normalize(query)
    tokens = set(normalized.split())
    physical_tokens = {
        "room", "clean", "cleanup", "mess", "hazard", "floor", "desk",
        "kitchen", "bedroom", "camera", "vision", "scan", "sensor",
        "battery", "distance", "move",
    }
    return bool(tokens & physical_tokens) or "physical environment" in normalized


def _mentions_project(query: str, projects) -> bool:
    normalized = _normalize(query)
    return any(_normalize(project.name) in normalized for project in projects)


def _briefing_summary(briefing) -> str:
    priorities = "; ".join(briefing.priority_items[:2])
    projects = ", ".join(briefing.active_projects[:3])
    bottleneck = briefing.current_bottlenecks[0] if briefing.current_bottlenecks else "No bottleneck recorded."
    return (
        f"Boss, here is the signal. Top priorities: {priorities}. "
        f"Active projects: {projects}. Current bottleneck: {bottleneck}. "
        f"Next best action: {briefing.next_best_action}"
    )


def _context_answer(result) -> str:
    if result.projects:
        project = result.projects[0]
        return (
            f"{project.name} is {project.status}. Current bottleneck: "
            f"{project.current_bottleneck}. Next action: {project.next_action}"
        )
    entry = result.entries[0]
    return f"{entry.name}: {entry.content}"


def _context_answer_is_verified(result) -> bool:
    if result.projects:
        return result.projects[0].verified
    if result.entries:
        return result.entries[0].verified
    return False


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
