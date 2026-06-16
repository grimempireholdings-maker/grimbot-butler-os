from __future__ import annotations

from .conversation_agent import run_conversation_agent
from .memory import BrainMemory
from .persona import directives_for_mode, resolve_permission
from .robot_memory import RobotMemory
from .schemas import (
    MayaComposedResponse,
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
    agent_response = run_conversation_agent(
        request=request,
        transcript=stt.transcript,
        memory=memory,
        memory_context=memory_context,
    )
    machine_output = agent_response.machine_output
    maya_response = MayaComposedResponse(
        mode=request.assistant_mode,
        permission=resolve_permission(request.assistant_mode, "suggest", agent_response.verified),
        verified=agent_response.verified,
        directives_applied=directives_for_mode(request.assistant_mode),
        machine_output=machine_output,
        user_response=agent_response.user_response,
    )
    speech_output = text_to_speech(agent_response.user_response)

    return VoiceConversationResponse(
        transcript=stt.transcript,
        memory_context=memory_context,
        agent_response=agent_response,
        maya_response=maya_response,
        speech_output=speech_output,
        machine_output=machine_output,
    )
