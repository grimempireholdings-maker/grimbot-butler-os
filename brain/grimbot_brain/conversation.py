from __future__ import annotations

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
    memory_context = RobotMemory(memory).relevant(
        RelevantMemoryRequest(
            query=stt.transcript,
            room_name=request.room_name,
            zone_name=request.zone_name,
            limit=10,
        )
    )
    machine_output = memory_context.model_dump()
    maya_response = compose_maya_response(
        MayaComposeRequest(
            raw_output=machine_output,
            mode=request.assistant_mode,
            response_mode=request.response_mode,
            verified=request.verified,
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
