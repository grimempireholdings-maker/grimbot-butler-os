import pytest

from grimbot_brain.conversation import run_voice_conversation
from grimbot_brain.main import app
from grimbot_brain.memory import BrainMemory
from grimbot_brain.robot_memory import RobotMemory
from grimbot_brain.room_scan import run_room_scan
from grimbot_brain.safety import validate_action
from grimbot_brain.schemas import (
    BrainCycleInput,
    RememberRequest,
    RobotIntent,
    RoomScanRequest,
    VoiceConversationRequest,
)
from grimbot_brain.voice import approved_audio_path, audio_directory, speech_to_text, text_to_speech


def test_voice_conversation_requires_push_to_talk(tmp_path) -> None:
    with pytest.raises(ValueError):
        run_voice_conversation(
            VoiceConversationRequest(push_to_talk=False, mock_transcript="hello"),
            BrainMemory(tmp_path / "memory.sqlite3"),
        )


def test_mock_speech_to_text_and_text_to_speech(monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")

    stt = speech_to_text(mock_transcript="what should I clean first?")
    tts = text_to_speech("Not verified yet. First: clear the floor.")

    assert stt.transcript == "what should I clean first?"
    assert stt.mode == "mock"
    assert tts.mode == "mock"
    assert tts.audio_path is None


def test_empty_mock_transcript_uses_safe_fallback(monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")

    stt = speech_to_text(mock_transcript="   ")
    tts = text_to_speech("   ")

    assert stt.transcript == "what should I clean first?"
    assert tts.text == "No response available."


def test_audio_path_must_be_inside_safe_audio_directory(tmp_path) -> None:
    safe_dir = tmp_path / "audio"
    safe_dir.mkdir()
    approved = safe_dir / "clip.wav"
    approved.write_bytes(b"fake audio")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"fake audio")

    assert approved_audio_path(str(approved), audio_dir=safe_dir) == approved
    assert approved_audio_path(str(outside), audio_dir=safe_dir) is None


def test_voice_conversation_integrates_memory_and_maya(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(
            room_name="Office",
            zone_name="Desk",
            mock_camera_frame="notebooks, drink containers, and a loose cable",
        ),
        brain_memory,
    )

    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="what should I clean first?",
            room_name="Office",
            zone_name="Desk",
            response_mode="cleanup_coaching",
        ),
        brain_memory,
    )

    assert result.transcript == "what should I clean first?"
    assert result.memory_context.next_best_action == "clear hazard: loose cord on floor"
    assert result.maya_response.machine_output == result.machine_output
    assert result.speech_output.text == result.maya_response.user_response
    assert "loose cord on floor" in result.speech_output.text


def test_voice_conversation_keeps_command_json_separate_from_speech(tmp_path) -> None:
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    RobotMemory(brain_memory).remember(
        RememberRequest(text="The hallway is usually clear.", room_name="Hallway")
    )

    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="what do you know about the hallway?",
            room_name="Hallway",
        ),
        brain_memory,
    )

    assert isinstance(result.machine_output, dict)
    assert result.speech_output.text != str(result.machine_output)


def test_safety_still_overrides_voice_context(tmp_path) -> None:
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    RobotMemory(brain_memory).remember(
        RememberRequest(text="The hallway is usually clear.", room_name="Hallway")
    )
    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="move forward",
            room_name="Hallway",
        ),
        brain_memory,
    )
    cycle_input = BrainCycleInput(battery_percentage=80, distance_cm=5, user_command=result.transcript)
    intent = RobotIntent(
        requested_action="move_forward",
        requested_speed=0.25,
        reason="Voice conversation context suggests hallway is clear",
    )

    command = validate_action(cycle_input, intent)

    assert command.action == "stop"
    assert command.reason == "Obstacle too close"


def test_voice_route_exists() -> None:
    assert "/voice/conversation" in [route.path for route in app.routes]


def test_audio_directory_uses_env(tmp_path, monkeypatch) -> None:
    target = tmp_path / "audio"
    monkeypatch.setenv("GRIMBOT_AUDIO_DIR", str(target))

    assert audio_directory() == target
    assert target.exists()
