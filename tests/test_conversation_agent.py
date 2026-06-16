from __future__ import annotations

from grimbot_brain.conversation import run_voice_conversation
from grimbot_brain.conversation_agent import build_conversation_prompt, provider_from_env
from grimbot_brain.memory import BrainMemory
from grimbot_brain.schemas import VoiceConversationRequest


def _chat(tmp_path, text: str):
    return run_voice_conversation(
        VoiceConversationRequest(push_to_talk=True, mock_transcript=text),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )


def test_casual_greeting_does_not_return_room_scan(tmp_path) -> None:
    result = _chat(tmp_path, "Hey Maya, how's it going?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "casual_chat"
    assert result.machine_output["room_scan_requested"] is False
    assert "scan room" not in result.speech_output.text.lower()


def test_casual_greeting_sounds_natural_and_non_template(tmp_path) -> None:
    result = _chat(
        tmp_path,
        "Hey Maya, how's it going? Ready for another riveting day at Grim Empire Holdings LLC?",
    )
    text = result.speech_output.text

    assert text.startswith("Boss, always.")
    assert not text.startswith("Not verified yet")
    assert "Here is the signal" not in text
    assert text.count("Boss") <= 1


def test_day_question_routes_to_briefing_intent(tmp_path) -> None:
    result = _chat(tmp_path, "How's my day looking?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "chief_of_staff_briefing"
    assert result.machine_output["active_projects"]
    assert "scan room" not in result.speech_output.text.lower()


def test_work_today_question_routes_to_briefing_not_room_scan(tmp_path) -> None:
    result = _chat(tmp_path, "What should I work on today?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "chief_of_staff_briefing"
    assert result.machine_output["active_projects"]
    assert "scan room" not in result.speech_output.text.lower()
    assert "clean" not in result.speech_output.text.lower()


def test_grimbot_question_routes_to_project_recall(tmp_path) -> None:
    result = _chat(tmp_path, "What do you remember about GrimBot?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "project_recall"
    assert result.machine_output["projects"][0]["name"] == "GrimBot Butler OS"
    assert "GrimBot Butler OS" in result.speech_output.text


def test_casual_detector_does_not_match_hi_inside_other_words(tmp_path) -> None:
    result = _chat(tmp_path, "This is vague.")

    assert result.agent_response is not None
    assert result.agent_response.intent == "unclear"
    assert result.machine_output["needs_clarification"] is True


def test_physical_room_request_routes_to_physical_intent(tmp_path) -> None:
    result = _chat(tmp_path, "Can you scan the room for hazards?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "room_or_physical_request"
    assert result.machine_output["conversation_intent"] == "room_or_physical_request"


def test_physical_context_does_not_claim_verified_from_request_flag(tmp_path) -> None:
    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Can you scan the room?",
            verified=True,
        ),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.agent_response is not None
    assert result.agent_response.intent == "room_or_physical_request"
    assert result.agent_response.verified is False
    assert result.maya_response.verified is False


def test_unclear_request_asks_one_clarifying_question(tmp_path) -> None:
    result = _chat(tmp_path, "Zephyr?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "unclear"
    assert result.machine_output["needs_clarification"] is True
    assert result.speech_output.text.count("?") == 1


def test_response_shape_is_consistent(tmp_path) -> None:
    result = _chat(tmp_path, "How's my day looking?")

    assert result.agent_response is not None
    assert result.speech_output.text == result.agent_response.user_response
    assert result.maya_response.user_response == result.agent_response.user_response
    assert result.maya_response.machine_output == result.machine_output
    assert result.agent_response.machine_output == result.machine_output


def test_machine_output_remains_separate(tmp_path) -> None:
    result = _chat(tmp_path, "Hey Maya")

    assert result.agent_response is not None
    assert isinstance(result.agent_response.machine_output, dict)
    assert result.speech_output.text != str(result.agent_response.machine_output)
    assert result.maya_response.machine_output == result.machine_output


def test_no_procedure_execution_external_tools_or_hardware(tmp_path) -> None:
    result = _chat(tmp_path, "Run the morning workflow procedure.")

    assert result.agent_response is not None
    assert result.agent_response.intent == "procedure_request"
    assert result.machine_output["procedure_execution"] == "not_available"
    assert result.machine_output["external_tools"] == "not_used"
    assert result.machine_output["hardware_control"] == "not_used"


def test_mock_provider_is_default(monkeypatch) -> None:
    monkeypatch.delenv("GRIMBOT_CONVERSATION_PROVIDER", raising=False)

    assert provider_from_env().name == "mock"


def test_missing_optional_provider_env_does_not_crash(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = _chat(tmp_path, "Hey Maya")

    assert result.agent_response is not None
    assert result.agent_response.intent == "casual_chat"
    assert result.machine_output["conversation_provider"] == "gemini"
    assert result.speech_output.text


def test_conversation_prompt_keeps_safety_boundaries() -> None:
    prompt = build_conversation_prompt(
        transcript="Hey Maya",
        intent="casual_chat",
        retrieved_context=[],
        machine_output={"conversation_mode": "casual"},
    )

    assert "Keep machine_output separate" in prompt
    assert "no motors" in prompt
    assert "Never start with a disclaimer" in prompt
