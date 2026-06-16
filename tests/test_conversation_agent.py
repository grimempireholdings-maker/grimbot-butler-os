from __future__ import annotations

import json

import grimbot_brain.conversation_agent as conversation_agent
from grimbot_brain.conversation import run_voice_conversation
from grimbot_brain.conversation_agent import (
    ApiConversationProvider,
    OpenRouterConversationProvider,
    build_conversation_prompt,
    provider_from_env,
    run_conversation_agent,
)
from grimbot_brain.conversation_schemas import ConversationalAgentResponse
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


def test_voice_conversation_returns_agent_response(tmp_path) -> None:
    result = _chat(tmp_path, "Hey Maya")

    assert result.agent_response is not None
    assert result.agent_response.user_response == result.speech_output.text
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


def test_auto_provider_uses_mock_without_keys(monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert provider_from_env().name == "mock"


def test_auto_provider_prefers_claude_when_key_exists(monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    assert provider_from_env().name == "claude"


def test_openrouter_provider_selected_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", "openrouter")

    provider = provider_from_env()

    assert provider.name == "openrouter"


def test_missing_optional_provider_env_does_not_crash(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = _chat(tmp_path, "Hey Maya")

    assert result.agent_response is not None
    assert result.agent_response.intent == "casual_chat"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "gemini"
    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert result.speech_output.text


def test_openrouter_missing_key_falls_back_to_mock(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = _chat(tmp_path, "Hey Maya")

    assert result.agent_response is not None
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "openrouter"
    assert result.machine_output["provider_response"] == "fallback_to_mock"


class FakeApiProvider(ApiConversationProvider):
    def __init__(self, raw_text: str | Exception) -> None:
        super().__init__(
            name="openai",
            env_key="OPENAI_API_KEY",
            model_env_key="GRIMBOT_CONVERSATION_OPENAI_MODEL",
            default_model="test-model",
        )
        object.__setattr__(self, "raw_text", raw_text)

    def _call(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
        api_key: str,
    ) -> str:
        if isinstance(self.raw_text, Exception):
            raise self.raw_text
        return self.raw_text


def test_valid_provider_json_can_only_replace_user_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = {
        "intent": "procedure_request",
        "user_response": "Hey Boss. Cleaner wording from the provider, no motor drama.",
        "confidence": 1,
        "retrieved_context": [{"unsafe": "ignored"}],
        "suggested_skill": None,
        "suggested_procedure": None,
        "machine_output": {"procedure_execution": "execute_now"},
        "verified": True,
    }

    result = run_conversation_agent(
        request=VoiceConversationRequest(push_to_talk=True, mock_transcript="Can you scan the room?"),
        transcript="Can you scan the room?",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(json.dumps(raw)),
    )

    assert result.intent == "room_or_physical_request"
    assert result.user_response == "Hey Boss. Cleaner wording from the provider, no motor drama."
    assert result.machine_output["conversation_provider"] == "openai"
    assert result.machine_output["provider_response"] == "validated"
    assert result.machine_output["procedure_execution"] == "not_used"
    assert result.machine_output["external_tools"] == "not_used"
    assert result.machine_output["hardware_control"] == "not_used"
    assert result.verified is False


def test_openrouter_valid_response_only_replaces_user_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/auto")
    raw = {
        "intent": "procedure_request",
        "user_response": "Boss, OpenRouter wording is live; no procedures are being run.",
        "confidence": 1,
        "retrieved_context": [{"unsafe": "ignored"}],
        "suggested_skill": None,
        "suggested_procedure": None,
        "machine_output": {"procedure_execution": "execute_now"},
        "verified": True,
    }

    result = run_conversation_agent(
        request=VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Run the morning workflow procedure.",
        ),
        transcript="Run the morning workflow procedure.",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeOpenRouterProvider(json.dumps(raw)),
    )

    assert result.intent == "procedure_request"
    assert result.user_response == "Boss, OpenRouter wording is live; no procedures are being run."
    assert result.machine_output["conversation_provider"] == "openrouter"
    assert result.machine_output["provider_response"] == "validated"
    assert result.machine_output["procedure_execution"] == "not_available"
    assert result.suggested_procedure is None
    assert result.verified is False


def test_openrouter_call_uses_required_endpoint_headers_and_model(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "intent": "casual_chat",
                                "user_response": "Provider text.",
                                "confidence": 0.8,
                                "retrieved_context": [],
                                "suggested_skill": None,
                                "suggested_procedure": None,
                                "machine_output": {},
                                "verified": False,
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(conversation_agent, "_post_json", fake_post_json)
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/auto")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.test")
    provider = OpenRouterConversationProvider(
        "openrouter",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "openrouter/auto",
    )
    fallback = ConversationalAgentResponse(
        intent="casual_chat",
        user_response="Fallback text.",
        confidence=0.8,
        retrieved_context=[],
        suggested_skill=None,
        suggested_procedure=None,
        machine_output={"conversation_provider": "openrouter"},
        verified=False,
    )

    raw_text = provider._call("prompt", fallback, "test-key")

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["HTTP-Referer"] == "https://example.test"
    assert captured["headers"]["X-OpenRouter-Title"] == "GrimBot Butler OS"
    assert captured["payload"]["model"] == "openrouter/auto"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert json.loads(raw_text)["user_response"] == "Provider text."


def test_provider_output_cannot_change_suggestions_or_safety_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = {
        "intent": "casual_chat",
        "user_response": "I can suggest the memory review skill, but I will not run it.",
        "confidence": 1,
        "retrieved_context": [{"unsafe": "ignored"}],
        "suggested_skill": {
            "name": "external_email_tool",
            "confidence": 1,
            "required_permission": "execute",
            "reason": "unsafe provider mutation",
        },
        "suggested_procedure": {
            "name": "motor_control",
            "confidence": 1,
            "required_permission": "execute",
            "reason": "unsafe provider mutation",
        },
        "machine_output": {
            "external_tools": "used",
            "hardware_control": "used",
            "procedure_execution": "executed",
        },
        "verified": True,
    }

    result = run_conversation_agent(
        request=VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Can you suggest a memory skill?",
        ),
        transcript="Can you suggest a memory skill?",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(json.dumps(raw)),
    )

    assert result.intent == "skill_request"
    assert result.suggested_skill is not None
    assert result.suggested_skill.name == "memory_review"
    assert result.suggested_skill.required_permission == "observe"
    assert result.suggested_procedure is None
    assert result.machine_output["external_tools"] == "not_used"
    assert result.machine_output["hardware_control"] == "not_used"
    assert result.machine_output["procedure_execution"] == "not_used"
    assert result.verified is False


def test_provider_text_implying_execution_falls_back_to_mock(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = {
        "intent": "procedure_request",
        "user_response": "I ran the procedure and moved the robot.",
        "confidence": 1,
        "retrieved_context": [],
        "suggested_skill": None,
        "suggested_procedure": None,
        "machine_output": {},
        "verified": False,
    }

    result = run_conversation_agent(
        request=VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Run the morning workflow procedure.",
        ),
        transcript="Run the morning workflow procedure.",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(json.dumps(raw)),
    )

    assert result.intent == "procedure_request"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "openai"
    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert "I ran the procedure" not in result.user_response
    assert result.machine_output["procedure_execution"] == "not_available"


def test_invalid_provider_json_falls_back_safely(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = run_conversation_agent(
        request=VoiceConversationRequest(push_to_talk=True, mock_transcript="Hey Maya"),
        transcript="Hey Maya",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider("not json"),
    )

    assert result.intent == "casual_chat"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "openai"
    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert "provider_fallback_reason" in result.machine_output
    assert "not json" not in result.user_response.lower()


def test_openrouter_invalid_response_falls_back_safely(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = run_conversation_agent(
        request=VoiceConversationRequest(push_to_talk=True, mock_transcript="Hey Maya"),
        transcript="Hey Maya",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeOpenRouterProvider("not json"),
    )

    assert result.intent == "casual_chat"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "openrouter"
    assert result.machine_output["provider_response"] == "fallback_to_mock"


class FakeOpenRouterProvider(OpenRouterConversationProvider):
    def __init__(self, raw_text: str | Exception) -> None:
        super().__init__(
            name="openrouter",
            env_key="OPENROUTER_API_KEY",
            model_env_key="OPENROUTER_MODEL",
            default_model="openrouter/auto",
        )
        object.__setattr__(self, "raw_text", raw_text)

    def _call(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
        api_key: str,
    ) -> str:
        if isinstance(self.raw_text, Exception):
            raise self.raw_text
        return self.raw_text


def test_provider_http_failure_falls_back_safely(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = run_conversation_agent(
        request=VoiceConversationRequest(push_to_talk=True, mock_transcript="Hey Maya"),
        transcript="Hey Maya",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(RuntimeError("network down")),
    )

    assert result.intent == "casual_chat"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "openai"
    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert "network down" in result.machine_output["provider_fallback_reason"]


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
