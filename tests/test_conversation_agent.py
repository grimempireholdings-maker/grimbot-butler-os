from __future__ import annotations

import json

import grimbot_brain.conversation_agent as conversation_agent
import pytest
from grimbot_brain.conversation import run_voice_conversation
from grimbot_brain.conversation_agent import (
    ApiConversationProvider,
    OpenRouterConversationProvider,
    build_conversation_prompt,
    classify_conversation_mode,
    provider_from_env,
    run_conversation_agent,
)
from grimbot_brain.conversation_retrieval import MAX_RETRIEVAL_QUERY_LENGTH, build_retrieval_query
from grimbot_brain.conversation_schemas import ConversationalAgentResponse
from grimbot_brain.memory import BrainMemory
from grimbot_brain.schemas import VoiceConversationRequest


def _chat(tmp_path, text: str):
    return run_voice_conversation(
        VoiceConversationRequest(push_to_talk=True, mock_transcript=text),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )


def test_casual_greeting_does_not_return_room_scan(tmp_path) -> None:
    result = _chat(tmp_path, "Hey Maya")

    assert result.agent_response is not None
    assert result.agent_response.intent == "casual_chat"
    assert result.machine_output["room_scan_requested"] is False
    assert "scan room" not in result.speech_output.text.lower()


def test_casual_greeting_sounds_natural_and_non_template(tmp_path) -> None:
    result = _chat(
        tmp_path,
        "Hey Maya, ready for another riveting day at Grim Empire Holdings LLC?",
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


def test_long_casual_message_does_not_crash_or_start_room_scan(tmp_path) -> None:
    long_message = (
        ("haha idk " * 45)
        + "Maya this is mostly me rambling through a casual check-in before we pick a lane. "
        + ("etc whatever " * 30)
    )

    result = _chat(tmp_path, long_message)

    assert result.agent_response is not None
    assert result.speech_output.text
    assert "String should have at most" not in result.speech_output.text
    assert "ValidationError" not in result.speech_output.text
    assert "traceback" not in result.speech_output.text.lower()
    assert result.agent_response.intent != "room_or_physical_request"
    assert result.machine_output.get("room_scan_requested") is not True
    assert len(result.machine_output["retrieval"]["query"]) <= MAX_RETRIEVAL_QUERY_LENGTH


def test_long_project_message_builds_short_project_context_query() -> None:
    long_message = (
        ("haha idk " * 35)
        + "I want Maya thinking clearly about GrimBot architecture, real estate, land flipping, "
        + "and Maya Console without clogging retrieval with the whole ramble. "
        + ("etc basically whatever " * 22)
    )

    retrieval_query = build_retrieval_query(long_message)

    assert len(retrieval_query.query) <= MAX_RETRIEVAL_QUERY_LENGTH
    assert "grimbot" in retrieval_query.query
    assert "real estate" in retrieval_query.query
    assert "maya" in retrieval_query.query
    assert "architecture" in retrieval_query.query
    assert "haha" not in retrieval_query.query
    assert "idk" not in retrieval_query.query
    assert "etc" not in retrieval_query.query


def test_core_semantic_anchors_survive_filler_stripping() -> None:
    anchors = [
        "GrimBot",
        "Maya",
        "real estate",
        "JARVIS",
        "Optimus",
        "architecture",
        "OpenClaw",
        "Codex",
        "procedure",
        "memory",
        "dreaming",
        "body",
        "robot",
    ]
    long_message = (
        ("haha idk etc whatever " * 25)
        + " ".join(anchors)
        + (" basically literally just stuff " * 25)
    )

    retrieval_query = build_retrieval_query(long_message)
    query = retrieval_query.query

    assert len(query) <= MAX_RETRIEVAL_QUERY_LENGTH
    for anchor in anchors:
        assert anchor.lower() in query
    assert "haha" not in query
    assert "idk" not in query
    assert "etc" not in query


def test_memory_retrieval_failure_degrades_without_leaking_validation_error(tmp_path, monkeypatch) -> None:
    def fail_relevant(*args, **kwargs):
        raise ValueError("RelevantMemoryRequest query String should have at most 500 characters")

    monkeypatch.setattr("grimbot_brain.robot_memory.RobotMemory.relevant", fail_relevant)

    result = _chat(
        tmp_path,
        ("haha " * 80) + "Maya, give me the useful signal on GrimBot and real estate priorities.",
    )

    assert result.agent_response is not None
    assert result.speech_output.text
    assert "String should have at most" not in result.speech_output.text
    assert "RelevantMemoryRequest" not in result.speech_output.text
    assert "ValidationError" not in result.speech_output.text
    assert result.machine_output["retrieval_status"] == "ok"
    assert "retrieval_errors" not in result.machine_output
    assert len(result.machine_output["retrieval"]["query"]) <= MAX_RETRIEVAL_QUERY_LENGTH


def test_context_retrieval_failure_degrades_without_leaking_traceback(tmp_path, monkeypatch) -> None:
    def fail_search(*args, **kwargs):
        raise RuntimeError("Traceback: ContextSearchRequest query String should have at most 500 characters")

    monkeypatch.setattr("grimbot_brain.identity.context_store.ContextStore.search", fail_search)

    result = _chat(
        tmp_path,
        ("haha idk " * 50) + "Maya, what do you remember about GrimBot architecture and Codex?",
    )

    assert result.agent_response is not None
    assert result.speech_output.text
    assert "Traceback" not in result.speech_output.text
    assert "ContextSearchRequest" not in result.speech_output.text
    assert "String should have at most" not in result.speech_output.text
    assert result.machine_output["retrieval_status"] == "fallback"
    assert result.machine_output["retrieval_errors"]["context"]["status"] == "fallback"
    assert len(result.machine_output["retrieval"]["query"]) <= MAX_RETRIEVAL_QUERY_LENGTH


class CapturingProvider:
    name = "mock"

    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str, fallback_response: ConversationalAgentResponse) -> ConversationalAgentResponse:
        self.prompt = prompt
        return fallback_response


def test_full_transcript_reaches_provider_while_retrieval_uses_short_query(tmp_path) -> None:
    transcript = (
        ("haha idk " * 40)
        + "Maya, keep the full messy human context about GrimBot, OpenClaw, Codex, "
        + "Optimus, JARVIS, real estate, architecture, body, robot, memory, dreaming, "
        + "and procedure work available to the conversation provider."
        + (" etc whatever " * 18)
    )
    provider = CapturingProvider()

    result = run_conversation_agent(
        request=VoiceConversationRequest(push_to_talk=True, mock_transcript=transcript),
        transcript=transcript,
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=provider,
    )

    assert transcript in provider.prompt
    assert len(result.machine_output["retrieval"]["query"]) <= MAX_RETRIEVAL_QUERY_LENGTH
    assert len(result.machine_output["retrieval"]["query"]) < len(transcript)
    assert "OpenClaw" in provider.prompt
    assert "openclaw" in result.machine_output["retrieval"]["query"]


def test_grimbot_question_routes_to_project_recall(tmp_path) -> None:
    result = _chat(tmp_path, "What do you remember about GrimBot?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "project_recall"
    assert result.machine_output["projects"][0]["name"] == "GrimBot Butler OS"
    assert "GrimBot Butler OS" in result.speech_output.text
    assert "bottleneck" in result.speech_output.text.lower()
    assert "next" in result.speech_output.text.lower()
    assert "which project" not in result.speech_output.text.lower()


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


def test_digital_room_routes_to_read_only_workspace_awareness(tmp_path) -> None:
    result = _chat(tmp_path, "Can you look around your digital room?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "workspace_awareness"
    assert result.machine_output["workspace_access"] == "read_only"
    assert result.machine_output["physical_vision"] == "not_active"
    assert "read-only" in result.speech_output.text.lower()
    assert "physical room" in result.speech_output.text.lower()
    assert result.machine_output["external_tools"] == "not_used"
    assert result.machine_output["hardware_control"] == "not_used"


@pytest.mark.parametrize(
    "message",
    [
        "Can you look around your digital room?",
        "What do you know about your repo?",
        "What can you see around you?",
        "What do you know about your architecture?",
        "What branch are you on?",
        "What changed recently?",
    ],
)
def test_workspace_intent_phrase_matrix(tmp_path, message) -> None:
    result = _chat(tmp_path, message)

    assert result.agent_response is not None
    assert result.agent_response.intent == "workspace_awareness"
    assert result.machine_output["workspace_access"] == "read_only"
    assert result.machine_output["physical_vision"] == "not_active"
    assert "read-only" in result.agent_response.user_response.lower()
    assert "physical" in result.agent_response.user_response.lower()


def test_camera_question_does_not_claim_physical_vision(tmp_path) -> None:
    result = _chat(tmp_path, "Can you see through the camera?")

    assert result.agent_response is not None
    assert result.agent_response.intent == "room_or_physical_request"
    assert result.machine_output["camera_access"] is False
    assert result.machine_output["vision_invoked"] is False
    text = result.speech_output.text.lower()
    assert "do not have camera access yet" in text or "don’t have camera access yet" in text
    assert "share the feed" not in text


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
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
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


@pytest.mark.parametrize(
    "provider_text",
    [
        "Read-only digital workspace access is active, and I can see the physical room.",
        "Read-only digital workspace inspection is separate from physical vision, and I modified the file.",
    ],
)
def test_provider_cannot_remove_workspace_or_physical_vision_boundary(
    tmp_path,
    monkeypatch,
    provider_text,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = {
        "intent": "workspace_awareness",
        "user_response": provider_text,
        "confidence": 1,
        "retrieved_context": [],
        "suggested_skill": None,
        "suggested_procedure": None,
        "machine_output": {},
        "verified": True,
    }

    result = run_conversation_agent(
        request=VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Look around your digital room.",
        ),
        transcript="Look around your digital room.",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(json.dumps(raw)),
    )

    assert result.intent == "workspace_awareness"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "openai"
    assert result.machine_output["workspace_access"] == "read_only"
    assert "read-only" in result.user_response.lower()
    assert "cannot see the physical room" in result.user_response.lower()


def test_provider_cannot_turn_camera_denial_into_camera_claim(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = {
        "intent": "room_or_physical_request",
        "user_response": "Yes, the camera feed is active and I can inspect the room.",
        "confidence": 1,
        "retrieved_context": [],
        "suggested_skill": None,
        "suggested_procedure": None,
        "machine_output": {},
        "verified": True,
    }

    result = run_conversation_agent(
        request=VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Can you see through the camera?",
        ),
        transcript="Can you see through the camera?",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(json.dumps(raw)),
    )

    assert result.intent == "room_or_physical_request"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["camera_access"] is False
    assert "do not have camera access yet" in result.user_response.lower()


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
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
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


def test_provider_text_implying_hardware_activation_falls_back_to_mock(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = {
        "intent": "casual_chat",
        "user_response": "Motor engaged. I activated the hardware.",
        "confidence": 1,
        "retrieved_context": [],
        "suggested_skill": None,
        "suggested_procedure": None,
        "machine_output": {},
        "verified": False,
    }

    result = run_conversation_agent(
        request=VoiceConversationRequest(push_to_talk=True, mock_transcript="Hey Maya"),
        transcript="Hey Maya",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(json.dumps(raw)),
    )

    assert result.intent == "casual_chat"
    assert result.machine_output["conversation_provider"] == "mock"
    assert result.machine_output["provider_attempted"] == "openai"
    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert "Motor engaged" not in result.user_response
    assert result.machine_output["hardware_control"] == "not_used"


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


def test_capabilities_manifest_is_in_every_provider_prompt() -> None:
    prompt = build_conversation_prompt(
        transcript="Morning Maya",
        intent="casual_chat",
        retrieved_context=[],
        machine_output={},
        conversation_mode="morning_orientation",
    )

    assert "CAPABILITIES manifest (verbatim)" in prompt
    assert '"has_camera_access": false' in prompt
    assert '"has_workspace_read_access": true' in prompt
    assert "You may ONLY claim awareness or capability" in prompt


def test_digital_room_uses_workspace_only_without_unsupported_awareness(tmp_path) -> None:
    result = _chat(tmp_path, "What can you see in your digital room?")
    text = result.speech_output.text.lower()

    assert result.machine_output["conversation_mode"] == "workspace_awareness"
    assert "local repo/workspace, read-only" in text
    for forbidden in ("devices", "device layout", "browser tabs", "camera", "microphone", "pending updates"):
        assert forbidden not in text


def test_feed_sharing_capability_claim_triggers_safe_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    raw = {
        "intent": "room_or_physical_request",
        "user_response": "Sure, share the feed and I can check what's visible through the camera.",
        "confidence": 1,
        "retrieved_context": [],
        "suggested_skill": None,
        "suggested_procedure": None,
        "machine_output": {},
        "verified": True,
    }
    result = run_conversation_agent(
        request=VoiceConversationRequest(push_to_talk=True, mock_transcript="Can you use my camera?"),
        transcript="Can you use my camera?",
        memory=BrainMemory(tmp_path / "memory.sqlite3"),
        provider=FakeApiProvider(json.dumps(raw)),
    )

    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert result.machine_output["provider_fallback_reason"]
    assert "do not have camera access yet" in result.user_response.lower()


def test_morning_orientation_is_broad_and_not_real_estate_default(tmp_path) -> None:
    result = _chat(tmp_path, "Morning Maya")

    assert result.machine_output["conversation_mode"] == "morning_orientation"
    assert result.machine_output["orientation_scope"] == "broad"
    assert len(result.machine_output["active_projects"]) >= 2
    assert result.machine_output.get("recommended_focus") is None


def test_interesting_today_orients_across_multiple_lanes(tmp_path) -> None:
    result = _chat(tmp_path, "Anything interesting happening today?")

    assert result.machine_output["conversation_mode"] == "morning_orientation"
    assert len(result.machine_output["active_projects"]) >= 2
    assert "active lanes" in result.speech_output.text.lower()


def test_hyperfocus_feedback_classifies_and_adjusts_now(tmp_path) -> None:
    result = _chat(tmp_path, "You keep hyperfocusing on real estate and it feels too scripted.")
    text = result.speech_output.text.lower()

    assert classify_conversation_mode("You keep hyperfocusing on real estate") == "feedback_about_maya"
    assert result.machine_output["conversation_mode"] == "feedback_about_maya"
    assert result.machine_output["behavior_adjusted_now"] is True
    assert "which project or lane" not in text


def test_reexplaining_feedback_does_not_repeat_generic_clarifier(tmp_path) -> None:
    result = _chat(tmp_path, "I already explained what I meant; stop asking me to pick a lane.")

    assert result.machine_output["conversation_mode"] == "feedback_about_maya"
    assert "which project or lane" not in result.speech_output.text.lower()
    assert "strategy, memory, skills" not in result.speech_output.text.lower()


def test_work_focus_rotates_recommendation_across_turns(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    request = VoiceConversationRequest(push_to_talk=True, mock_transcript="What should I work on today?")

    first = run_voice_conversation(request, memory)
    second = run_voice_conversation(request, memory)

    assert first.machine_output["recommended_focus"]
    assert second.machine_output["recommended_focus"]
    assert first.machine_output["recommended_focus"] != second.machine_output["recommended_focus"]


def test_work_today_includes_more_than_one_active_priority(tmp_path) -> None:
    result = _chat(tmp_path, "What should I work on today?")

    assert result.machine_output["conversation_mode"] == "work_focus"
    assert len(result.machine_output["active_projects"]) >= 2
