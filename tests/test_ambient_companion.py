from __future__ import annotations

import json
import sqlite3

import grimbot_brain.conversation_agent as conversation_agent
from grimbot_brain.conversation_agent import ApiConversationProvider, run_conversation_agent
from grimbot_brain.conversation_schemas import ConversationalAgentResponse
from grimbot_brain.memory import BrainMemory
from grimbot_brain.schemas import VoiceConversationRequest
from grimbot_brain.web_search import SearchItem, SearchResult


class AmbientTestProvider(ApiConversationProvider):
    def __init__(self, mode: str, wording: str | None = None) -> None:
        super().__init__("test", "TEST_KEY", "TEST_MODEL", "test-model")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "wording", wording)

    def classify_mode(self, prompt: str, timeout: float = 3.0) -> str:
        return json.dumps({"mode": self.mode, "needs_web_search": False, "search_query": None})

    def generate(self, prompt: str, fallback_response: ConversationalAgentResponse):
        if self.wording is None:
            return fallback_response
        parsed = fallback_response.model_copy(update={"user_response": self.wording})
        return conversation_agent._safe_provider_response(fallback_response, parsed, self.name)


def _run(tmp_path, text: str, provider, *, ambient_mode: bool = True):
    request = VoiceConversationRequest(
        push_to_talk=True,
        mock_transcript=text,
        ambient_mode=ambient_mode,
    )
    return run_conversation_agent(
        request=request,
        transcript=text,
        memory=BrainMemory(tmp_path / "ambient.sqlite3"),
        provider=provider,
    )


def test_morning_ramp_is_human_first_and_uses_plain_context(tmp_path) -> None:
    result = _run(tmp_path, "Good morning Maya. I'm groggy.", AmbientTestProvider("morning_ramp"))

    assert result.machine_output["conversation_mode"] == "morning_ramp"
    assert result.machine_output["calendar_access"] is False
    assert result.machine_output["approval_execution"] == "not_available"
    assert "sprint" in result.user_response.lower()
    assert "adaptive state" not in result.user_response.lower()


def test_only_morning_ramp_gets_proactive_cached_weather_hook(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return SearchResult(
            query=query,
            success=True,
            cached=False,
            results=[SearchItem(title="Local Weather", url="https://weather.example/today")],
            answer="A mild morning is expected.",
            days=1,
        )

    monkeypatch.setattr(conversation_agent, "search_web", fake_search)
    morning = _run(tmp_path, "Morning Maya", AmbientTestProvider("morning_ramp"))
    _run(tmp_path, "Just hanging out", AmbientTestProvider("casual_presence"))

    assert len(calls) == 1
    assert calls[0][0] == "today's weather forecast for Lima, Ohio"
    assert calls[0][1]["days"] == 1
    assert morning.machine_output["search_trigger_reason"] == "proactive_morning_weather"
    assert morning.machine_output["proactive_search"] is True
    assert "Sources:" in morning.user_response


def test_ambient_toggle_disables_new_modes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conversation_agent, "search_web", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    result = _run(
        tmp_path,
        "Morning Maya",
        AmbientTestProvider("morning_ramp"),
        ambient_mode=False,
    )

    assert result.machine_output["conversation_mode"] == "morning_orientation"
    assert result.machine_output["search_triggered"] is False


def test_internal_labels_are_rejected_in_ordinary_conversation(tmp_path) -> None:
    result = _run(
        tmp_path,
        "Morning Maya",
        AmbientTestProvider("morning_ramp", "My adaptive state and classifier flagged morning_ramp."),
    )

    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert "adaptive state" not in result.user_response.lower()


def test_direct_architecture_question_may_explain_internals(tmp_path) -> None:
    result = _run(
        tmp_path,
        "How do you work?",
        AmbientTestProvider("capability_question", "My adaptive state guides tone, while classification routes the turn."),
    )

    assert result.machine_output["provider_response"] == "validated"
    assert "adaptive state" in result.user_response.lower()


def test_ambient_context_reads_recent_memories(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    memory.log_episode("conversation", "Julian wanted a slower morning.", 0.8)
    context = conversation_agent.build_ambient_context(memory)

    assert context.recent_notes == ["Julian wanted a slower morning."]
    assert context.calendar_available is False


def test_ambient_context_does_not_initialize_or_mutate_adaptive_state(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "read-only.sqlite3")
    with sqlite3.connect(memory.db_path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM adaptive_state_signals").fetchone()[0]

    conversation_agent.build_ambient_context(memory)

    with sqlite3.connect(memory.db_path) as connection:
        after = connection.execute("SELECT COUNT(*) FROM adaptive_state_signals").fetchone()[0]
    assert before == after == 0
