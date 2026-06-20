from __future__ import annotations

import json
import sqlite3

import grimbot_brain.conversation_agent as conversation_agent
import grimbot_brain.web_search as web_search
import pytest
from grimbot_brain.capabilities import capabilities_manifest
from grimbot_brain.conversation_agent import (
    _build_classification_prompt,
    _parse_classification,
    _safe_provider_response,
    run_conversation_agent,
)
from grimbot_brain.conversation_schemas import ConversationClassification, ConversationalAgentResponse
from grimbot_brain.memory import BrainMemory
from grimbot_brain.schemas import VoiceConversationRequest
from grimbot_brain.web_search import SearchItem, SearchResult, clear_search_cache, search_web


def _request(text: str) -> VoiceConversationRequest:
    return VoiceConversationRequest(push_to_talk=True, mock_transcript=text)


def _decision(mode: str, *, search: bool = False, query: str | None = None):
    return ConversationClassification(mode=mode, needs_web_search=search, search_query=query)


def test_search_cache_prevents_second_api_call_and_logs_each_use(tmp_path, monkeypatch) -> None:
    clear_search_cache()
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    calls: list[tuple[str, int, str]] = []

    def fake_request(
        query: str,
        max_results: int,
        api_key: str,
        topic: str,
        days: int | None,
    ) -> dict:
        calls.append((query, max_results, api_key))
        return {
            "answer": "A current summary.",
            "results": [
                {"title": "Source One", "url": "https://example.com/one", "content": "Live snippet."}
            ],
        }

    monkeypatch.setattr(web_search, "_request_search", fake_request)
    memory = BrainMemory(tmp_path / "memory.sqlite3")

    first = search_web("  Current   AI news  ", memory=memory)
    second = search_web("current ai NEWS", memory=memory)

    assert first.success is True
    assert second.success is True
    assert second.cached is True
    assert len(calls) == 1
    with sqlite3.connect(memory.db_path) as connection:
        rows = connection.execute(
            "SELECT content FROM episodic_memories WHERE kind = 'web_search' ORDER BY id"
        ).fetchall()
    assert len(rows) == 2
    assert json.loads(rows[-1][0])["cached"] is True


def test_search_timeout_returns_failure_instead_of_raising(tmp_path, monkeypatch) -> None:
    clear_search_cache()
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(web_search, "_request_search", lambda *args: (_ for _ in ()).throw(TimeoutError()))

    result = search_web("current local weather", memory=BrainMemory(tmp_path / "memory.sqlite3"))

    assert result.success is False
    assert result.results == []
    assert "timed out" in (result.reason or "").lower()


def test_tavily_request_uses_fixed_endpoint_bearer_auth_and_no_raw_content(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"query":"latest news","answer":"","results":[]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(web_search.urllib.request, "urlopen", fake_urlopen)
    payload = web_search._request_search("latest news", 5, "tvly-test")
    request = captured["request"]
    body = json.loads(request.data.decode("utf-8"))

    assert request.full_url == "https://api.tavily.com/search"
    assert request.get_header("Authorization") == "Bearer tvly-test"
    assert captured["timeout"] == 5.0
    assert body["include_raw_content"] is False
    assert body["search_depth"] == "basic"
    assert payload["results"] == []


def test_classifier_extracts_short_query_not_raw_transcript() -> None:
    raw_transcript = (
        "haha okay so I am rambling a little before the actual question " * 8
        + "what changed in AI agent frameworks this week"
    )
    raw = json.dumps(
        {
            "mode": "project_context",
            "needs_web_search": True,
            "search_query": raw_transcript,
        }
    )

    decision = _parse_classification(raw)

    assert decision.needs_web_search is True
    assert decision.search_query
    assert len(decision.search_query) <= 240
    assert decision.search_query != raw_transcript
    assert "agent" in decision.search_query.lower()


def test_classifier_prompt_distinguishes_external_world_from_workspace() -> None:
    prompt = _build_classification_prompt("what is happening out there, any news?", ())

    assert '"needs_web_search":true' in prompt
    assert "External-world/current-information requests are not workspace awareness" in prompt


def test_search_failure_produces_honest_conversation_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        conversation_agent,
        "classify_conversation_decision_with_fallback",
        lambda *args, **kwargs: (_decision("capability_question", search=True, query="current news"), "llm"),
    )
    monkeypatch.setattr(
        conversation_agent,
        "search_web",
        lambda *args, **kwargs: SearchResult(
            query="current news", success=False, reason="Tavily timed out after 5 seconds."
        ),
    )

    result = run_conversation_agent(
        _request("What's happening out there, any news?"),
        "What's happening out there, any news?",
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.machine_output["classification_source"] == "llm"
    assert result.machine_output["search_triggered"] is True
    assert result.machine_output["search_success"] is False
    assert result.machine_output["search_results"] == []
    assert "tried to search" in result.user_response.lower()
    assert "did not come back" in result.user_response.lower()
    assert "will not invent" in result.user_response.lower()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("casual", "Hey Maya, how are you?"),
        ("morning_orientation", "Morning Maya."),
        ("project_context", "What do you remember about GrimBot?"),
    ],
)
def test_non_search_decision_never_calls_search(tmp_path, monkeypatch, mode, message) -> None:
    monkeypatch.setattr(
        conversation_agent,
        "classify_conversation_decision_with_fallback",
        lambda *args, **kwargs: (_decision(mode), "llm"),
    )

    def unexpected_search(*args, **kwargs):
        raise AssertionError("non-search classification must not trigger web search")

    monkeypatch.setattr(conversation_agent, "search_web", unexpected_search)
    result = run_conversation_agent(
        _request(message),
        message,
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.machine_output["search_triggered"] is False
    assert result.machine_output["search_results"] == []


def test_live_news_request_uses_results_instead_of_capability_decline(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        conversation_agent,
        "classify_conversation_decision_with_fallback",
        lambda *args, **kwargs: (_decision("capability_question", search=True, query="latest major news"), "llm"),
    )
    monkeypatch.setattr(
        conversation_agent,
        "search_web",
        lambda *args, **kwargs: SearchResult(
            query="latest major news",
            success=True,
            answer="A concise live-news summary.",
            results=[
                SearchItem(
                    title="Example Newsroom",
                    url="https://example.com/news",
                    snippet="A current event summary from the search API.",
                )
            ],
        ),
    )

    result = run_conversation_agent(
        _request("What's happening out there, any news?"),
        "What's happening out there, any news?",
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.machine_output["search_triggered"] is True
    assert result.machine_output["search_success"] is True
    assert result.machine_output["search_results"][0]["title"] == "Example Newsroom"
    assert "Example Newsroom" in result.user_response
    assert "https://example.com/news" in result.user_response
    assert "do not have internet access" not in result.user_response.lower()


def test_implicit_local_weather_query_is_grounded_in_verified_profile_location(tmp_path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        conversation_agent,
        "classify_conversation_decision_with_fallback",
        lambda *args, **kwargs: (
            _decision("capability_question", search=True, query="current local weather"),
            "llm",
        ),
    )

    def fake_search(query, **kwargs):
        captured["query"] = query
        return SearchResult(query=query, success=True, answer="Clear skies.")

    monkeypatch.setattr(conversation_agent, "search_web", fake_search)
    result = run_conversation_agent(
        _request("What's the weather looking like today?"),
        "What's the weather looking like today?",
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert captured["query"] == "current local weather for Lima, Ohio"
    assert result.machine_output["search_query"] == "current local weather for Lima, Ohio"


def test_manifest_reports_bounded_web_search_capability() -> None:
    manifest = capabilities_manifest()

    assert manifest["has_web_search"] is True
    assert "no browsing" in manifest["web_search_scope"]
    assert manifest["has_real_time_market_data"] is False


def test_provider_search_summary_without_source_url_falls_back() -> None:
    fallback = ConversationalAgentResponse(
        intent="unclear",
        user_response="Summary\nSource: https://example.com/news",
        confidence=0.9,
        retrieved_context=[],
        machine_output={
            "search_triggered": True,
            "search_success": True,
            "search_results": [{"title": "Source", "url": "https://example.com/news", "snippet": "x"}],
        },
        verified=False,
    )
    parsed = fallback.model_copy(update={"user_response": "Here is the latest, without a citation."})

    result = _safe_provider_response(fallback, parsed, "openrouter")

    assert result.user_response == fallback.user_response
    assert result.machine_output["provider_response"] == "fallback_to_mock"
    assert "omitted source" in result.machine_output["provider_fallback_reason"]


def test_provider_cannot_hide_failed_search() -> None:
    fallback = ConversationalAgentResponse(
        intent="unclear",
        user_response="I tried to search, but it did not come back.",
        confidence=0.9,
        retrieved_context=[],
        machine_output={"search_triggered": True, "search_success": False, "search_results": []},
        verified=False,
    )
    parsed = fallback.model_copy(update={"user_response": "Here are today's definitely current headlines."})

    result = _safe_provider_response(fallback, parsed, "openrouter")

    assert result.user_response == fallback.user_response
    assert "failed search" in result.machine_output["provider_fallback_reason"]


def test_provider_truthful_failed_search_contraction_is_validated() -> None:
    fallback = ConversationalAgentResponse(
        intent="unclear",
        user_response="I tried to search, but it did not come back.",
        confidence=0.9,
        retrieved_context=[],
        machine_output={"search_triggered": True, "search_success": False, "search_results": []},
        verified=False,
    )
    truthful = "I tried the live search, but the API key isn't configured, so I won't invent an answer."
    parsed = fallback.model_copy(update={"user_response": truthful})

    result = _safe_provider_response(fallback, parsed, "openrouter")

    assert result.user_response == truthful
    assert result.machine_output["provider_response"] == "validated"
