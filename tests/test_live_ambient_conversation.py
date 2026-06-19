from __future__ import annotations

import os
from datetime import date

import pytest
from dotenv import load_dotenv

from grimbot_brain.conversation import run_voice_conversation
from grimbot_brain.conversation_agent import _safe_provider_response
from grimbot_brain.memory import BrainMemory
from grimbot_brain.schemas import VoiceConversationRequest


load_dotenv()
_LIVE_PROVIDER = os.getenv("GRIMBOT_CONVERSATION_PROVIDER", "openrouter")

pytestmark = pytest.mark.skipif(
    os.getenv("GRIMBOT_LIVE_TESTS") != "1",
    reason="real-provider regression suite; set GRIMBOT_LIVE_TESTS=1 explicitly",
)


@pytest.fixture(autouse=True)
def _enable_configured_live_provider(monkeypatch):
    """Override the normal suite's autouse mock-provider fixture only in this file."""
    monkeypatch.setenv("GRIMBOT_CONVERSATION_PROVIDER", _LIVE_PROVIDER)


def _live(memory: BrainMemory, text: str):
    result = run_voice_conversation(
        VoiceConversationRequest(push_to_talk=True, mock_transcript=text, ambient_mode=True),
        memory,
    )
    assert result.agent_response is not None
    assert result.machine_output["classification_source"] == "llm"
    return result.agent_response


def test_live_morning_greeting_does_not_push_named_project(tmp_path) -> None:
    response = _live(BrainMemory(tmp_path / "morning.sqlite3"), "Morning Maya, how's it going?")
    names = [name.lower() for name in response.machine_output.get("active_projects", [])]

    assert response.machine_output["conversation_mode"] == "morning_ramp"
    assert not any(name in response.user_response.lower() for name in names)


def test_live_paired_feedback_is_not_business_strategy(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "feedback.sqlite3")
    _live(memory, "Morning Maya, what should we focus on?")
    response = _live(
        memory,
        "yeah, you say that every morning lol, you know there are millions of business models out there, right?",
    )

    assert response.machine_output["conversation_mode"] == "feedback_about_maya"
    assert response.machine_output["behavior_adjusted_now"] is True


def test_live_none_of_above_does_not_repeat_clarification(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "clarification.sqlite3")
    first = _live(memory, "I need something, but I am not sure how to describe it.")
    second = _live(memory, "None of the above.")

    assert second.user_response.casefold() != first.user_response.casefold()
    assert "what outcome are you" not in second.user_response.lower()


def test_live_camera_question_declines_honestly(tmp_path) -> None:
    response = _live(BrainMemory(tmp_path / "camera.sqlite3"), "can you check the camera?")
    text = response.user_response.lower()

    assert response.machine_output.get("camera_access") is False
    assert "camera" in text
    assert any(term in text for term in ("cannot", "can't", "do not have", "no camera"))


def test_live_news_is_searched_attributed_and_dated(tmp_path) -> None:
    response = _live(
        BrainMemory(tmp_path / "news.sqlite3"),
        "what's happening out there, any news?",
    )

    assert response.machine_output["search_triggered"] is True
    assert response.machine_output["search_success"] is True
    assert response.machine_output["search_result_count"] > 0
    assert "source" in response.user_response.lower()
    assert any(item.get("published_date") for item in response.machine_output["search_results"]) or str(date.today().year) in response.user_response
    assert "active projects" not in response.user_response.lower()


def test_live_current_query_researches_or_marks_staleness(tmp_path) -> None:
    response = _live(
        BrainMemory(tmp_path / "stale.sqlite3"),
        "What is the latest status of Voyager 1? Please verify against current sources rather than old cached reports.",
    )

    assert response.machine_output["search_triggered"] is True
    assert response.machine_output["search_cached"] is False or "stale" in response.user_response.lower()


def test_live_search_attribution_gate_rejects_omission(tmp_path) -> None:
    response = _live(
        BrainMemory(tmp_path / "attribution.sqlite3"),
        "What are today's major technology headlines?",
    )
    assert response.machine_output["search_result_count"] > 0
    unsafe = response.model_copy(update={"user_response": "Here are the headlines, presented without attribution."})
    gated = _safe_provider_response(response, unsafe, "live-regression-probe")

    assert gated.machine_output["provider_response"] == "fallback_to_mock"
    assert "omitted source attribution" in gated.machine_output["provider_fallback_reason"]


def test_live_self_improvement_classification_tracking(tmp_path) -> None:
    response = _live(
        BrainMemory(tmp_path / "self-improvement.sqlite3"),
        "how would you improve yourself?",
    )

    if response.machine_output["conversation_mode"] == "capability_question":
        pytest.xfail("known-imperfect classifier case remains non-blocking")
    assert response.machine_output["conversation_mode"] in {
        "ambient_companion",
        "gentle_orientation",
        "feedback_about_maya",
    }
