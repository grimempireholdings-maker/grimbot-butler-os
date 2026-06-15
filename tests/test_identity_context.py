from __future__ import annotations

import sqlite3

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from grimbot_brain.conversation import run_voice_conversation
from grimbot_brain.identity.context_schemas import (
    ContextRememberRequest,
    ContextSearchRequest,
    PriorityUpdateRequest,
)
from grimbot_brain.identity.context_store import ContextStore
from grimbot_brain.maya_core import build_maya_briefing
from grimbot_brain.memory import BrainMemory
from grimbot_brain.robot_memory import RobotMemory
from grimbot_brain.room_scan import run_room_scan
from grimbot_brain.schemas import MayaBriefingRequest, RoomScanRequest, VoiceConversationRequest


def test_context_tables_initialize(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")

    with sqlite3.connect(memory.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {"identity_context", "identity_projects"} <= tables


def test_default_seed_loads_with_source_separation(tmp_path) -> None:
    store = ContextStore(BrainMemory(tmp_path / "memory.sqlite3"))
    summary = store.summary()

    assert summary.person_profile[0].name == "Julian Shelton"
    assert any(project.name == "GrimBot Butler OS" for project in summary.projects)
    roles = {entry.name: entry.content for entry in summary.relationships}
    assert "second-mind" in roles["Julian Prime"]
    assert "Chief of Staff" in roles["Maya"]
    assert "robot shell" in roles["GrimBot"]
    assert all(entry.verified for entry in summary.person_profile)


def test_project_and_priority_retrieval(tmp_path) -> None:
    store = ContextStore(BrainMemory(tmp_path / "memory.sqlite3"))

    projects = store.projects()
    priorities = store.priorities()

    assert projects[0].name == "Real Estate Acquisitions"
    assert projects[0].priority == 100
    assert priorities[0].name == "Revenue stabilization"


def test_context_search_finds_grimbot_project(tmp_path) -> None:
    store = ContextStore(BrainMemory(tmp_path / "memory.sqlite3"))

    result = store.search(ContextSearchRequest(query="what do you remember about GrimBot?"))

    assert result.projects
    assert result.projects[0].name == "GrimBot Butler OS"
    assert "operator testing" in result.projects[0].current_bottleneck.lower()
    assert result.needs_clarification is False


def test_context_remember_and_priority_update(tmp_path) -> None:
    store = ContextStore(BrainMemory(tmp_path / "memory.sqlite3"))

    remembered = store.remember(
        ContextRememberRequest(
            context_type="decision",
            name="Hardware sequencing",
            content="Validate Maya Console before adding hardware.",
            priority=88,
            verified=True,
        )
    )
    updated = store.update_priority(
        PriorityUpdateRequest(
            name="GrimBot Butler OS",
            priority=90,
            current_bottleneck="Daily operator testing",
            next_action="Use Maya Console for one real briefing.",
            verified=True,
        )
    )

    assert remembered.source == "julian_prime"
    assert remembered.verified is True
    assert updated.priority == 90
    assert updated.current_bottleneck == "Daily operator testing"
    assert updated.source == "julian_prime"


def test_verified_context_requires_julian_prime_source() -> None:
    with pytest.raises(ValidationError):
        ContextRememberRequest(
            context_type="decision",
            name="Untrusted verification",
            content="Maya inferred this.",
            source="maya",
            verified=True,
        )

    with pytest.raises(ValidationError):
        ContextRememberRequest(
            context_type="decision",
            name="Spoofed seed",
            content="Caller cannot write internal seed data.",
            source="portfolio_seed",
        )


def test_seed_refreshes_stale_rows_but_preserves_operator_updates(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    memory = BrainMemory(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE identity_projects
            SET current_bottleneck = 'stale seed text'
            WHERE normalized_name = 'maya console'
            """
        )
        connection.execute(
            """
            UPDATE identity_projects
            SET current_bottleneck = 'operator-owned context',
                source = 'julian_prime',
                verified = 1
            WHERE normalized_name = 'grimbot butler os'
            """
        )
        connection.commit()

    refreshed = ContextStore(BrainMemory(db_path)).projects()
    projects = {project.name: project for project in refreshed}

    assert projects["Maya Console"].current_bottleneck != "stale seed text"
    assert projects["GrimBot Butler OS"].current_bottleneck == "operator-owned context"
    assert projects["GrimBot Butler OS"].source == "julian_prime"


def test_birddash_seed_matches_local_lead_dashboard(tmp_path) -> None:
    projects = {
        project.name: project
        for project in ContextStore(BrainMemory(tmp_path / "memory.sqlite3")).projects()
    }

    bird_dash = projects["BirdDash"]
    assert bird_dash.status == "building"
    assert "lead dashboard" in bird_dash.current_bottleneck.lower()
    assert "csv import" in bird_dash.next_action.lower()


def test_maya_briefing_prioritizes_projects_over_room_scan(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    briefing = build_maya_briefing(
        MayaBriefingRequest(),
        RobotMemory(memory),
        ContextStore(memory),
    )

    assert "Stabilize cash flow" in briefing.priority_items[0]
    assert briefing.active_projects[0].startswith("Real Estate Acquisitions")
    assert briefing.next_best_action != "scan room for current conditions"


def test_day_question_returns_chief_of_staff_briefing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="How's my day looking?",
        ),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.machine_output["active_projects"]
    assert "cash flow" in result.speech_output.text.lower()
    assert "scan room" not in result.speech_output.text.lower()


def test_empty_voice_input_clarifies_instead_of_defaulting_to_cleanup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    result = run_voice_conversation(
        VoiceConversationRequest(push_to_talk=True, mock_transcript=" "),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.transcript == "input unavailable"
    assert result.machine_output["needs_clarification"] is True
    assert "scan room" not in result.speech_output.text.lower()
    assert "clean" not in result.speech_output.text.lower()


def test_grimbot_question_returns_project_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="What do you remember about GrimBot?",
        ),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.machine_output["projects"][0]["name"] == "GrimBot Butler OS"
    assert "GrimBot Butler OS" in result.speech_output.text
    assert "scan room" not in result.speech_output.text.lower()


def test_unverified_project_context_cannot_be_presented_as_verified(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    ContextStore(memory).update_priority(
        PriorityUpdateRequest(
            name="GrimBot Butler OS",
            priority=90,
            current_bottleneck="Unconfirmed field report",
            verified=False,
        )
    )

    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="What do you remember about GrimBot?",
            verified=True,
        ),
        memory,
    )

    assert result.machine_output["projects"][0]["verified"] is False
    assert result.maya_response.verified is False
    assert result.speech_output.text.startswith("Not verified yet.")


def test_unverified_priority_makes_briefing_unverified(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    ContextStore(memory).remember(
        ContextRememberRequest(
            context_type="priority",
            name="Revenue stabilization",
            content="Unconfirmed replacement priority.",
            priority=100,
            source="maya",
            verified=False,
        )
    )

    briefing = build_maya_briefing(
        MayaBriefingRequest(verified=True),
        RobotMemory(memory),
        ContextStore(memory),
    )

    assert briefing.verified is False


def test_unknown_question_asks_one_clarifying_question(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Tell me about the zephyr initiative.",
        ),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.machine_output["needs_clarification"] is True
    assert result.machine_output["clarification_question"]
    assert "which project" in result.speech_output.text.lower()
    assert "scan room" not in result.speech_output.text.lower()


def test_context_search_does_not_overmatch_or_return_room_memory(tmp_path) -> None:
    store = ContextStore(BrainMemory(tmp_path / "memory.sqlite3"))

    result = store.search(
        ContextSearchRequest(query="boardroom carpet cleanup protocol")
    )

    assert result.entries == []
    assert result.projects == []
    assert result.needs_clarification is True
    assert not hasattr(result, "hazards")
    assert not hasattr(result, "mess_zones")
    assert not hasattr(result, "cleanup_tasks")


def test_physical_intent_uses_exact_tokens_not_substrings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="Review the boardroom strategy.",
        ),
        BrainMemory(tmp_path / "memory.sqlite3"),
    )

    assert result.machine_output["needs_clarification"] is True
    assert result.machine_output.get("next_best_action") != "scan room for current conditions"


def test_room_memory_remains_available_for_physical_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_VOICE_MOCK", "true")
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(
            room_name="Office",
            zone_name="Desk",
            mock_camera_frame="notebooks and a loose cable",
        ),
        memory,
    )

    result = run_voice_conversation(
        VoiceConversationRequest(
            push_to_talk=True,
            mock_transcript="What should I clean first?",
            room_name="Office",
            zone_name="Desk",
            response_mode="cleanup_coaching",
        ),
        memory,
    )

    assert result.machine_output["next_best_action"] == "clear hazard: loose cord on floor"
    assert "loose cord on floor" in result.speech_output.text


def test_context_routes_exist() -> None:
    from grimbot_brain.main import app, context_update_priority

    paths = {route.path for route in app.routes}
    assert {
        "/context",
        "/context/projects",
        "/context/priorities",
        "/context/relationships",
        "/context/search",
        "/context/remember",
        "/context/update-priority",
    } <= paths
    context_routes = [
        route
        for route in app.routes
        if route.path.startswith("/context")
    ]
    assert all(route.response_model is not None for route in context_routes)

    with pytest.raises(HTTPException) as exc_info:
        context_update_priority(
            PriorityUpdateRequest(name="Unknown Project", priority=50)
        )
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Unknown project: Unknown Project"
