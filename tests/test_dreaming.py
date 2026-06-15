import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

import grimbot_brain.main as main_module
from grimbot_brain.dreaming.consolidator import Consolidator
from grimbot_brain.dreaming.dream_schemas import DreamRunRequest, PromotionReviewRequest
from grimbot_brain.dreaming.dreaming_engine import DreamingEngine
from grimbot_brain.dreaming.forgetter import Forgetter
from grimbot_brain.dreaming.providers.rule_based import Episode, RuleBasedProvider
from grimbot_brain.main import app
from grimbot_brain.maya_core import build_maya_briefing
from grimbot_brain.memory import BrainMemory
from grimbot_brain.robot_memory import RobotMemory
from grimbot_brain.safety import validate_action
from grimbot_brain.schemas import (
    BrainCycleInput,
    MayaBriefingRequest,
    RelevantMemoryRequest,
    RobotIntent,
    SkillRunRequest,
)
from grimbot_brain.skills import create_default_registry


def test_v07_database_migrates_dreaming_tables_and_columns(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE episodic_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id INTEGER,
                zone_id INTEGER,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE semantic_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id INTEGER,
                zone_id INTEGER,
                fact_key TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.7,
                importance REAL NOT NULL DEFAULT 0.5,
                count INTEGER NOT NULL DEFAULT 1,
                first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(room_id, zone_id, fact_key)
            )
            """
        )
        connection.commit()

    BrainMemory(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        episode_columns = {row[1] for row in connection.execute("PRAGMA table_info(episodic_memories)")}
        fact_columns = {row[1] for row in connection.execute("PRAGMA table_info(semantic_facts)")}

    assert {"promotion_queue", "dream_cycles"}.issubset(tables)
    assert {"consolidated", "anchor"}.issubset(episode_columns)
    assert {"created_at", "last_reinforced", "tags", "tier"}.issubset(fact_columns)


def test_rule_based_consolidation_clusters_repeated_episodes() -> None:
    episodes = [
        _episode(1, "Loose cord near the desk", room_name="Office"),
        _episode(2, "Loose cord near the desk", room_name="Office"),
        _episode(3, "Dishes on the counter", room_name="Kitchen"),
    ]

    candidates = Consolidator(RuleBasedProvider()).consolidate(episodes)

    assert len(candidates) == 1
    assert candidates[0].frequency == 2
    assert candidates[0].content == "Repeated observation in Office: Loose cord near the desk"
    assert {"hazard", "safety", "room:office", "object:cord"}.issubset(candidates[0].tags)


def test_manual_dream_creates_fact_queue_and_log(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Loose cord near the desk")

    result = DreamingEngine(memory).run(DreamRunRequest())
    promotions = DreamingEngine(memory).promotions()
    status = DreamingEngine(memory).status()

    assert result.cycle.status == "completed"
    assert result.cycle.episodes_processed == 2
    assert result.cycle.facts_created == 1
    assert result.promotions_created == 1
    assert promotions[0].status == "pending"
    assert status.active is False
    assert status.latest_cycle == result.cycle


def test_repeated_dream_run_reinforces_without_duplicate_queue_entries(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Loose cord near the desk")
    engine = DreamingEngine(memory)

    first = engine.run(DreamRunRequest())
    second = engine.run(DreamRunRequest())

    assert first.cycle.facts_created == 1
    assert second.cycle.facts_created == 0
    assert second.promotions_created == 0
    assert len(engine.facts()) == 1
    assert len(engine.promotions()) == 1
    with sqlite3.connect(memory.db_path) as connection:
        count = connection.execute("SELECT count FROM semantic_facts").fetchone()[0]
    assert count == 2


def test_semantic_fact_deduplication_ignores_case_and_punctuation(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Loose cord near the desk")
    engine = DreamingEngine(memory)
    engine.run(DreamRunRequest())
    with sqlite3.connect(memory.db_path) as connection:
        room_id = connection.execute("SELECT id FROM rooms WHERE name = 'office'").fetchone()[0]
        connection.execute("DELETE FROM episodic_memories")
        for _ in range(2):
            connection.execute(
                """
                INSERT INTO episodic_memories (room_id, kind, content, importance)
                VALUES (?, 'observation', 'LOOSE CORD near the desk!!!', 0.6)
                """,
                (room_id,),
            )
        connection.commit()

    engine.run(DreamRunRequest())

    assert len(engine.facts()) == 1
    assert len(engine.promotions()) == 1


def test_pending_fact_is_not_active_until_manual_approval(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Loose cord near the desk")
    engine = DreamingEngine(memory)

    engine.run(DreamRunRequest())
    before = RobotMemory(memory).relevant(RelevantMemoryRequest(query="desk"))
    room_before = RobotMemory(memory).room_summary("Office")
    briefing_before = build_maya_briefing(
        MayaBriefingRequest(room_name="Office"),
        RobotMemory(memory),
    )
    skill_before = create_default_registry(memory).run(
        "memory_review",
        SkillRunRequest(inputs={"room_name": "Office"}, permission="observe"),
    )
    promotion = engine.promotions()[0]
    approved = engine.approve(promotion.id, PromotionReviewRequest(note="Reviewed by Chief"))
    after = RobotMemory(memory).relevant(RelevantMemoryRequest(query="desk"))

    assert before.semantic_facts == []
    assert room_before.semantic_facts == []
    assert briefing_before.fyi
    assert all("Loose cord near the desk" not in item for item in briefing_before.fyi)
    assert skill_before.machine_output["data"]["semantic_facts"] == []
    assert approved.status == "approved"
    assert after.semantic_facts[0]["content"] == "Repeated observation in Office: Loose cord near the desk"


def test_manual_rejection_keeps_fact_out_of_active_memory(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Notebooks collect on the desk")
    engine = DreamingEngine(memory)
    engine.run(DreamRunRequest())

    rejected = engine.reject(engine.promotions()[0].id, PromotionReviewRequest(note="Not reliable"))
    relevant = RobotMemory(memory).relevant(RelevantMemoryRequest(query="desk"))

    assert rejected.status == "rejected"
    assert relevant.semantic_facts == []


def test_promotion_cannot_be_reviewed_twice(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Notebooks collect on the desk")
    engine = DreamingEngine(memory)
    engine.run(DreamRunRequest())
    promotion_id = engine.promotions()[0].id
    engine.approve(promotion_id, PromotionReviewRequest())

    with pytest.raises(ValueError, match="already been reviewed"):
        engine.reject(promotion_id, PromotionReviewRequest())


def test_orphaned_promotion_cannot_be_approved(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO promotion_queue (fact_id, status)
            VALUES (999, 'pending')
            """
        )
        promotion_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
        connection.commit()

    with pytest.raises(KeyError, match="Unknown promotion"):
        DreamingEngine(memory).approve(promotion_id, PromotionReviewRequest())


def test_anchor_approval_promotes_fact_to_core(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Emergency stop must remain available")
    engine = DreamingEngine(memory)
    engine.run(DreamRunRequest())

    promoted = engine.approve(
        engine.promotions()[0].id,
        PromotionReviewRequest(note="Core safety memory", anchor=True),
    )

    assert promoted.status == "anchor"
    assert promoted.fact.tier == "core"


def test_forgetting_removes_only_low_value_unprotected_facts(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    stale = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
    with sqlite3.connect(memory.db_path) as connection:
        connection.row_factory = sqlite3.Row
        removable_id = _insert_fact(connection, "old clutter note", stale, tags=[], tier="semantic")
        _insert_fact(connection, "Emergency stop protocol", stale, tags=["safety"], tier="semantic")
        _insert_fact(connection, "Permanent preference", stale, tags=[], tier="core")
        connection.commit()

        forgotten = Forgetter().forget_stale_facts(connection)
        connection.commit()
        remaining = {row[0] for row in connection.execute("SELECT id FROM semantic_facts")}

    assert forgotten == 1
    assert removable_id not in remaining
    assert len(remaining) == 2


def test_forgetting_recognizes_safety_words_with_punctuation(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    stale = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
    with sqlite3.connect(memory.db_path) as connection:
        connection.row_factory = sqlite3.Row
        content_id = _insert_fact(connection, "hazard-related note", stale, tags=[], tier="semantic")
        tag_id = _insert_fact(connection, "critical note", stale, tags=["safety:critical"], tier="semantic")
        fire_id = _insert_fact(connection, "fire-risk near battery", stale, tags=[], tier="semantic")
        connection.commit()

        forgotten = Forgetter().forget_stale_facts(connection)
        remaining = {row[0] for row in connection.execute("SELECT id FROM semantic_facts")}

    assert forgotten == 0
    assert {content_id, tag_id, fire_id}.issubset(remaining)


def test_fact_response_shape_sanitizes_corrupt_optional_fields(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute(
            """
            INSERT INTO semantic_facts (
                fact_key, content, confidence, importance, count,
                first_seen, last_seen, created_at, last_reinforced, tags, tier
            )
            VALUES ('corrupt-shape', 'Fact', 5, 0.5, 1, ?, ?, ?, ?, ?, 'invalid')
            """,
            (now, now, now, now, json.dumps([1, "tag"])),
        )
        connection.commit()

    fact = DreamingEngine(memory).facts()[0]

    assert fact.confidence == 1.0
    assert fact.tags == ["1", "tag"]
    assert fact.tier == "semantic"


def test_forgetting_deletes_rejected_queue_with_rejected_fact(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    stale = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
    with sqlite3.connect(memory.db_path) as connection:
        connection.row_factory = sqlite3.Row
        fact_id = _insert_fact(connection, "discarded clutter note", stale, tags=[], tier="semantic")
        connection.execute(
            "INSERT INTO promotion_queue (fact_id, status) VALUES (?, 'rejected')",
            (fact_id,),
        )
        connection.commit()

        forgotten = Forgetter().forget_stale_facts(connection)
        connection.commit()
        fact = connection.execute("SELECT 1 FROM semantic_facts WHERE id = ?", (fact_id,)).fetchone()
        queue = connection.execute("SELECT 1 FROM promotion_queue WHERE fact_id = ?", (fact_id,)).fetchone()

    assert forgotten == 1
    assert fact is None
    assert queue is None


def test_approved_fact_is_never_forgotten(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    stale = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
    with sqlite3.connect(memory.db_path) as connection:
        connection.row_factory = sqlite3.Row
        fact_id = _insert_fact(connection, "Approved old fact", stale, tags=[], tier="semantic")
        connection.execute(
            "INSERT INTO promotion_queue (fact_id, status) VALUES (?, 'approved')",
            (fact_id,),
        )
        connection.commit()

        forgotten = Forgetter().forget_stale_facts(connection)
        exists = connection.execute("SELECT 1 FROM semantic_facts WHERE id = ?", (fact_id,)).fetchone()

    assert forgotten == 0
    assert exists is not None


def test_dream_cycle_does_not_modify_episodes_or_adaptive_state(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Notebooks collect on the desk", anchor=True)
    with sqlite3.connect(memory.db_path) as connection:
        before_episodes = connection.execute(
            "SELECT id, consolidated, anchor FROM episodic_memories ORDER BY id"
        ).fetchall()
        before_state = connection.execute(
            "SELECT name, current_value FROM adaptive_state_signals ORDER BY name"
        ).fetchall()

    DreamingEngine(memory).run(DreamRunRequest())

    with sqlite3.connect(memory.db_path) as connection:
        after_episodes = connection.execute(
            "SELECT id, consolidated, anchor FROM episodic_memories ORDER BY id"
        ).fetchall()
        after_state = connection.execute(
            "SELECT name, current_value FROM adaptive_state_signals ORDER BY name"
        ).fetchall()

    assert after_episodes == before_episodes
    assert after_state == before_state


def test_failed_dream_cycle_rolls_back_partial_fact_writes(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Notebooks collect on the desk")
    engine = DreamingEngine(memory)

    def fail_after_write(connection, candidates):
        connection.execute(
            """
            INSERT INTO semantic_facts (
                fact_key, content, confidence, importance, count,
                first_seen, last_seen, created_at, last_reinforced, tags, tier
            )
            VALUES ('partial', 'partial fact', 0.5, 0.5, 1,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP, '[]', 'semantic')
            """
        )
        raise RuntimeError("injected dream failure")

    monkeypatch.setattr(engine, "_store_candidates", fail_after_write)

    with pytest.raises(RuntimeError, match="injected dream failure"):
        engine.run(DreamRunRequest())

    status = engine.status()
    with sqlite3.connect(memory.db_path) as connection:
        facts = connection.execute("SELECT content FROM semantic_facts").fetchall()
        promotions = connection.execute("SELECT id FROM promotion_queue").fetchall()

    assert status.latest_cycle is not None
    assert status.latest_cycle.status == "failed"
    assert status.latest_cycle.completed_at is not None
    assert status.latest_cycle.error_message == "injected dream failure"
    assert facts == []
    assert promotions == []


def test_stale_running_cycle_is_marked_failed(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute(
            """
            INSERT INTO dream_cycles (started_at, status)
            VALUES (datetime('now', '-2 hours'), 'running')
            """
        )
        connection.commit()

    status = DreamingEngine(memory).status()

    assert status.active is False
    assert status.latest_cycle is not None
    assert status.latest_cycle.status == "failed"
    assert status.latest_cycle.error_message == "Dream cycle interrupted before completion"


def test_concurrent_manual_dream_cycle_is_rejected(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute("INSERT INTO dream_cycles (status) VALUES ('running')")
        connection.commit()

    with pytest.raises(ValueError, match="already running"):
        DreamingEngine(memory).run(DreamRunRequest())
    monkeypatch.setattr(main_module, "memory", memory)
    with pytest.raises(HTTPException) as exc_info:
        main_module.dream_run(DreamRunRequest())

    with sqlite3.connect(memory.db_path) as connection:
        cycles = connection.execute("SELECT status FROM dream_cycles").fetchall()
    assert exc_info.value.status_code == 409
    assert cycles == [("running",)]


def test_dream_endpoint_response_shapes_and_review_conflicts(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Loose cord near the desk")
    monkeypatch.setattr(main_module, "memory", memory)

    run_response = main_module.dream_run(DreamRunRequest())
    status_response = main_module.dream_status()
    facts_response = main_module.dream_facts()
    promotions_response = main_module.dream_promotions()
    promotion_id = promotions_response[0].id
    approval_response = main_module.dream_approve(
        promotion_id,
        PromotionReviewRequest(note="Reviewed"),
    )

    assert set(run_response.model_dump()) == {"cycle", "candidate_facts", "promotions_created"}
    assert set(status_response.model_dump()) == {"trigger_policy", "active", "latest_cycle"}
    assert set(facts_response[0].model_dump()) == {
        "fact_id",
        "content",
        "confidence",
        "created_at",
        "last_reinforced",
        "tags",
        "tier",
    }
    assert set(promotions_response[0].model_dump()) == {
        "id",
        "fact_id",
        "status",
        "created_at",
        "reviewed_at",
        "review_note",
        "fact",
    }
    assert approval_response.status == "approved"
    with pytest.raises(HTTPException) as exc_info:
        main_module.dream_reject(
            promotion_id,
            PromotionReviewRequest(note="Too late"),
        )
    assert exc_info.value.status_code == 409


def test_dream_routes_are_manual_only_and_safety_still_wins(tmp_path) -> None:
    paths = {route.path for route in app.routes}
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    _insert_repeated_episodes(memory, "Move forward through the hallway")
    DreamingEngine(memory).run(DreamRunRequest())

    command = validate_action(
        BrainCycleInput(battery_percentage=80, distance_cm=5, user_command="move forward"),
        RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="dream candidate"),
    )

    assert {
        "/dream/run",
        "/dream/status",
        "/dream/facts",
        "/dream/promotions",
        "/dream/promotions/{promotion_id}/approve",
        "/dream/promotions/{promotion_id}/reject",
    }.issubset(paths)
    assert not any("idle" in path or "automatic" in path for path in paths)
    assert command.action == "stop"
    assert command.reason == "Obstacle too close"


def _episode(
    episode_id: int,
    content: str,
    room_name: str | None = None,
) -> Episode:
    return Episode(
        episode_id=episode_id,
        content=content,
        kind="observation",
        importance=0.5,
        created_at="2026-06-14T00:00:00+00:00",
        anchor=False,
        room_id=1 if room_name else None,
        zone_id=None,
        room_name=room_name,
        zone_name=None,
    )


def _insert_repeated_episodes(memory: BrainMemory, content: str, anchor: bool = False) -> None:
    with sqlite3.connect(memory.db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO rooms (name, display_name) VALUES ('office', 'Office')"
        )
        room_id = int(cursor.lastrowid)
        for _ in range(2):
            connection.execute(
                """
                INSERT INTO episodic_memories (room_id, kind, content, importance, anchor)
                VALUES (?, 'observation', ?, 0.5, ?)
                """,
                (room_id, content, int(anchor)),
            )
        connection.commit()


def _insert_fact(
    connection: sqlite3.Connection,
    content: str,
    timestamp: str,
    tags: list[str],
    tier: str,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO semantic_facts (
            fact_key, content, confidence, importance, count,
            first_seen, last_seen, created_at, last_reinforced, tags, tier
        )
        VALUES (?, ?, 0.1, 0.0, 1, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"test:{content}",
            content,
            timestamp,
            timestamp,
            timestamp,
            timestamp,
            json.dumps(tags),
            tier,
        ),
    )
    return int(cursor.lastrowid)
