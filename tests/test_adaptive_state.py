import math
import sqlite3

from grimbot_brain.adaptive_state import AdaptiveState
from grimbot_brain.memory import BrainMemory
from grimbot_brain.response_composer import compose_maya_response
from grimbot_brain.robot_memory import RobotMemory
from grimbot_brain.room_scan import run_room_scan
from grimbot_brain.safety import validate_action
from grimbot_brain.schemas import (
    BrainCycleInput,
    MayaComposeRequest,
    RobotIntent,
    RoomScanRequest,
    RelevantMemoryRequest,
    SkillRunRequest,
    StateDecayRequest,
    StateEventRequest,
)
from grimbot_brain.skills import create_default_registry


def test_state_initialization_creates_all_signals(tmp_path) -> None:
    state = AdaptiveState(BrainMemory(tmp_path / "memory.sqlite3")).snapshot()

    assert set(state.values) == {
        "attention",
        "urgency",
        "novelty",
        "confidence",
        "reward",
        "friction",
        "fatigue",
        "curiosity",
    }
    assert state.values["urgency"] == 0.2
    assert state.next_best_action == "continue normal observation and suggestion"


def test_state_persists_across_instances(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    AdaptiveState(BrainMemory(db_path)).apply_event(
        StateEventRequest(event_type="cleanup_succeeded", intensity=1, reason="cleanup complete")
    )

    values = AdaptiveState(BrainMemory(db_path)).values()

    assert values["reward"] > 0.25
    assert values["confidence"] > 0.55


def test_missing_signal_initialization_preserves_existing_values(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    state = AdaptiveState(memory)
    state.apply_event(StateEventRequest(event_type="hazard_observed", intensity=1, reason="keep urgency"))
    before = state.values()["urgency"]
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute("DELETE FROM adaptive_state_signals WHERE name = ?", ("curiosity",))
        connection.commit()

    after = AdaptiveState(memory).values()

    assert after["urgency"] == before
    assert after["curiosity"] == 0.3


def test_state_updates_are_bounded(tmp_path) -> None:
    state = AdaptiveState(BrainMemory(tmp_path / "memory.sqlite3"))

    for _ in range(10):
        state.apply_event(
            StateEventRequest(
                event_type="hazard_observed",
                intensity=1,
                reason="loose cord seen again",
                metadata={"count": 10},
            )
        )

    snapshot = state.snapshot()
    assert snapshot.values["urgency"] == 1.0
    assert snapshot.values["attention"] == 1.0
    assert all(0 <= value <= 1 for value in snapshot.values.values())


def test_state_initialization_sanitizes_corrupt_sqlite_rows(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    state = AdaptiveState(memory)
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute(
            "UPDATE adaptive_state_signals SET current_value = ? WHERE name = ?",
            ("nan", "urgency"),
        )
        connection.execute(
            """
            INSERT INTO adaptive_state_signals
                (name, current_value, min_value, max_value, baseline, decay_rate, source, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("unknown_signal", 999, 0, 1, 0.5, 0.1, "test", "corrupt row"),
        )
        connection.commit()

    snapshot = AdaptiveState(memory).snapshot()

    assert "unknown_signal" not in snapshot.values
    assert snapshot.values["urgency"] == 0.0


def test_decay_moves_values_toward_baseline(tmp_path) -> None:
    state = AdaptiveState(BrainMemory(tmp_path / "memory.sqlite3"))
    state.apply_event(StateEventRequest(event_type="hazard_observed", intensity=1, reason="test hazard"))
    before = state.snapshot().values["urgency"]

    after = state.decay(StateDecayRequest(elapsed_seconds=3600, reason="test decay")).values["urgency"]

    assert after < before
    assert after > 0.2


def test_decay_handles_negative_and_nonfinite_values(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    state = AdaptiveState(memory)
    with sqlite3.connect(memory.db_path) as connection:
        connection.execute("UPDATE adaptive_state_signals SET current_value = ? WHERE name = ?", (-5, "reward"))
        connection.execute("UPDATE adaptive_state_signals SET current_value = ? WHERE name = ?", (math.inf, "fatigue"))
        connection.commit()

    snapshot = state.decay(StateDecayRequest(elapsed_seconds=3600, reason="sanitize bad values"))

    assert 0 <= snapshot.values["reward"] <= 1
    assert 0 <= snapshot.values["fatigue"] <= 1


def test_repeated_hazard_increases_urgency_from_room_scan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory(tmp_path / "memory.sqlite3")

    for _ in range(2):
        run_room_scan(
            RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="loose cable"),
            memory,
        )

    values = AdaptiveState(memory).values()
    assert values["urgency"] > 0.5
    assert values["attention"] > 0.5


def test_successful_cleanup_increases_reward_and_confidence(tmp_path) -> None:
    state = AdaptiveState(BrainMemory(tmp_path / "memory.sqlite3"))
    before = state.values()

    state.apply_event(StateEventRequest(event_type="cleanup_succeeded", intensity=0.8, reason="desk reset completed"))
    after = state.values()

    assert after["reward"] > before["reward"]
    assert after["confidence"] > before["confidence"]


def test_maya_response_style_adapts_to_high_urgency() -> None:
    response = compose_maya_response(
        MayaComposeRequest(
            raw_output={"next_best_action": "clear hazard: loose cord on floor"},
            response_mode="cleanup_coaching",
            adaptive_state={"urgency": 0.8, "attention": 0.8},
            verified=False,
        )
    )

    assert "Urgency is elevated, so I will keep this concise." in response.user_response
    assert "clear hazard: loose cord on floor" in response.user_response


def test_maya_response_ignores_nonfinite_state_values() -> None:
    response = compose_maya_response(
        MayaComposeRequest(
            raw_output={"next_best_action": "scan room"},
            response_mode="cleanup_coaching",
            adaptive_state={"urgency": math.nan, "confidence": math.inf},
            verified=False,
        )
    )

    assert "Urgency is elevated" not in response.user_response
    assert "Confidence is high" not in response.user_response


def test_skill_ranking_changes_with_state(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    state = AdaptiveState(memory)
    baseline_first = create_default_registry(memory, state).list_skills()[0].name

    state.apply_event(
        StateEventRequest(
            event_type="discovery",
            intensity=1,
            reason="new room objects discovered",
            metadata={"object_count": 8},
        )
    )
    elevated_first = create_default_registry(memory, state).list_skills()[0].name

    assert baseline_first != elevated_first
    assert elevated_first == "memory_review"


def test_skill_ranking_is_deterministic_for_ties(tmp_path) -> None:
    state = AdaptiveState(BrainMemory(tmp_path / "memory.sqlite3"))

    ranked = state.rank_skill_names(["zzz_unknown", "aaa_unknown"])

    assert ranked == ["aaa_unknown", "zzz_unknown"]


def test_state_influences_relevant_memory_next_action(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="loose cable"),
        memory,
    )

    result = RobotMemory(memory).relevant(
        RelevantMemoryRequest(
            query="what should I clean first?",
            room_name="Office",
            adaptive_state={"urgency": 0.8},
        )
    )

    assert result.next_best_action == "clear hazard: loose cord on floor before lower-priority organization"


def test_extreme_external_state_is_clamped_for_memory_relevance(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="loose cable"),
        memory,
    )

    result = RobotMemory(memory).relevant(
        RelevantMemoryRequest(
            query="what should I clean first?",
            room_name="Office",
            adaptive_state={"urgency": 999, "friction": -999, "curiosity": math.nan},
        )
    )

    assert result.next_best_action == "clear hazard: loose cord on floor before lower-priority organization"


def test_safety_still_overrides_state_influenced_actions(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    state = AdaptiveState(memory)
    state.apply_event(
        StateEventRequest(event_type="hazard_observed", intensity=1, reason="hazard pressure", metadata={"count": 4})
    )
    skill_result = create_default_registry(memory, state).run(
        "task_breakdown",
        SkillRunRequest(inputs={"task": "move forward"}, permission="ask_approval"),
    )
    cycle_input = BrainCycleInput(battery_percentage=80, distance_cm=5, user_command="move forward")
    intent = RobotIntent(
        requested_action="move_forward",
        requested_speed=0.25,
        reason=skill_result.machine_output["next_best_action"],
    )

    command = validate_action(cycle_input, intent)

    assert skill_result.allowed is True
    assert command.action == "stop"
    assert command.reason == "Obstacle too close"
