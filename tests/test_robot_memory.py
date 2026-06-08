from grimbot_brain.memory import BrainMemory
from grimbot_brain.robot_memory import RobotMemory
from grimbot_brain.room_scan import run_room_scan
from grimbot_brain.safety import validate_action
from grimbot_brain.schemas import (
    BrainCycleInput,
    RelevantMemoryRequest,
    RememberRequest,
    RobotIntent,
    RoomScanRequest,
)


def test_remember_creates_room_and_fact(tmp_path) -> None:
    memory = RobotMemory(BrainMemory(tmp_path / "memory.sqlite3"))

    response = memory.remember(
        RememberRequest(
            text="The desk zone often has notebooks.",
            room_name="Office",
            zone_name="Desk",
            importance=0.8,
        )
    )
    rooms = memory.list_rooms()
    summary = memory.room_summary("office")

    assert response["episodic_memory_id"] == 1
    assert rooms[0].name == "Office"
    assert summary.room_name == "Office"
    assert summary.semantic_facts[0]["content"] == "The desk zone often has notebooks."


def test_room_scan_saves_objects_hazards_mess_and_cleanup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")

    run_room_scan(
        RoomScanRequest(
            room_name="Office",
            zone_name="Desk",
            mock_camera_frame="laundry, dishes, cable, and spill",
        ),
        brain_memory,
    )
    memory = RobotMemory(brain_memory)
    summary = memory.room_summary("Office")

    assert "cord" in [item.name for item in summary.known_objects]
    assert "loose cord on floor" in [item.name for item in summary.hazards]
    assert "spill area" in [item.name for item in summary.mess_zones]
    assert summary.recommended_first_cleanup_action == "clear hazard: loose cord on floor"


def test_deduplication_updates_counts_instead_of_creating_duplicates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")

    for _ in range(2):
        run_room_scan(
            RoomScanRequest(
                room_name="Office",
                zone_name="Desk",
                mock_camera_frame="loose cable",
            ),
            brain_memory,
        )

    hazards = RobotMemory(brain_memory).hazards(room_name="Office")

    assert len([hazard for hazard in hazards if hazard.name == "loose cord on floor"]) == 1
    assert hazards[0].count == 2
    assert hazards[0].confidence > 0.75


def test_retrieval_by_room_hazard_and_mess_zone(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Kitchen", zone_name="Counter", mock_camera_frame="dishes and spill"),
        brain_memory,
    )
    robot_memory = RobotMemory(brain_memory)

    room = robot_memory.room_summary("Kitchen")
    hazards = robot_memory.hazards(room_name="Kitchen")
    mess_zones = robot_memory.mess_zones(room_name="Kitchen", zone_name="Counter")

    assert room.room_name == "Kitchen"
    assert hazards[0].name == "possible liquid spill"
    assert "dishes on table" in [mess.name for mess in mess_zones]


def test_relevant_memory_answers_cleanup_priority(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="notebooks, drink, and cable"),
        brain_memory,
    )

    result = RobotMemory(brain_memory).relevant(
        RelevantMemoryRequest(query="what should I clean first?", room_name="Office")
    )

    assert result.next_best_action == "clear hazard: loose cord on floor"
    assert result.hazards[0].name == "loose cord on floor"


def test_unknown_room_filter_does_not_return_all_hazards(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="loose cable"),
        brain_memory,
    )

    robot_memory = RobotMemory(brain_memory)
    hazards = robot_memory.hazards(room_name="Kitchen")
    relevant = robot_memory.relevant(
        RelevantMemoryRequest(query="what hazards have you seen?", room_name="Kitchen")
    )

    assert hazards == []
    assert relevant.hazards == []
    assert relevant.cleanup_tasks == []
    assert relevant.next_best_action == "scan room for current conditions"


def test_unknown_zone_filter_does_not_return_all_room_hazards(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="loose cable"),
        brain_memory,
    )

    hazards = RobotMemory(brain_memory).hazards(room_name="Office", zone_name="Closet")

    assert hazards == []


def test_normalization_prevents_blank_room_and_zone_names(tmp_path) -> None:
    memory = RobotMemory(BrainMemory(tmp_path / "memory.sqlite3"))

    memory.remember(
        RememberRequest(
            text="Keep this as a test memory.",
            room_name="!!!",
            zone_name="???",
        )
    )
    summary = memory.room_summary("unknown")

    assert summary.room_name == "!!!"
    assert summary.zones[0].name == "???"


def test_safety_still_overrides_memory_informed_movement(tmp_path) -> None:
    memory = RobotMemory(BrainMemory(tmp_path / "memory.sqlite3"))
    memory.remember(
        RememberRequest(
            text="The hallway is usually clear.",
            room_name="Hallway",
            importance=0.9,
        )
    )
    cycle_input = BrainCycleInput(battery_percentage=80, distance_cm=5, user_command="move forward")
    intent = RobotIntent(
        requested_action="move_forward",
        requested_speed=0.25,
        reason="Memory says hallway is usually clear",
    )

    command = validate_action(cycle_input, intent)

    assert command.action == "stop"
    assert command.reason == "Obstacle too close"
