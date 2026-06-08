from grimbot_brain.memory import BrainMemory
from grimbot_brain.schemas import BrainCycleInput, PerceptionResult, RobotCommand, RobotIntent, RoomScanResult


def test_memory_logs_and_reads_recent_cycles(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "cycles.sqlite3")
    cycle_input = BrainCycleInput(battery_percentage=80, distance_cm=100, user_command="stop")
    perception = PerceptionResult(mode="mock", scene_summary="Clear", confidence=0.75)
    intent = RobotIntent(requested_action="stop", requested_speed=0, reason="User requested stop")
    command = RobotCommand(action="stop", speed=0, reason="User requested stop")

    cycle_id = memory.log_cycle(cycle_input, perception, intent, command)
    rows = memory.recent_cycles(limit=1)

    assert cycle_id == 1
    assert len(rows) == 1
    assert rows[0]["command"] == {"action": "stop", "speed": 0.0, "reason": "User requested stop"}


def test_recent_cycles_limit_is_clamped(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "cycles.sqlite3")

    rows = memory.recent_cycles(limit=1000)

    assert rows == []


def test_memory_logs_room_scan_results(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "cycles.sqlite3")
    scan = RoomScanResult(
        room_summary="Clear room",
        visible_objects=["floor"],
        mess_zones=["general surfaces"],
        hazards=[],
        suggested_cleanup_order=["general surfaces"],
        next_best_action="general surfaces",
        mode="mock",
    )

    scan_id = memory.log_room_scan(scan)
    rows = memory.recent_room_scans(limit=1)

    assert scan_id == 1
    assert rows[0]["scan_result"]["room_summary"] == "Clear room"
