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


def test_existing_v02_database_initializes_v03_tables(tmp_path) -> None:
    import sqlite3

    db_path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                cycle_input TEXT NOT NULL,
                perception TEXT NOT NULL,
                intent TEXT NOT NULL,
                command TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE room_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                scan_result TEXT NOT NULL
            )
            """
        )
        connection.commit()

    upgraded = BrainMemory(db_path)
    with upgraded.db_path.open("rb"):
        pass

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {
        "rooms",
        "room_zones",
        "known_objects",
        "hazards",
        "mess_observations",
        "cleanup_tasks",
        "episodic_memories",
        "semantic_facts",
    }.issubset(tables)
