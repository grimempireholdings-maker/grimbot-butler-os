from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .schemas import BrainCycleInput, PerceptionResult, RobotCommand, RobotIntent, RoomScanResult


class BrainMemory:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.getenv("GRIMBOT_DB_PATH", "memory/grimbot_brain.sqlite3"))
        self._initialize()

    def log_cycle(
        self,
        cycle_input: BrainCycleInput,
        perception: PerceptionResult,
        intent: RobotIntent,
        command: RobotCommand,
    ) -> int:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO cycles (cycle_input, perception, intent, command)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cycle_input.model_dump_json(),
                    perception.model_dump_json(),
                    intent.model_dump_json(),
                    command.model_dump_json(),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_cycles(self, limit: int = 10) -> list[dict]:
        safe_limit = max(1, min(limit, 100))
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, cycle_input, perception, intent, command
                FROM cycles
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "cycle_input": json.loads(row["cycle_input"]),
                "perception": json.loads(row["perception"]),
                "intent": json.loads(row["intent"]),
                "command": json.loads(row["command"]),
            }
            for row in rows
        ]

    def log_room_scan(self, scan: RoomScanResult) -> int:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO room_scans (scan_result)
                VALUES (?)
                """,
                (scan.model_dump_json(),),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_room_scans(self, limit: int = 10) -> list[dict]:
        safe_limit = max(1, min(limit, 100))
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, scan_result
                FROM room_scans
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "scan_result": json.loads(row["scan_result"]),
            }
            for row in rows
        ]

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycles (
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
                CREATE TABLE IF NOT EXISTS room_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    scan_result TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rooms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    importance REAL NOT NULL DEFAULT 0.5,
                    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS room_zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    importance REAL NOT NULL DEFAULT 0.5,
                    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(room_id, name),
                    FOREIGN KEY(room_id) REFERENCES rooms(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS known_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER,
                    zone_id INTEGER,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    importance REAL NOT NULL DEFAULT 0.5,
                    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(room_id, zone_id, name),
                    FOREIGN KEY(room_id) REFERENCES rooms(id),
                    FOREIGN KEY(zone_id) REFERENCES room_zones(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hazards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER,
                    zone_id INTEGER,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.75,
                    importance REAL NOT NULL DEFAULT 0.85,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(room_id, zone_id, name),
                    FOREIGN KEY(room_id) REFERENCES rooms(id),
                    FOREIGN KEY(zone_id) REFERENCES room_zones(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mess_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER,
                    zone_id INTEGER,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    importance REAL NOT NULL DEFAULT 0.6,
                    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(room_id, zone_id, name),
                    FOREIGN KEY(room_id) REFERENCES rooms(id),
                    FOREIGN KEY(zone_id) REFERENCES room_zones(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cleanup_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER,
                    zone_id INTEGER,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    count INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    importance REAL NOT NULL DEFAULT 0.65,
                    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(room_id, zone_id, name, status),
                    FOREIGN KEY(room_id) REFERENCES rooms(id),
                    FOREIGN KEY(zone_id) REFERENCES room_zones(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER,
                    zone_id INTEGER,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(room_id) REFERENCES rooms(id),
                    FOREIGN KEY(zone_id) REFERENCES room_zones(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_facts (
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
                    UNIQUE(room_id, zone_id, fact_key),
                    FOREIGN KEY(room_id) REFERENCES rooms(id),
                    FOREIGN KEY(zone_id) REFERENCES room_zones(id)
                )
                """
            )
            connection.commit()
