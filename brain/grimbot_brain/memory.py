from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .schemas import BrainCycleInput, PerceptionResult, RobotCommand, RobotIntent


class BrainMemory:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.getenv("GRIMBOT_DB_PATH", "grimbot_brain.sqlite3"))
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
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, cycle_input, perception, intent, command
                FROM cycles
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
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
            connection.commit()
