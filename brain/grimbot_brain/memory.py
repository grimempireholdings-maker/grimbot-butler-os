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

    def log_episode(self, kind: str, content: str, importance: float = 0.5) -> int:
        """Record a non-spatial event for later review and dreaming."""
        safe_kind = str(kind).strip()[:80] or "event"
        safe_content = str(content).strip()[:8000] or "No event detail provided."
        safe_importance = max(0.0, min(float(importance), 1.0))
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO episodic_memories (room_id, zone_id, kind, content, importance)
                VALUES (NULL, NULL, ?, ?, ?)
                """,
                (safe_kind, safe_content, safe_importance),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def recent_episodes(self, limit: int = 5) -> list[dict]:
        """Return recent non-spatial memories for read-only orientation."""
        safe_limit = max(1, min(limit, 25))
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, kind, content, importance
                FROM episodic_memories
                WHERE room_id IS NULL AND zone_id IS NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_state_signals (
                    name TEXT PRIMARY KEY,
                    current_value REAL NOT NULL,
                    min_value REAL NOT NULL,
                    max_value REAL NOT NULL,
                    baseline REAL NOT NULL,
                    decay_rate REAL NOT NULL,
                    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL DEFAULT 'system',
                    reason TEXT NOT NULL DEFAULT 'initialized'
                )
                """
            )
            self._ensure_column(connection, "episodic_memories", "consolidated", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "episodic_memories", "anchor", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "semantic_facts", "created_at", "TEXT")
            self._ensure_column(connection, "semantic_facts", "last_reinforced", "TEXT")
            self._ensure_column(connection, "semantic_facts", "tags", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(connection, "semantic_facts", "tier", "TEXT NOT NULL DEFAULT 'semantic'")
            connection.execute(
                """
                UPDATE semantic_facts
                SET created_at = COALESCE(created_at, first_seen),
                    last_reinforced = COALESCE(last_reinforced, last_seen),
                    tags = COALESCE(tags, '[]'),
                    tier = CASE WHEN tier = 'core' THEN 'core' ELSE 'semantic' END
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id INTEGER NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'approved', 'rejected', 'anchor')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TEXT,
                    review_note TEXT,
                    FOREIGN KEY(fact_id) REFERENCES semantic_facts(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dream_cycles (
                    dream_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    episodes_processed INTEGER NOT NULL DEFAULT 0,
                    facts_created INTEGER NOT NULL DEFAULT 0,
                    facts_forgotten INTEGER NOT NULL DEFAULT 0,
                    contradictions_flagged INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'running'
                        CHECK(status IN ('running', 'completed', 'failed')),
                    error_message TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS procedures (
                    procedure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK(status IN ('active', 'archived', 'flagged')),
                    source TEXT NOT NULL
                        CHECK(source IN ('human_defined', 'observed_pattern', 'dream_inferred', 'skill_composed')),
                    procedure_confidence REAL NOT NULL
                        CHECK(procedure_confidence >= 0.0 AND procedure_confidence <= 1.0),
                    required_permission TEXT NOT NULL
                        CHECK(required_permission IN ('observe', 'suggest', 'ask_approval', 'execute')),
                    trigger_phrases TEXT NOT NULL,
                    definition_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    archived_at TEXT,
                    UNIQUE(normalized_name, version)
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_procedures_one_active_name
                ON procedures(normalized_name)
                WHERE status = 'active'
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS procedure_executions (
                    execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    procedure_id INTEGER NOT NULL,
                    procedure_version INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'recorded'
                        CHECK(status IN ('recorded', 'completed', 'failed', 'cancelled')),
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    outcome TEXT,
                    FOREIGN KEY(procedure_id) REFERENCES procedures(procedure_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_procedures (
                    pending_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'approved', 'rejected')),
                    proposal_json TEXT NOT NULL,
                    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TEXT,
                    review_note TEXT,
                    approved_procedure_id INTEGER,
                    FOREIGN KEY(approved_procedure_id) REFERENCES procedures(procedure_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_type TEXT NOT NULL
                        CHECK(context_type IN (
                            'person_profile', 'mission', 'venture', 'project',
                            'priority', 'relationship', 'decision', 'constraint',
                            'protocol', 'belief', 'current_bottleneck', 'next_action'
                        )),
                    normalized_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 50
                        CHECK(priority >= 0 AND priority <= 100),
                    source TEXT NOT NULL
                        CHECK(source IN ('julian_prime', 'maya', 'grimbot', 'board', 'portfolio_seed')),
                    verified INTEGER NOT NULL DEFAULT 0 CHECK(verified IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(context_type, normalized_name)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_name TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('active', 'building', 'experiment', 'archived', 'paused')),
                    priority INTEGER NOT NULL
                        CHECK(priority >= 0 AND priority <= 100),
                    current_bottleneck TEXT NOT NULL,
                    next_action TEXT NOT NULL,
                    related_entities TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL
                        CHECK(source IN ('julian_prime', 'maya', 'grimbot', 'board', 'portfolio_seed')),
                    verified INTEGER NOT NULL DEFAULT 0 CHECK(verified IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            from .identity.default_seed import CONTEXT_SEED, PROJECT_SEED

            for context_type, name, content, priority in CONTEXT_SEED:
                normalized_name = " ".join(
                    part for part in "".join(
                        character.lower() if character.isalnum() else " "
                        for character in name
                    ).split()
                )
                connection.execute(
                    """
                    INSERT INTO identity_context
                        (context_type, normalized_name, name, content, priority, source, verified)
                    VALUES (?, ?, ?, ?, ?, 'portfolio_seed', 1)
                    ON CONFLICT(context_type, normalized_name) DO UPDATE SET
                        name = excluded.name,
                        content = excluded.content,
                        priority = excluded.priority,
                        verified = excluded.verified,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE identity_context.source = 'portfolio_seed'
                      AND (
                          identity_context.name != excluded.name
                          OR identity_context.content != excluded.content
                          OR identity_context.priority != excluded.priority
                          OR identity_context.verified != excluded.verified
                      )
                    """,
                    (context_type, normalized_name, name, content, priority),
                )
            for project in PROJECT_SEED:
                normalized_name = " ".join(
                    part for part in "".join(
                        character.lower() if character.isalnum() else " "
                        for character in project["name"]
                    ).split()
                )
                connection.execute(
                    """
                    INSERT INTO identity_projects
                        (normalized_name, name, status, priority, current_bottleneck,
                         next_action, related_entities, source, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'portfolio_seed', 1)
                    ON CONFLICT(normalized_name) DO UPDATE SET
                        name = excluded.name,
                        status = excluded.status,
                        priority = excluded.priority,
                        current_bottleneck = excluded.current_bottleneck,
                        next_action = excluded.next_action,
                        related_entities = excluded.related_entities,
                        verified = excluded.verified,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE identity_projects.source = 'portfolio_seed'
                      AND (
                          identity_projects.name != excluded.name
                          OR identity_projects.status != excluded.status
                          OR identity_projects.priority != excluded.priority
                          OR identity_projects.current_bottleneck != excluded.current_bottleneck
                          OR identity_projects.next_action != excluded.next_action
                          OR identity_projects.related_entities != excluded.related_entities
                          OR identity_projects.verified != excluded.verified
                      )
                    """,
                    (
                        normalized_name,
                        project["name"],
                        project["status"],
                        project["priority"],
                        project["current_bottleneck"],
                        project["next_action"],
                        json.dumps(project["related_entities"]),
                    ),
                )
            connection.commit()

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
