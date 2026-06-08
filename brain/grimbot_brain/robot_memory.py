from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass

from .memory import BrainMemory
from .schemas import (
    MemoryRecord,
    RelevantMemoryRequest,
    RelevantMemoryResult,
    RememberRequest,
    RoomMemorySummary,
    RoomScanResult,
)


DEFAULT_ROOM = "unknown room"
DEFAULT_ZONE = "general"
MEMORY_TABLES = {"room_zones", "known_objects", "hazards", "mess_observations", "cleanup_tasks"}


@dataclass(frozen=True)
class LocationIds:
    room_id: int | None
    zone_id: int | None


class RobotMemory:
    def __init__(self, memory: BrainMemory) -> None:
        self.memory = memory

    def remember(self, request: RememberRequest) -> dict:
        with self._connect() as connection:
            location = self._ensure_location(connection, request.room_name, request.zone_name, request.importance)
            episodic_id = self._insert_episodic(
                connection,
                location,
                kind=request.kind,
                content=request.text,
                importance=request.importance,
            )
            fact = self._upsert_semantic_fact(connection, location, request.text, request.importance)
            connection.commit()

        return {"episodic_memory_id": episodic_id, "semantic_fact": fact}

    def ingest_room_scan(
        self,
        scan: RoomScanResult,
        room_name: str | None = None,
        zone_name: str | None = None,
    ) -> None:
        with self._connect() as connection:
            location = self._ensure_location(connection, room_name or DEFAULT_ROOM, zone_name or DEFAULT_ZONE, 0.6)
            self._insert_episodic(connection, location, "observation", scan.room_summary, 0.55)

            for visible_object in scan.visible_objects:
                self._upsert_observation(connection, "known_objects", location, visible_object, 0.6)
            for hazard in scan.hazards:
                self._upsert_observation(connection, "hazards", location, hazard, 0.9)
            for mess_zone in scan.mess_zones:
                self._upsert_observation(connection, "mess_observations", location, mess_zone, 0.7)
            for cleanup_task in scan.suggested_cleanup_order:
                self._upsert_observation(connection, "cleanup_tasks", location, cleanup_task, 0.75)

            connection.commit()

    def list_rooms(self) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, display_name AS name, NULL AS room_name, NULL AS zone_name,
                       count, confidence, importance, first_seen, last_seen
                FROM rooms
                ORDER BY importance DESC, count DESC, last_seen DESC
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def room_summary(self, room_name: str) -> RoomMemorySummary:
        with self._connect() as connection:
            room = self._get_room(connection, room_name)
            if not room:
                return RoomMemorySummary(
                    room_name=room_name,
                    recommended_first_cleanup_action="scan this room",
                )

            room_id = int(room["id"])
            zones = self._records(connection, "room_zones", room_id=room_id)
            objects = self._records(connection, "known_objects", room_id=room_id)
            hazards = self._records(connection, "hazards", room_id=room_id)
            mess_zones = self._records(connection, "mess_observations", room_id=room_id)
            cleanup_tasks = self._records(connection, "cleanup_tasks", room_id=room_id)
            episodic = self._episodic(connection, room_id)
            facts = self._facts(connection, room_id)

        return RoomMemorySummary(
            room_name=room["display_name"],
            zones=zones,
            known_objects=objects,
            hazards=hazards,
            mess_zones=mess_zones,
            cleanup_tasks=cleanup_tasks,
            episodic_memories=episodic,
            semantic_facts=facts,
            recommended_first_cleanup_action=self._next_action(hazards, mess_zones, cleanup_tasks),
        )

    def hazards(self, room_name: str | None = None, zone_name: str | None = None, limit: int = 20) -> list[MemoryRecord]:
        return self._filtered_records("hazards", room_name, zone_name, limit)

    def mess_zones(self, room_name: str | None = None, zone_name: str | None = None, limit: int = 20) -> list[MemoryRecord]:
        return self._filtered_records("mess_observations", room_name, zone_name, limit)

    def relevant(self, request: RelevantMemoryRequest) -> RelevantMemoryResult:
        state_values = request.adaptive_state or {}
        urgency = _state_value(state_values, "urgency")
        curiosity = _state_value(state_values, "curiosity")
        retrieval_limit = min(50, request.limit + 5) if urgency >= 0.65 or curiosity >= 0.65 else request.limit
        hazards = self.hazards(request.room_name, request.zone_name, retrieval_limit)
        mess_zones = self.mess_zones(request.room_name, request.zone_name, retrieval_limit)
        with self._connect() as connection:
            room_id = self._room_id(connection, request.room_name)
            if request.room_name and room_id is None:
                cleanup_tasks = []
                facts = []
            else:
                zone_id = self._zone_id(connection, room_id, request.zone_name)
                if request.zone_name and zone_id is None:
                    cleanup_tasks = []
                    facts = []
                else:
                    cleanup_tasks = self._records(
                        connection,
                        "cleanup_tasks",
                        room_id=room_id,
                        zone_id=zone_id,
                        limit=retrieval_limit,
                    )
                    facts = self._facts(connection, room_id, limit=retrieval_limit)

        return RelevantMemoryResult(
            query=request.query,
            room_name=request.room_name,
            hazards=hazards,
            mess_zones=mess_zones,
            cleanup_tasks=cleanup_tasks,
            semantic_facts=facts,
            next_best_action=self._state_next_action(state_values, hazards, mess_zones, cleanup_tasks),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.memory.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_location(
        self,
        connection: sqlite3.Connection,
        room_name: str | None,
        zone_name: str | None,
        importance: float,
    ) -> LocationIds:
        room_id = self._ensure_room(connection, room_name or DEFAULT_ROOM, importance)
        zone_id = self._ensure_zone(connection, room_id, zone_name or DEFAULT_ZONE, importance)
        return LocationIds(room_id=room_id, zone_id=zone_id)

    def _ensure_room(self, connection: sqlite3.Connection, room_name: str, importance: float) -> int:
        key = _normalize(room_name)
        row = connection.execute("SELECT id FROM rooms WHERE name = ?", (key,)).fetchone()
        if row:
            connection.execute(
                """
                UPDATE rooms
                SET count = count + 1,
                    confidence = MIN(1.0, confidence + 0.03),
                    importance = MAX(importance, ?),
                    last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (importance, row["id"]),
            )
            return int(row["id"])

        cursor = connection.execute(
            """
            INSERT INTO rooms (name, display_name, importance)
            VALUES (?, ?, ?)
            """,
            (key, _display_name(room_name, DEFAULT_ROOM), importance),
        )
        return int(cursor.lastrowid)

    def _ensure_zone(self, connection: sqlite3.Connection, room_id: int, zone_name: str, importance: float) -> int:
        key = _normalize(zone_name)
        row = connection.execute(
            "SELECT id FROM room_zones WHERE room_id = ? AND name = ?",
            (room_id, key),
        ).fetchone()
        if row:
            connection.execute(
                """
                UPDATE room_zones
                SET count = count + 1,
                    confidence = MIN(1.0, confidence + 0.03),
                    importance = MAX(importance, ?),
                    last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (importance, row["id"]),
            )
            return int(row["id"])

        cursor = connection.execute(
            """
            INSERT INTO room_zones (room_id, name, display_name, importance)
            VALUES (?, ?, ?, ?)
            """,
            (room_id, key, _display_name(zone_name, DEFAULT_ZONE), importance),
        )
        return int(cursor.lastrowid)

    def _upsert_observation(
        self,
        connection: sqlite3.Connection,
        table: str,
        location: LocationIds,
        name: str,
        importance: float,
    ) -> MemoryRecord:
        if table not in MEMORY_TABLES - {"room_zones"}:
            raise ValueError(f"Unsupported memory table: {table}")

        key = _normalize(name)
        row = connection.execute(
            f"SELECT id FROM {table} WHERE room_id = ? AND zone_id = ? AND name = ?",
            (location.room_id, location.zone_id, key),
        ).fetchone()
        if row:
            connection.execute(
                f"""
                UPDATE {table}
                SET count = count + 1,
                    confidence = MIN(1.0, confidence + 0.05),
                    importance = MAX(importance, ?),
                    last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (importance, row["id"]),
            )
            record_id = int(row["id"])
        else:
            cursor = connection.execute(
                f"""
                INSERT INTO {table} (room_id, zone_id, name, display_name, importance)
                VALUES (?, ?, ?, ?, ?)
                """,
                (location.room_id, location.zone_id, key, _display_name(name, "unknown item"), importance),
            )
            record_id = int(cursor.lastrowid)

        return self._record_by_id(connection, table, record_id)

    def _insert_episodic(
        self,
        connection: sqlite3.Connection,
        location: LocationIds,
        kind: str,
        content: str,
        importance: float,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO episodic_memories (room_id, zone_id, kind, content, importance)
            VALUES (?, ?, ?, ?, ?)
            """,
            (location.room_id, location.zone_id, kind, content.strip(), importance),
        )
        return int(cursor.lastrowid)

    def _upsert_semantic_fact(
        self,
        connection: sqlite3.Connection,
        location: LocationIds,
        content: str,
        importance: float,
    ) -> dict:
        fact_key = _normalize(content)[:120]
        row = connection.execute(
            "SELECT id FROM semantic_facts WHERE room_id = ? AND zone_id = ? AND fact_key = ?",
            (location.room_id, location.zone_id, fact_key),
        ).fetchone()
        if row:
            connection.execute(
                """
                UPDATE semantic_facts
                SET count = count + 1,
                    confidence = MIN(1.0, confidence + 0.04),
                    importance = MAX(importance, ?),
                    last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (importance, row["id"]),
            )
            fact_id = int(row["id"])
        else:
            cursor = connection.execute(
                """
                INSERT INTO semantic_facts (room_id, zone_id, fact_key, content, importance)
                VALUES (?, ?, ?, ?, ?)
                """,
                (location.room_id, location.zone_id, fact_key, content.strip(), importance),
            )
            fact_id = int(cursor.lastrowid)

        return dict(connection.execute("SELECT * FROM semantic_facts WHERE id = ?", (fact_id,)).fetchone())

    def _filtered_records(
        self,
        table: str,
        room_name: str | None,
        zone_name: str | None,
        limit: int,
    ) -> list[MemoryRecord]:
        with self._connect() as connection:
            room_id = self._room_id(connection, room_name)
            if room_name and room_id is None:
                return []

            zone_id = self._zone_id(connection, room_id, zone_name)
            if zone_name and zone_id is None:
                return []

            return self._records(connection, table, room_id=room_id, zone_id=zone_id, limit=limit)

    def _records(
        self,
        connection: sqlite3.Connection,
        table: str,
        room_id: int | None = None,
        zone_id: int | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        if table not in MEMORY_TABLES:
            raise ValueError(f"Unsupported memory table: {table}")

        if table == "room_zones":
            return self._zone_records(connection, room_id, limit)

        conditions = []
        params: list[object] = []
        if room_id is not None:
            conditions.append(f"{table}.room_id = ?")
            params.append(room_id)
        if zone_id is not None:
            conditions.append(f"{table}.zone_id = ?")
            params.append(zone_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = connection.execute(
            f"""
            SELECT {table}.id,
                   {table}.display_name AS name,
                   rooms.display_name AS room_name,
                   room_zones.display_name AS zone_name,
                   {table}.count,
                   {table}.confidence,
                   {table}.importance,
                   {table}.first_seen,
                   {table}.last_seen
            FROM {table}
            LEFT JOIN rooms ON rooms.id = {table}.room_id
            LEFT JOIN room_zones ON room_zones.id = {table}.zone_id
            {where}
            ORDER BY {table}.importance DESC, {table}.count DESC, {table}.last_seen DESC
            LIMIT ?
            """,
            (*params, max(1, min(limit, 100))),
        ).fetchall()
        return [self._record(row) for row in rows]

    def _zone_records(
        self,
        connection: sqlite3.Connection,
        room_id: int | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        conditions = []
        params: list[object] = []
        if room_id is not None:
            conditions.append("room_zones.room_id = ?")
            params.append(room_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = connection.execute(
            f"""
            SELECT room_zones.id,
                   room_zones.display_name AS name,
                   rooms.display_name AS room_name,
                   room_zones.display_name AS zone_name,
                   room_zones.count,
                   room_zones.confidence,
                   room_zones.importance,
                   room_zones.first_seen,
                   room_zones.last_seen
            FROM room_zones
            LEFT JOIN rooms ON rooms.id = room_zones.room_id
            {where}
            ORDER BY room_zones.importance DESC, room_zones.count DESC, room_zones.last_seen DESC
            LIMIT ?
            """,
            (*params, max(1, min(limit, 100))),
        ).fetchall()
        return [self._record(row) for row in rows]

    def _record_by_id(self, connection: sqlite3.Connection, table: str, record_id: int) -> MemoryRecord:
        row = connection.execute(
            f"""
            SELECT {table}.id,
                   {table}.display_name AS name,
                   rooms.display_name AS room_name,
                   room_zones.display_name AS zone_name,
                   {table}.count,
                   {table}.confidence,
                   {table}.importance,
                   {table}.first_seen,
                   {table}.last_seen
            FROM {table}
            LEFT JOIN rooms ON rooms.id = {table}.room_id
            LEFT JOIN room_zones ON room_zones.id = {table}.zone_id
            WHERE {table}.id = ?
            """,
            (record_id,),
        ).fetchone()
        return self._record(row)

    def _record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            name=row["name"],
            room_name=row["room_name"],
            zone_name=row["zone_name"],
            count=int(row["count"]),
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )

    def _episodic(self, connection: sqlite3.Connection, room_id: int, limit: int = 10) -> list[dict]:
        rows = connection.execute(
            """
            SELECT id, kind, content, importance, created_at
            FROM episodic_memories
            WHERE room_id = ?
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
            """,
            (room_id, max(1, min(limit, 50))),
        ).fetchall()
        return [dict(row) for row in rows]

    def _facts(self, connection: sqlite3.Connection, room_id: int | None, limit: int = 10) -> list[dict]:
        if room_id is None:
            rows = connection.execute(
                """
                SELECT id, content, confidence, importance, count, first_seen, last_seen
                FROM semantic_facts
                ORDER BY importance DESC, count DESC, last_seen DESC
                LIMIT ?
                """,
                (max(1, min(limit, 50)),),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, content, confidence, importance, count, first_seen, last_seen
                FROM semantic_facts
                WHERE room_id = ?
                ORDER BY importance DESC, count DESC, last_seen DESC
                LIMIT ?
                """,
                (room_id, max(1, min(limit, 50))),
            ).fetchall()
        return [dict(row) for row in rows]

    def _get_room(self, connection: sqlite3.Connection, room_name: str) -> sqlite3.Row | None:
        return connection.execute("SELECT * FROM rooms WHERE name = ?", (_normalize(room_name),)).fetchone()

    def _room_id(self, connection: sqlite3.Connection, room_name: str | None) -> int | None:
        if not room_name:
            return None
        row = self._get_room(connection, room_name)
        return int(row["id"]) if row else None

    def _zone_id(self, connection: sqlite3.Connection, room_id: int | None, zone_name: str | None) -> int | None:
        if room_id is None or not zone_name:
            return None
        row = connection.execute(
            "SELECT id FROM room_zones WHERE room_id = ? AND name = ?",
            (room_id, _normalize(zone_name)),
        ).fetchone()
        return int(row["id"]) if row else None

    def _next_action(
        self,
        hazards: list[MemoryRecord],
        mess_zones: list[MemoryRecord],
        cleanup_tasks: list[MemoryRecord],
    ) -> str:
        if hazards:
            return f"clear hazard: {hazards[0].name}"
        if cleanup_tasks:
            return cleanup_tasks[0].name
        if mess_zones:
            return f"clean recurring mess zone: {mess_zones[0].name}"
        return "scan room for current conditions"

    def _state_next_action(
        self,
        state_values: dict[str, float],
        hazards: list[MemoryRecord],
        mess_zones: list[MemoryRecord],
        cleanup_tasks: list[MemoryRecord],
    ) -> str:
        base_action = self._next_action(hazards, mess_zones, cleanup_tasks)
        if _state_value(state_values, "urgency") >= 0.65 and hazards:
            return f"{base_action} before lower-priority organization"
        if _state_value(state_values, "friction") >= 0.60 and mess_zones:
            return f"pick one small cleanup step for: {mess_zones[0].name}"
        if _state_value(state_values, "curiosity") >= 0.65 and not hazards:
            return "review recent room memory before choosing a cleanup task"
        return base_action


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return normalized or "unknown"


def _display_name(value: str, fallback: str) -> str:
    display = value.strip()
    return display[:120] if display else fallback


def _state_value(values: dict[str, float], key: str) -> float:
    try:
        value = float(values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
