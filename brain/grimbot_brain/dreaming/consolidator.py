from __future__ import annotations

import sqlite3

from .providers.rule_based import Episode, FactCandidate


class Consolidator:
    def __init__(self, provider: object) -> None:
        self.provider = provider

    def load_episodes(self, connection: sqlite3.Connection, limit: int) -> list[Episode]:
        rows = connection.execute(
            """
            SELECT episodic_memories.id,
                   episodic_memories.content,
                   episodic_memories.kind,
                   episodic_memories.importance,
                   episodic_memories.created_at,
                   episodic_memories.anchor,
                   episodic_memories.room_id,
                   episodic_memories.zone_id,
                   rooms.display_name AS room_name,
                   room_zones.display_name AS zone_name
            FROM episodic_memories
            LEFT JOIN rooms ON rooms.id = episodic_memories.room_id
            LEFT JOIN room_zones ON room_zones.id = episodic_memories.zone_id
            WHERE episodic_memories.consolidated = 0
            ORDER BY episodic_memories.created_at ASC, episodic_memories.id ASC
            LIMIT ?
            """,
            (max(1, min(limit, 2000)),),
        ).fetchall()
        return [
            Episode(
                episode_id=int(row["id"]),
                content=row["content"],
                kind=row["kind"],
                importance=float(row["importance"]),
                created_at=row["created_at"],
                anchor=bool(row["anchor"]),
                room_id=row["room_id"],
                zone_id=row["zone_id"],
                room_name=row["room_name"],
                zone_name=row["zone_name"],
            )
            for row in rows
        ]

    def consolidate(self, episodes: list[Episode]) -> list[FactCandidate]:
        return self.provider.consolidate(episodes)

    def contradictions(self, candidates: list[FactCandidate]) -> int:
        return self.provider.contradictions(candidates)
