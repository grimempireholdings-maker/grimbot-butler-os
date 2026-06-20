from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timezone

from ..memory import BrainMemory
from .context_schemas import (
    ContextEntry,
    ContextRememberRequest,
    ContextSearchRequest,
    ContextSearchResult,
    ContextSummary,
    PriorityUpdateRequest,
    ProjectContext,
)


class ContextStore:
    def __init__(self, memory: BrainMemory) -> None:
        self.memory = memory

    def summary(self) -> ContextSummary:
        return ContextSummary(
            person_profile=self.entries("person_profile"),
            primary_location=self.primary_location(),
            missions=self.entries("mission"),
            ventures=self.entries("venture"),
            projects=self.projects(),
            priorities=self.priorities(),
            relationships=self.relationships(),
            bottlenecks=self.entries("current_bottleneck"),
            next_actions=self.entries("next_action"),
            protocols=self.entries("protocol"),
        )

    def primary_location(self) -> str | None:
        """Return Julian's verified profile location; never infer or geolocate it."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT content FROM identity_context
                WHERE context_type = 'person_profile'
                  AND normalized_name = 'primary location'
                  AND verified = 1
                LIMIT 1
                """
            ).fetchone()
        return str(row["content"]).strip() if row and str(row["content"]).strip() else None

    def entries(self, context_type: str, limit: int = 50) -> list[ContextEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM identity_context
                WHERE context_type = ?
                ORDER BY priority DESC, last_updated DESC, id ASC
                LIMIT ?
                """,
                (context_type, max(1, min(limit, 50))),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def projects(self) -> list[ProjectContext]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM identity_projects
                WHERE status != 'archived'
                ORDER BY priority DESC, last_updated DESC, id ASC
                """
            ).fetchall()
        return [self._project(row) for row in rows]

    def priorities(self) -> list[ContextEntry]:
        return self.entries("priority")

    def relationships(self) -> list[ContextEntry]:
        return self.entries("relationship")

    def search(self, request: ContextSearchRequest) -> ContextSearchResult:
        stopwords = {
            "a", "about", "and", "are", "do", "for", "how", "i", "is", "me",
            "current", "like", "looking", "my", "of", "on", "please", "project",
            "remember", "tell", "the", "this", "to", "want", "what", "would", "you",
        }
        terms = [
            term
            for term in _normalize(request.query).split()
            if len(term) > 1 and term not in stopwords
        ]
        entries = self._search_entries(terms, request.context_types, request.limit)
        projects = (
            self._search_projects(terms, request.limit)
            if not request.context_types or "project" in request.context_types
            else []
        )
        next_action = ""
        if projects:
            next_action = projects[0].next_action
        elif entries:
            next_action = (
                entries[0].content
                if entries[0].context_type == "next_action"
                else "Review this context against current priorities before acting."
            )
        else:
            next_action = "Ask one clarifying question before recommending action."
        return ContextSearchResult(
            query=request.query,
            entries=entries,
            projects=projects,
            next_best_action=next_action,
            needs_clarification=not entries and not projects,
            clarification_question=(
                "Which project, priority, person, or decision do you want me to focus on?"
                if not entries and not projects
                else None
            ),
        )

    def remember(self, request: ContextRememberRequest) -> ContextEntry:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO identity_context
                    (context_type, normalized_name, name, content, priority, source,
                     verified, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(context_type, normalized_name) DO UPDATE SET
                    name = excluded.name,
                    content = excluded.content,
                    priority = excluded.priority,
                    source = excluded.source,
                    verified = excluded.verified,
                    last_updated = excluded.last_updated
                """,
                (
                    request.context_type,
                    _normalize(request.name),
                    request.name.strip(),
                    request.content.strip(),
                    request.priority,
                    request.source,
                    int(request.verified),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM identity_context WHERE context_type = ? AND normalized_name = ?",
                (request.context_type, _normalize(request.name)),
            ).fetchone()
            connection.commit()
        return self._entry(row)

    def update_priority(self, request: PriorityUpdateRequest) -> ProjectContext:
        normalized = _normalize(request.name)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM identity_projects WHERE normalized_name = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown project: {request.name}")
            connection.execute(
                """
                UPDATE identity_projects
                SET priority = ?,
                    status = COALESCE(?, status),
                    current_bottleneck = COALESCE(?, current_bottleneck),
                    next_action = COALESCE(?, next_action),
                    source = 'julian_prime',
                    verified = ?,
                    last_updated = ?
                WHERE normalized_name = ?
                """,
                (
                    request.priority,
                    request.status,
                    _clean_optional(request.current_bottleneck),
                    _clean_optional(request.next_action),
                    int(request.verified),
                    _now(),
                    normalized,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM identity_projects WHERE normalized_name = ?",
                (normalized,),
            ).fetchone()
            connection.commit()
        return self._project(updated)

    def _search_entries(
        self,
        terms: list[str],
        context_types: list[str] | None,
        limit: int,
    ) -> list[ContextEntry]:
        if not terms:
            return []
        filters = []
        params: list[object] = []
        for term in terms:
            filters.append("(normalized_name LIKE ? OR lower(content) LIKE ?)")
            params.extend((f"%{term}%", f"%{term}%"))
        type_clause = ""
        if context_types:
            placeholders = ",".join("?" for _ in context_types)
            type_clause = f" AND context_type IN ({placeholders})"
            params.extend(context_types)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM identity_context
                WHERE ({" OR ".join(filters)}){type_clause}
                ORDER BY priority DESC, last_updated DESC
                LIMIT 100
                """,
                params,
            ).fetchall()
        entries = [self._entry(row) for row in rows]
        ranked = [
            (entry, _entry_relevance(entry, terms))
            for entry in entries
        ]
        minimum_matches = max(1, math.ceil(len(terms) / 2))
        ranked = [
            (entry, score)
            for entry, score in ranked
            if score[1] >= minimum_matches
        ]
        ranked.sort(key=lambda item: (-item[1][0], -item[0].priority, item[0].name.lower()))
        return [entry for entry, _ in ranked[: max(1, min(limit, 50))]]

    def _search_projects(self, terms: list[str], limit: int) -> list[ProjectContext]:
        if not terms:
            return []
        filters = []
        params: list[object] = []
        for term in terms:
            filters.append(
                "(normalized_name LIKE ? OR lower(current_bottleneck) LIKE ? "
                "OR lower(next_action) LIKE ? OR lower(related_entities) LIKE ?)"
            )
            params.extend((f"%{term}%", f"%{term}%", f"%{term}%", f"%{term}%"))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM identity_projects
                WHERE status != 'archived' AND ({" OR ".join(filters)})
                ORDER BY priority DESC, last_updated DESC
                LIMIT 100
                """,
                params,
            ).fetchall()
        projects = [self._project(row) for row in rows]
        ranked = [
            (project, _project_relevance(project, terms))
            for project in projects
        ]
        minimum_matches = max(1, math.ceil(len(terms) / 2))
        ranked = [
            (project, score)
            for project, score in ranked
            if score[1] >= minimum_matches
        ]
        ranked.sort(key=lambda item: (-item[1][0], -item[0].priority, item[0].name.lower()))
        return [project for project, _ in ranked[: max(1, min(limit, 50))]]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.memory.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _entry(row: sqlite3.Row) -> ContextEntry:
        return ContextEntry(
            id=row["id"],
            context_type=row["context_type"],
            name=row["name"],
            content=row["content"],
            priority=row["priority"],
            source=row["source"],
            verified=bool(row["verified"]),
            created_at=row["created_at"],
            last_updated=row["last_updated"],
        )

    @staticmethod
    def _project(row: sqlite3.Row) -> ProjectContext:
        return ProjectContext(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            priority=row["priority"],
            current_bottleneck=row["current_bottleneck"],
            next_action=row["next_action"],
            last_updated=row["last_updated"],
            related_entities=json.loads(row["related_entities"]),
            source=row["source"],
            verified=bool(row["verified"]),
        )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _entry_relevance(entry: ContextEntry, terms: list[str]) -> tuple[int, int]:
    name_tokens = set(_normalize(entry.name).split())
    content_tokens = set(_normalize(entry.content).split())
    matched = {term for term in terms if term in name_tokens or term in content_tokens}
    score = sum(5 if term in name_tokens else 1 for term in matched)
    return score, len(matched)


def _project_relevance(project: ProjectContext, terms: list[str]) -> tuple[int, int]:
    name_tokens = set(_normalize(project.name).split())
    related_tokens = set(_normalize(" ".join(project.related_entities)).split())
    detail_tokens = set(
        _normalize(f"{project.current_bottleneck} {project.next_action}").split()
    )
    matched = {
        term
        for term in terms
        if term in name_tokens or term in related_tokens or term in detail_tokens
    }
    score = sum(
        6 if term in name_tokens else 2 if term in related_tokens else 1
        for term in matched
    )
    return score, len(matched)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
