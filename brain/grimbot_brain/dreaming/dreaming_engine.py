from __future__ import annotations

import hashlib
import json
import re
import sqlite3

from ..memory import BrainMemory
from .consolidator import Consolidator
from .dream_schemas import (
    DreamCycle,
    DreamRunRequest,
    DreamRunResult,
    DreamStatus,
    PromotionQueueItem,
    PromotionReviewRequest,
    SemanticFact,
)
from .forgetter import Forgetter
from .promoter import Promoter, _semantic_fact
from .providers import MockProvider, RuleBasedProvider
from .providers.rule_based import FactCandidate


class DreamCycleConflictError(ValueError):
    """Raised when a second manual dream is requested during an active cycle."""


class DreamingEngine:
    def __init__(self, memory: BrainMemory) -> None:
        self.memory = memory
        self.promoter = Promoter()

    def run(self, request: DreamRunRequest) -> DreamRunResult:
        dream_id = self._start_cycle()

        try:
            provider = RuleBasedProvider() if request.provider == "rule_based" else MockProvider()
            consolidator = Consolidator(provider)
            with self._connect() as connection:
                episodes = consolidator.load_episodes(connection, request.episode_limit)
                candidates = consolidator.consolidate(episodes)
                contradictions = consolidator.contradictions(candidates)
                facts, facts_created, promotions_created = self._store_candidates(connection, candidates)
                facts_forgotten = Forgetter().forget_stale_facts(connection) if request.run_forgetting else 0
                connection.execute(
                    """
                    UPDATE dream_cycles
                    SET completed_at = CURRENT_TIMESTAMP,
                        episodes_processed = ?,
                        facts_created = ?,
                        facts_forgotten = ?,
                        contradictions_flagged = ?,
                        status = 'completed'
                    WHERE dream_id = ?
                    """,
                    (len(episodes), facts_created, facts_forgotten, contradictions, dream_id),
                )
                connection.commit()
                cycle = self._cycle_by_id(connection, dream_id)
            return DreamRunResult(
                cycle=cycle,
                candidate_facts=facts,
                promotions_created=promotions_created,
            )
        except Exception as exc:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE dream_cycles
                    SET completed_at = CURRENT_TIMESTAMP,
                        status = 'failed',
                        error_message = ?
                    WHERE dream_id = ?
                    """,
                    (str(exc)[:1000], dream_id),
                )
                connection.commit()
            raise

    def status(self) -> DreamStatus:
        with self._connect() as connection:
            self._mark_stale_cycles(connection)
            connection.commit()
            row = connection.execute(
                "SELECT * FROM dream_cycles ORDER BY dream_id DESC LIMIT 1"
            ).fetchone()
        cycle = self._cycle(row) if row else None
        return DreamStatus(
            active=bool(cycle and cycle.status == "running"),
            latest_cycle=cycle,
        )

    def facts(self) -> list[SemanticFact]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM semantic_facts
                ORDER BY
                    CASE tier WHEN 'core' THEN 0 ELSE 1 END,
                    confidence DESC,
                    last_reinforced DESC,
                    id DESC
                """
            ).fetchall()
        return [_semantic_fact(row) for row in rows]

    def promotions(self) -> list[PromotionQueueItem]:
        with self._connect() as connection:
            return self.promoter.list_promotions(connection)

    def approve(self, promotion_id: int, request: PromotionReviewRequest) -> PromotionQueueItem:
        with self._connect() as connection:
            return self.promoter.review(connection, promotion_id, request, approved=True)

    def reject(self, promotion_id: int, request: PromotionReviewRequest) -> PromotionQueueItem:
        if request.anchor:
            raise ValueError("Rejected promotions cannot be anchors")
        with self._connect() as connection:
            return self.promoter.review(connection, promotion_id, request, approved=False)

    def _store_candidates(
        self,
        connection: sqlite3.Connection,
        candidates: list[FactCandidate],
    ) -> tuple[list[SemanticFact], int, int]:
        facts: list[SemanticFact] = []
        facts_created = 0
        promotions_created = 0
        for candidate in candidates:
            fact_key = _candidate_key(candidate)
            row = connection.execute(
                """
                SELECT *
                FROM semantic_facts
                WHERE room_id IS ? AND zone_id IS ? AND fact_key = ?
                """,
                (candidate.room_id, candidate.zone_id, fact_key),
            ).fetchone()
            if row:
                improved = (
                    candidate.confidence > float(row["confidence"])
                    or candidate.importance > float(row["importance"])
                    or candidate.frequency > int(row["count"])
                )
                if improved:
                    connection.execute(
                        """
                        UPDATE semantic_facts
                        SET confidence = MIN(1.0, MAX(confidence, ?)),
                            importance = MAX(importance, ?),
                            count = MIN(1000000, MAX(count, ?)),
                            last_seen = CURRENT_TIMESTAMP,
                            last_reinforced = CURRENT_TIMESTAMP,
                            tags = ?
                        WHERE id = ?
                        """,
                        (
                            candidate.confidence,
                            candidate.importance,
                            candidate.frequency,
                            json.dumps(candidate.tags),
                            row["id"],
                        ),
                    )
                fact_id = int(row["id"])
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO semantic_facts (
                        room_id, zone_id, fact_key, content, confidence, importance, count,
                        first_seen, last_seen, created_at, last_reinforced, tags, tier
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, 'semantic')
                    """,
                    (
                        candidate.room_id,
                        candidate.zone_id,
                        fact_key,
                        candidate.content,
                        candidate.confidence,
                        candidate.importance,
                        candidate.frequency,
                        json.dumps(candidate.tags),
                    ),
                )
                fact_id = int(cursor.lastrowid)
                facts_created += 1

            existing_queue = connection.execute(
                "SELECT id FROM promotion_queue WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if not existing_queue:
                connection.execute(
                    "INSERT INTO promotion_queue (fact_id, status) VALUES (?, 'pending')",
                    (fact_id,),
                )
                promotions_created += 1
            fact_row = connection.execute("SELECT * FROM semantic_facts WHERE id = ?", (fact_id,)).fetchone()
            facts.append(_semantic_fact(fact_row))
        return facts, facts_created, promotions_created

    def _cycle_by_id(self, connection: sqlite3.Connection, dream_id: int) -> DreamCycle:
        row = connection.execute(
            "SELECT * FROM dream_cycles WHERE dream_id = ?",
            (dream_id,),
        ).fetchone()
        return self._cycle(row)

    def _start_cycle(self) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._mark_stale_cycles(connection)
            active = connection.execute(
                "SELECT dream_id FROM dream_cycles WHERE status = 'running' LIMIT 1"
            ).fetchone()
            if active:
                connection.rollback()
                raise DreamCycleConflictError(f"Dream cycle {active['dream_id']} is already running")
            cursor = connection.execute("INSERT INTO dream_cycles (status) VALUES ('running')")
            connection.commit()
            return int(cursor.lastrowid)

    def _mark_stale_cycles(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE dream_cycles
            SET completed_at = CURRENT_TIMESTAMP,
                status = 'failed',
                error_message = 'Dream cycle interrupted before completion'
            WHERE status = 'running'
              AND started_at < datetime('now', '-1 hour')
            """
        )

    def _cycle(self, row: sqlite3.Row) -> DreamCycle:
        return DreamCycle(
            dream_id=int(row["dream_id"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            episodes_processed=int(row["episodes_processed"]),
            facts_created=int(row["facts_created"]),
            facts_forgotten=int(row["facts_forgotten"]),
            contradictions_flagged=int(row["contradictions_flagged"]),
            status=row["status"],
            error_message=row["error_message"],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.memory.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _candidate_key(candidate: FactCandidate) -> str:
    payload = "|".join(
        (
            str(candidate.room_id),
            str(candidate.zone_id),
            re.sub(r"[^a-z0-9]+", " ", candidate.content.lower()).strip(),
        )
    )
    return f"dream:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"
