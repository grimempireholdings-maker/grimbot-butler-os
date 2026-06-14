from __future__ import annotations

import json
import math
import sqlite3

from .dream_schemas import PromotionQueueItem, PromotionReviewRequest, SemanticFact


class Promoter:
    def list_promotions(self, connection: sqlite3.Connection) -> list[PromotionQueueItem]:
        rows = connection.execute(
            """
            SELECT promotion_queue.id AS promotion_id,
                   promotion_queue.status,
                   promotion_queue.created_at AS promotion_created_at,
                   promotion_queue.reviewed_at,
                   promotion_queue.review_note,
                   semantic_facts.*
            FROM promotion_queue
            JOIN semantic_facts ON semantic_facts.id = promotion_queue.fact_id
            ORDER BY
                CASE promotion_queue.status WHEN 'pending' THEN 0 ELSE 1 END,
                promotion_queue.id DESC
            """
        ).fetchall()
        return [self._promotion(row) for row in rows]

    def review(
        self,
        connection: sqlite3.Connection,
        promotion_id: int,
        request: PromotionReviewRequest,
        approved: bool,
    ) -> PromotionQueueItem:
        row = connection.execute(
            """
            SELECT promotion_queue.status, promotion_queue.fact_id
            FROM promotion_queue
            JOIN semantic_facts ON semantic_facts.id = promotion_queue.fact_id
            WHERE promotion_queue.id = ?
            """,
            (promotion_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Unknown promotion: {promotion_id}")
        if row["status"] != "pending":
            raise ValueError("Promotion has already been reviewed")

        status = "anchor" if approved and request.anchor else "approved" if approved else "rejected"
        cursor = connection.execute(
            """
            UPDATE promotion_queue
            SET status = ?,
                reviewed_at = CURRENT_TIMESTAMP,
                review_note = ?
            WHERE id = ? AND status = 'pending'
            """,
            (status, request.note, promotion_id),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise ValueError("Promotion has already been reviewed")
        if approved:
            connection.execute(
                """
                UPDATE semantic_facts
                SET tier = ?,
                    last_reinforced = CURRENT_TIMESTAMP,
                    last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                ("core" if request.anchor else "semantic", row["fact_id"]),
            )
        connection.commit()

        reviewed = connection.execute(
            """
            SELECT promotion_queue.id AS promotion_id,
                   promotion_queue.status,
                   promotion_queue.created_at AS promotion_created_at,
                   promotion_queue.reviewed_at,
                   promotion_queue.review_note,
                   semantic_facts.*
            FROM promotion_queue
            JOIN semantic_facts ON semantic_facts.id = promotion_queue.fact_id
            WHERE promotion_queue.id = ?
            """,
            (promotion_id,),
        ).fetchone()
        return self._promotion(reviewed)

    def _promotion(self, row: sqlite3.Row) -> PromotionQueueItem:
        return PromotionQueueItem(
            id=int(row["promotion_id"]),
            fact_id=int(row["id"]),
            status=row["status"],
            created_at=row["promotion_created_at"],
            reviewed_at=row["reviewed_at"],
            review_note=row["review_note"],
            fact=_semantic_fact(row),
        )


def _semantic_fact(row: sqlite3.Row) -> SemanticFact:
    try:
        tags = json.loads(row["tags"] or "[]")
    except (json.JSONDecodeError, TypeError):
        tags = []
    confidence = float(row["confidence"])
    if not math.isfinite(confidence):
        confidence = 0.0
    return SemanticFact(
        fact_id=int(row["id"]),
        content=str(row["content"])[:2000] or "Unknown fact",
        confidence=max(0.0, min(1.0, confidence)),
        created_at=str(row["created_at"] or row["first_seen"]),
        last_reinforced=str(row["last_reinforced"] or row["last_seen"]),
        tags=[str(tag)[:120] for tag in tags[:30]] if isinstance(tags, list) else [],
        tier="core" if row["tier"] == "core" else "semantic",
    )
