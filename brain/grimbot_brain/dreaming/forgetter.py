from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timezone


PROTECTED_WORDS = {
    "battery",
    "cable",
    "collision",
    "cord",
    "danger",
    "emergency",
    "fire",
    "hazard",
    "obstacle",
    "safety",
    "smoke",
    "spill",
    "unsafe",
}


class Forgetter:
    def __init__(self, threshold: float = 0.18) -> None:
        self.threshold = max(0.0, min(1.0, threshold))

    def score(self, importance: float, frequency: int, last_reinforced: str) -> float:
        importance_score = max(0.0, min(1.0, _safe_float(importance)))
        frequency_score = min(1.0, max(0, _safe_int(frequency)) / 5.0)
        reinforced_at = _parse_time(last_reinforced)
        age_days = max(0.0, (_now() - reinforced_at).total_seconds() / 86400.0)
        recency_score = math.exp(-age_days / 90.0)
        return importance_score * 0.5 + frequency_score * 0.3 + recency_score * 0.2

    def forget_stale_facts(self, connection: sqlite3.Connection) -> int:
        rows = connection.execute(
            """
            SELECT semantic_facts.*,
                   promotion_queue.status AS promotion_status
            FROM semantic_facts
            LEFT JOIN promotion_queue ON promotion_queue.fact_id = semantic_facts.id
            """
        ).fetchall()
        forgotten = 0
        for row in rows:
            if self._protected(row):
                continue
            score = self.score(
                importance=float(row["importance"]),
                frequency=int(row["count"]),
                last_reinforced=row["last_reinforced"] or row["last_seen"],
            )
            if score >= self.threshold:
                continue
            connection.execute(
                "DELETE FROM promotion_queue WHERE fact_id = ? AND status = 'rejected'",
                (row["id"],),
            )
            connection.execute("DELETE FROM semantic_facts WHERE id = ?", (row["id"],))
            forgotten += 1
        return forgotten

    def _protected(self, row: sqlite3.Row) -> bool:
        if row["tier"] == "core":
            return True
        if row["promotion_status"] in {"pending", "approved", "anchor"}:
            return True
        tags = {
            word
            for tag in _parse_tags(row["tags"])
            for word in re.findall(r"[a-z0-9]+", str(tag).lower())
        }
        words = set(re.findall(r"[a-z0-9]+", str(row["content"]).lower()))
        return bool(PROTECTED_WORDS & (tags | words))


def _parse_tags(value: str) -> list[str]:
    try:
        tags = json.loads(value or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return tags if isinstance(tags, list) else []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return _now()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
