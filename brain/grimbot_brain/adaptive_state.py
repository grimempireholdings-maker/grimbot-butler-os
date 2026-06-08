from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .memory import BrainMemory
from .schemas import (
    StateDecayRequest,
    StateEventRequest,
    StateEventResponse,
    StateSignal,
    StateSignalName,
    StateSnapshot,
)


@dataclass(frozen=True)
class SignalDefinition:
    baseline: float
    min_value: float = 0.0
    max_value: float = 1.0
    decay_rate: float = 0.08


SIGNALS: dict[StateSignalName, SignalDefinition] = {
    "attention": SignalDefinition(0.35, decay_rate=0.10),
    "urgency": SignalDefinition(0.20, decay_rate=0.14),
    "novelty": SignalDefinition(0.20, decay_rate=0.12),
    "confidence": SignalDefinition(0.55, decay_rate=0.05),
    "reward": SignalDefinition(0.25, decay_rate=0.10),
    "friction": SignalDefinition(0.15, decay_rate=0.12),
    "fatigue": SignalDefinition(0.20, decay_rate=0.08),
    "curiosity": SignalDefinition(0.30, decay_rate=0.07),
}


class AdaptiveState:
    def __init__(self, memory: BrainMemory) -> None:
        self.memory = memory
        self.ensure_initialized()

    def snapshot(self) -> StateSnapshot:
        signals = self._all_signals()
        values = {signal.name: signal.current_value for signal in signals}
        return StateSnapshot(
            signals=signals,
            values=values,
            next_best_action=self._next_action(values),
        )

    def apply_event(self, request: StateEventRequest) -> StateEventResponse:
        deltas = self._event_deltas(request)
        updated: list[StateSignal] = []
        for name, delta in deltas.items():
            updated.append(self._adjust_signal(name, delta, "event", request.reason))

        return StateEventResponse(
            event_type=request.event_type,
            updated_signals=updated,
            snapshot=self.snapshot(),
        )

    def decay(self, request: StateDecayRequest) -> StateSnapshot:
        now = _now()
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM adaptive_state_signals").fetchall()
            for row in rows:
                definition = SIGNALS[row["name"]]
                elapsed = request.elapsed_seconds
                if elapsed is None:
                    elapsed = max(0.0, (now - _parse_time(row["last_updated"])).total_seconds())
                periods = elapsed / 3600.0
                old_value = _clamp(row["current_value"], definition.min_value, definition.max_value)
                pull = 1 - math.exp(-definition.decay_rate * periods)
                new_value = old_value + (definition.baseline - old_value) * pull
                connection.execute(
                    """
                    UPDATE adaptive_state_signals
                    SET current_value = ?,
                        last_updated = ?,
                        source = ?,
                        reason = ?
                    WHERE name = ?
                    """,
                    (
                        _clamp(new_value, definition.min_value, definition.max_value),
                        _format_time(now),
                        "decay",
                        request.reason,
                        row["name"],
                    ),
                )
            connection.commit()
        return self.snapshot()

    def reset(self, reason: str = "manual reset") -> StateSnapshot:
        now = _format_time(_now())
        with self._connect() as connection:
            for name, definition in SIGNALS.items():
                connection.execute(
                    """
                    INSERT INTO adaptive_state_signals
                        (name, current_value, min_value, max_value, baseline, decay_rate, last_updated, source, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        current_value = excluded.current_value,
                        min_value = excluded.min_value,
                        max_value = excluded.max_value,
                        baseline = excluded.baseline,
                        decay_rate = excluded.decay_rate,
                        last_updated = excluded.last_updated,
                        source = excluded.source,
                        reason = excluded.reason
                    """,
                    (
                        name,
                        definition.baseline,
                        definition.min_value,
                        definition.max_value,
                        definition.baseline,
                        definition.decay_rate,
                        now,
                        "reset",
                        reason[:500],
                    ),
                )
            connection.commit()
        return self.snapshot()

    def values(self) -> dict[str, float]:
        return self.snapshot().values

    def rank_skill_names(self, skill_names: list[str]) -> list[str]:
        values = self.values()

        def score(name: str) -> tuple[float, str]:
            normalized = name.strip().lower()
            base = 0.0
            if normalized == "room_cleanup_plan":
                base += values["urgency"] * 2.0 + values["attention"] + values["confidence"]
            elif normalized == "memory_review":
                base += values["novelty"] + values["curiosity"] + values["attention"] * 0.5
            elif normalized == "maya_briefing":
                base += values["urgency"] + values["friction"] + values["attention"]
            elif normalized == "checklist_builder":
                base += values["friction"] + values["fatigue"] + values["confidence"] * 0.4
            elif normalized == "task_breakdown":
                base += values["curiosity"] + values["confidence"] - values["friction"] * 0.5
            return (-_safe_float(base), normalized)

        return sorted(skill_names, key=score)

    def ensure_initialized(self) -> None:
        now = _format_time(_now())
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM adaptive_state_signals").fetchall()
            existing = {row["name"] for row in rows if row["name"] in SIGNALS}
            for row in rows:
                if row["name"] not in SIGNALS:
                    connection.execute("DELETE FROM adaptive_state_signals WHERE name = ?", (row["name"],))
                    continue
                definition = SIGNALS[row["name"]]
                connection.execute(
                    """
                    UPDATE adaptive_state_signals
                    SET current_value = ?,
                        min_value = ?,
                        max_value = ?,
                        baseline = ?,
                        decay_rate = ?,
                        last_updated = COALESCE(last_updated, ?)
                    WHERE name = ?
                    """,
                    (
                        _clamp(row["current_value"], definition.min_value, definition.max_value),
                        definition.min_value,
                        definition.max_value,
                        definition.baseline,
                        definition.decay_rate,
                        now,
                        row["name"],
                    ),
                )
            connection.commit()

        missing = set(SIGNALS) - existing
        if missing:
            with self._connect() as connection:
                for name in missing:
                    definition = SIGNALS[name]
                    connection.execute(
                        """
                        INSERT INTO adaptive_state_signals
                            (name, current_value, min_value, max_value, baseline, decay_rate, last_updated, source, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            name,
                            definition.baseline,
                            definition.min_value,
                            definition.max_value,
                            definition.baseline,
                            definition.decay_rate,
                            now,
                            "initialize",
                            "initialized missing adaptive state signal",
                        ),
                    )
                connection.commit()

    def _event_deltas(self, request: StateEventRequest) -> dict[StateSignalName, float]:
        intensity = request.intensity
        repeated_count = _metadata_int(request.metadata, "count")
        hazard_count = _metadata_int(request.metadata, "hazard_count")
        object_count = _metadata_int(request.metadata, "object_count")
        mess_count = _metadata_int(request.metadata, "mess_count")

        if request.event_type == "hazard_observed":
            repeat_boost = min(0.20, max(0, repeated_count - 1) * 0.05)
            return {
                "urgency": 0.12 + intensity * 0.20 + repeat_boost,
                "attention": 0.10 + intensity * 0.18 + repeat_boost,
                "confidence": 0.02,
            }
        if request.event_type == "cleanup_succeeded":
            return {"reward": 0.18 + intensity * 0.18, "confidence": 0.10 + intensity * 0.12, "friction": -0.08}
        if request.event_type == "recommendation_ignored":
            return {"friction": 0.12 + intensity * 0.16, "confidence": -0.05, "reward": -0.04}
        if request.event_type == "discovery":
            discovery_boost = min(0.20, object_count * 0.03)
            return {"novelty": 0.12 + intensity * 0.16 + discovery_boost, "curiosity": 0.10 + intensity * 0.14}
        if request.event_type == "unsafe_sensor":
            return {"urgency": 0.22 + intensity * 0.20, "attention": 0.12, "fatigue": 0.10 + intensity * 0.10}
        if request.event_type == "low_battery":
            return {"fatigue": 0.16 + intensity * 0.18, "urgency": 0.08 + intensity * 0.10}
        if request.event_type == "room_scan_observation":
            return {
                "urgency": min(0.40, hazard_count * 0.12),
                "attention": min(0.35, (hazard_count + mess_count) * 0.08),
                "novelty": min(0.30, object_count * 0.05),
                "curiosity": min(0.25, object_count * 0.04),
            }
        if request.event_type == "memory_frequency":
            capped_repeat = min(repeated_count, 5)
            capped_hazards = min(hazard_count, 5)
            return {
                "attention": min(0.18, capped_repeat * 0.03),
                "urgency": min(0.18, capped_hazards * 0.035),
                "confidence": min(0.12, capped_repeat * 0.02),
            }
        return {}

    def _adjust_signal(self, name: StateSignalName, delta: float, source: str, reason: str) -> StateSignal:
        definition = SIGNALS[name]
        now = _format_time(_now())
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM adaptive_state_signals WHERE name = ?", (name,)).fetchone()
            current = _clamp(row["current_value"], definition.min_value, definition.max_value) if row else definition.baseline
            next_value = _clamp(current + delta, definition.min_value, definition.max_value)
            connection.execute(
                """
                INSERT INTO adaptive_state_signals
                    (name, current_value, min_value, max_value, baseline, decay_rate, last_updated, source, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    current_value = excluded.current_value,
                    min_value = excluded.min_value,
                    max_value = excluded.max_value,
                    baseline = excluded.baseline,
                    decay_rate = excluded.decay_rate,
                    last_updated = excluded.last_updated,
                    source = excluded.source,
                    reason = excluded.reason
                """,
                (
                    name,
                    next_value,
                    definition.min_value,
                    definition.max_value,
                    definition.baseline,
                    definition.decay_rate,
                    now,
                    source,
                    reason[:500],
                ),
            )
            connection.commit()
            updated = connection.execute("SELECT * FROM adaptive_state_signals WHERE name = ?", (name,)).fetchone()
        return _signal(updated)

    def _all_signals(self) -> list[StateSignal]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM adaptive_state_signals ORDER BY name"
            ).fetchall()
        return [_signal(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.memory.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _next_action(self, values: dict[str, float]) -> str:
        if values["urgency"] >= 0.65:
            return "keep response concise and prioritize safety-relevant cleanup"
        if values["friction"] >= 0.60:
            return "use lower-pressure coaching and ask for one small next step"
        if values["novelty"] >= 0.60 or values["curiosity"] >= 0.65:
            return "suggest an exploratory room scan or memory review"
        if values["confidence"] >= 0.70:
            return "make a firmer recommendation while preserving safety checks"
        return "continue normal observation and suggestion"


def _signal(row: sqlite3.Row) -> StateSignal:
    definition = SIGNALS[row["name"]]
    return StateSignal(
        name=row["name"],
        current_value=round(_clamp(row["current_value"], definition.min_value, definition.max_value), 4),
        min_value=definition.min_value,
        max_value=definition.max_value,
        baseline=definition.baseline,
        decay_rate=definition.decay_rate,
        last_updated=row["last_updated"],
        source=row["source"],
        reason=row["reason"],
    )


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(metadata.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _clamp(value: object, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, _safe_float(value, minimum)))


def _safe_float(value: object, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return number


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
