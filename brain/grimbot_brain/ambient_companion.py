from __future__ import annotations

import sqlite3
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .dreaming import DreamingEngine
from .identity.context_store import ContextStore
from .memory import BrainMemory
from .procedural_memory.procedure_store import ProcedureStore
from .workspace.workspace_inspector import WorkspaceInspector


class AmbientContext(BaseModel):
    """Read-only daily context; internal architecture stays backstage."""

    model_config = ConfigDict(extra="forbid")

    current_time: str
    time_of_day: str
    things_that_matter: list[str] = Field(default_factory=list, max_length=5)
    active_lanes: list[str] = Field(default_factory=list, max_length=5)
    open_loops: list[str] = Field(default_factory=list, max_length=5)
    things_waiting_for_review: list[str] = Field(default_factory=list, max_length=5)
    what_changed_since_last_time: list[str] = Field(default_factory=list, max_length=5)
    recent_notes: list[str] = Field(default_factory=list, max_length=5)
    tone_guidance: list[str] = Field(default_factory=list, max_length=4)
    calendar_available: bool = False

    def wording_context(self) -> list[dict]:
        rows: list[dict] = [
            {"type": "current_time", "value": self.current_time, "time_of_day": self.time_of_day},
            {"type": "what_matters_today", "values": self.things_that_matter},
            {"type": "active_lanes", "values": self.active_lanes},
            {"type": "open_loops", "values": self.open_loops},
            {"type": "things_needing_your_approval", "values": self.things_waiting_for_review},
            {"type": "what_changed_since_last_time", "values": self.what_changed_since_last_time},
            {"type": "recent_notes", "values": self.recent_notes},
            {"type": "tone_guidance", "values": self.tone_guidance},
        ]
        return [row for row in rows if row.get("value") or row.get("values")]


def build_ambient_context(memory: BrainMemory) -> AmbientContext:
    """Assemble context without executing, approving, or mutating anything."""
    now = datetime.now().astimezone()
    context = _safe_call(lambda: ContextStore(memory).summary())
    projects = context.projects[:5] if context else []
    priorities = [entry.content for entry in context.priorities[:5]] if context else []

    approvals: list[str] = []
    promotions = _safe_call(lambda: DreamingEngine(memory).promotions(), [])
    approvals.extend(item.fact.content for item in promotions if item.status == "pending")
    procedures = _safe_call(lambda: ProcedureStore(memory).list_pending(), [])
    approvals.extend(item.proposal.name for item in procedures)

    overview = _safe_call(lambda: WorkspaceInspector().overview())
    episodes = _safe_call(lambda: memory.recent_episodes(5), [])
    state = _safe_call(lambda: _read_state_values(memory), {})

    return AmbientContext(
        current_time=now.isoformat(timespec="minutes"),
        time_of_day=_time_of_day(now.hour),
        things_that_matter=priorities,
        active_lanes=[project.name for project in projects],
        open_loops=[project.current_bottleneck for project in projects],
        things_waiting_for_review=approvals[:5],
        what_changed_since_last_time=overview.recent_commits[:5] if overview else [],
        recent_notes=[str(item["content"])[:300] for item in episodes],
        tone_guidance=_tone_guidance(state),
    )


def _safe_call(call, default=None):
    try:
        return call()
    except Exception:
        return default


def _read_state_values(memory: BrainMemory) -> dict[str, float]:
    """Read existing tone signals without initializing, decaying, or updating them."""
    with sqlite3.connect(memory.db_path) as connection:
        rows = connection.execute(
            "SELECT name, current_value FROM adaptive_state_signals"
        ).fetchall()
    return {str(name): float(value) for name, value in rows}


def _time_of_day(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 22:
        return "evening"
    return "late night"


def _tone_guidance(state: dict[str, float]) -> list[str]:
    guidance = ["Stay calm, warm, and low-pressure."]
    if state.get("fatigue", 0) >= 0.55 or state.get("friction", 0) >= 0.55:
        guidance.append("Use shorter sentences and offer only one gentle next step.")
    if state.get("urgency", 0) < 0.6:
        guidance.append("Do not manufacture urgency.")
    if state.get("curiosity", 0) >= 0.65:
        guidance.append("Leave room for exploration rather than forcing execution.")
    return guidance[:4]
