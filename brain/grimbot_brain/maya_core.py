from __future__ import annotations

from .persona import directives_for_mode, resolve_permission
from .robot_memory import RobotMemory
from .schemas import MayaBriefing, MayaBriefingRequest, RelevantMemoryRequest


def build_maya_briefing(request: MayaBriefingRequest, memory: RobotMemory) -> MayaBriefing:
    relevant = memory.relevant(
        RelevantMemoryRequest(
            query="maya briefing",
            room_name=request.room_name,
            zone_name=request.zone_name,
            limit=10,
        )
    )
    permission = resolve_permission(request.mode, "suggest", request.verified)
    hazards = [hazard.name for hazard in relevant.hazards[:5]]
    mess_zones = [mess.name for mess in relevant.mess_zones[:5]]
    cleanup_tasks = [task.name for task in relevant.cleanup_tasks[:5]]
    facts = [str(fact.get("content", "")) for fact in relevant.semantic_facts[:5] if fact.get("content")]

    priority_items = hazards + cleanup_tasks
    if not priority_items:
        priority_items = [relevant.next_best_action]

    wins = []
    if facts:
        wins.append("Useful room context is available.")
    if cleanup_tasks:
        wins.append("Cleanup order is already structured.")

    fyi = mess_zones + facts
    if not fyi:
        fyi = ["No recurring room context yet."]

    return MayaBriefing(
        mode=request.mode,
        permission=permission,
        verified=request.verified,
        priority_items=priority_items[:10],
        fyi=fyi[:10],
        wins=wins[:10],
        hazards=hazards[:10],
        next_best_action=relevant.next_best_action,
    )
