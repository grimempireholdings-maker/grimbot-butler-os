from __future__ import annotations

from .identity.context_store import ContextStore
from .persona import directives_for_mode, resolve_permission
from .robot_memory import RobotMemory
from .schemas import MayaBriefing, MayaBriefingRequest, RelevantMemoryRequest


def build_maya_briefing(
    request: MayaBriefingRequest,
    memory: RobotMemory,
    context: ContextStore | None = None,
) -> MayaBriefing:
    context = context or ContextStore(memory.memory)
    context_summary = context.summary()
    relevant = memory.relevant(
        RelevantMemoryRequest(
            query="maya briefing",
            room_name=request.room_name,
            zone_name=request.zone_name,
            limit=10,
        )
    )
    hazards = [hazard.name for hazard in relevant.hazards[:5]]
    mess_zones = [mess.name for mess in relevant.mess_zones[:5]]
    cleanup_tasks = [task.name for task in relevant.cleanup_tasks[:5]]
    facts = [str(fact.get("content", "")) for fact in relevant.semantic_facts[:5] if fact.get("content")]

    priority_items = [item.content for item in context_summary.priorities[:5]]
    active_projects = [
        f"{project.name} ({project.status})"
        for project in context_summary.projects[:5]
    ]
    current_bottlenecks = [
        f"{project.name}: {project.current_bottleneck}"
        for project in context_summary.projects[:5]
    ]
    next_actions = [
        f"{project.name}: {project.next_action}"
        for project in context_summary.projects[:5]
    ]
    context_records = [
        *context_summary.priorities[:5],
        *context_summary.projects[:5],
    ]
    effective_verified = (
        request.verified
        and bool(context_records)
        and all(record.verified for record in context_records)
    )
    permission = resolve_permission(request.mode, "suggest", effective_verified)
    if request.room_name or request.zone_name:
        priority_items.extend(hazards + cleanup_tasks)

    wins = ["Maya Console is available for live Chief of Staff testing."]
    if context_summary.projects:
        wins.append(f"{len(context_summary.projects)} active or developing projects are structured.")

    fyi = active_projects + mess_zones + facts
    if request.room_name or request.zone_name:
        next_best_action = relevant.next_best_action
    else:
        next_best_action = (
            context_summary.projects[0].next_action
            if context_summary.projects
            else "Clarify today's highest-value outcome."
        )

    return MayaBriefing(
        mode=request.mode,
        permission=permission,
        verified=effective_verified,
        priority_items=priority_items[:10],
        fyi=fyi[:10],
        wins=wins[:10],
        hazards=hazards[:10],
        active_projects=active_projects[:10],
        current_bottlenecks=current_bottlenecks[:10],
        next_actions=next_actions[:10],
        next_best_action=next_best_action,
    )
