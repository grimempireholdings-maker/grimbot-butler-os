from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from .adaptive_state import AdaptiveState
from .maya_core import build_maya_briefing
from .memory import BrainMemory
from .persona import resolve_permission
from .response_composer import compose_maya_response
from .robot_memory import RobotMemory
from .schemas import (
    MayaBriefingRequest,
    MayaComposeRequest,
    PermissionLevel,
    RelevantMemoryRequest,
    SkillCategory,
    SkillInfo,
    SkillMachineOutput,
    SkillRunRequest,
    SkillRunResult,
)

PERMISSION_ORDER: dict[PermissionLevel, int] = {
    "observe": 0,
    "suggest": 1,
    "ask_approval": 2,
    "execute": 3,
}


class Skill(ABC):
    name: str
    description: str
    category: SkillCategory
    required_permission: PermissionLevel
    inputs_schema: dict[str, Any]
    outputs_schema: dict[str, Any]

    def info(self) -> SkillInfo:
        return SkillInfo(
            name=self.name,
            description=self.description,
            category=self.category,
            required_permission=self.required_permission,
            inputs_schema=self.inputs_schema,
            outputs_schema=self.outputs_schema,
        )

    def can_execute(self, permission: PermissionLevel) -> bool:
        return PERMISSION_ORDER[permission] >= PERMISSION_ORDER[self.required_permission]

    @abstractmethod
    def execute(self, inputs: dict[str, Any], memory: BrainMemory) -> dict[str, Any]:
        raise NotImplementedError


class SkillRegistry:
    def __init__(self, memory: BrainMemory, adaptive_state: AdaptiveState | None = None) -> None:
        self.memory = memory
        self.adaptive_state = adaptive_state
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[_normalize_skill_name(skill.name)] = skill

    def list_skills(self) -> list[SkillInfo]:
        skill_names = [skill.name for skill in self._skills.values()]
        if self.adaptive_state:
            skill_names = self.adaptive_state.rank_skill_names(skill_names)
        else:
            skill_names = sorted(skill_names)
        return [self._skills[_normalize_skill_name(name)].info() for name in skill_names]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(_normalize_skill_name(name))

    def find_by_category(self, category: str) -> list[SkillInfo]:
        normalized = category.strip().lower()
        return [skill for skill in self.list_skills() if skill.category == normalized]

    def run(self, name: str, request: SkillRunRequest) -> SkillRunResult:
        skill = self.get(name)
        if not skill:
            raise KeyError(f"Unknown skill: {name}")

        effective_permission = resolve_permission(request.assistant_mode, request.permission, request.verified)
        allowed = skill.can_execute(effective_permission)
        state_values = self.adaptive_state.values() if self.adaptive_state and request.include_state else None
        if allowed:
            inputs = dict(request.inputs)
            if state_values is not None:
                inputs.setdefault("adaptive_state", state_values)
            machine_output = _structured_output(skill.name, skill.execute(inputs, self.memory))
            if state_values is not None:
                machine_output["data"]["adaptive_state"] = state_values
                machine_output["data"]["state_next_best_action"] = self.adaptive_state.snapshot().next_best_action
                if skill.name == "room_cleanup_plan" and _state_value(state_values, "urgency") >= 0.65:
                    machine_output["next_best_action"] = f"{machine_output['next_best_action']} with concise safety priority"
        else:
            machine_output = SkillMachineOutput(
                skill=skill.name,
                status="blocked",
                next_best_action="ask for approval before running this skill",
                data={
                    "reason": f"Permission '{effective_permission}' is below required '{skill.required_permission}'",
                    "required_permission": skill.required_permission,
                },
            ).model_dump()

        maya_response = compose_maya_response(
            MayaComposeRequest(
                raw_output=machine_output,
                mode=request.assistant_mode,
                verified=request.verified,
                requested_permission=effective_permission,
                user_goal=f"run skill {skill.name}",
                adaptive_state=state_values,
            )
        )
        return SkillRunResult(
            skill=skill.info(),
            allowed=allowed,
            permission=effective_permission,
            machine_output=machine_output,
            maya_response=maya_response,
        )


class RoomCleanupPlanSkill(Skill):
    name = "room_cleanup_plan"
    description = "Build a memory-backed cleanup plan for a room or zone."
    category: SkillCategory = "planning"
    required_permission: PermissionLevel = "suggest"
    inputs_schema = {"room_name": "optional string", "zone_name": "optional string"}
    outputs_schema = {"plan": "list", "next_best_action": "string", "hazards": "list", "mess_zones": "list"}

    def execute(self, inputs: dict[str, Any], memory: BrainMemory) -> dict[str, Any]:
        robot_memory = RobotMemory(memory)
        room_name = _optional_text(inputs.get("room_name"))
        zone_name = _optional_text(inputs.get("zone_name"))
        state_values = _optional_state(inputs.get("adaptive_state"))
        relevant = robot_memory.relevant(
            RelevantMemoryRequest(
                query="room cleanup plan",
                room_name=room_name,
                zone_name=zone_name,
                limit=10,
                adaptive_state=state_values,
            )
        )
        plan = [f"Clear hazard: {item.name}" for item in relevant.hazards]
        plan.extend(f"Clean mess zone: {item.name}" for item in relevant.mess_zones)
        plan.extend(f"Task: {item.name}" for item in relevant.cleanup_tasks)
        if not plan:
            plan = ["Scan the room before planning cleanup."]

        return {
            "room_summary": (
                "Boss, I can run the room cleanup planning skill. "
                f"Based on memory, {relevant.next_best_action}. Permission level: suggest."
            ),
            "room_name": room_name,
            "zone_name": zone_name,
            "hazards": [item.model_dump() for item in relevant.hazards],
            "mess_zones": [item.model_dump() for item in relevant.mess_zones],
            "plan": plan[:10],
            "next_best_action": relevant.next_best_action,
            "permission_level": self.required_permission,
        }


class ChecklistBuilderSkill(Skill):
    name = "checklist_builder"
    description = "Turn a goal into a concise checklist."
    category: SkillCategory = "productivity"
    required_permission: PermissionLevel = "suggest"
    inputs_schema = {"goal": "string", "items": "optional list"}
    outputs_schema = {"checklist": "list", "next_best_action": "string"}

    def execute(self, inputs: dict[str, Any], memory: BrainMemory) -> dict[str, Any]:
        goal = _text(inputs.get("goal"), "Complete the task")
        items = inputs.get("items")
        checklist = [str(item).strip() for item in items if str(item).strip()] if isinstance(items, list) else []
        if not checklist:
            checklist = [f"Clarify outcome: {goal}", "Identify blockers", "Complete highest-value step", "Review result"]
        return {"goal": goal, "checklist": checklist[:20], "next_best_action": checklist[0]}


class MemoryReviewSkill(Skill):
    name = "memory_review"
    description = "Summarize relevant robot memory."
    category: SkillCategory = "memory"
    required_permission: PermissionLevel = "observe"
    inputs_schema = {"room_name": "optional string", "zone_name": "optional string", "query": "optional string"}
    outputs_schema = {"summary": "string", "hazards": "list", "mess_zones": "list", "next_best_action": "string"}

    def execute(self, inputs: dict[str, Any], memory: BrainMemory) -> dict[str, Any]:
        room_name = _optional_text(inputs.get("room_name"))
        zone_name = _optional_text(inputs.get("zone_name"))
        query = _text(inputs.get("query"), "memory review")
        state_values = _optional_state(inputs.get("adaptive_state"))
        relevant = RobotMemory(memory).relevant(
            RelevantMemoryRequest(
                query=query,
                room_name=room_name,
                zone_name=zone_name,
                limit=10,
                adaptive_state=state_values,
            )
        )
        return {
            "summary": f"Found {len(relevant.hazards)} hazards and {len(relevant.mess_zones)} mess zones.",
            "hazards": [item.model_dump() for item in relevant.hazards],
            "mess_zones": [item.model_dump() for item in relevant.mess_zones],
            "semantic_facts": relevant.semantic_facts,
        }


class MayaBriefingSkill(Skill):
    name = "maya_briefing"
    description = "Generate a structured Maya briefing from robot memory."
    category: SkillCategory = "briefing"
    required_permission: PermissionLevel = "suggest"
    inputs_schema = {"room_name": "optional string", "zone_name": "optional string", "verified": "optional bool"}
    outputs_schema = {"priority_items": "list", "fyi": "list", "wins": "list", "hazards": "list", "next_best_action": "string"}

    def execute(self, inputs: dict[str, Any], memory: BrainMemory) -> dict[str, Any]:
        briefing = build_maya_briefing(
            MayaBriefingRequest(
                room_name=_optional_text(inputs.get("room_name")),
                zone_name=_optional_text(inputs.get("zone_name")),
                verified=bool(inputs.get("verified", False)),
            ),
            RobotMemory(memory),
        )
        payload = briefing.model_dump()
        return payload


class TaskBreakdownSkill(Skill):
    name = "task_breakdown"
    description = "Break a task into small safe steps."
    category: SkillCategory = "planning"
    required_permission: PermissionLevel = "ask_approval"
    inputs_schema = {"task": "string"}
    outputs_schema = {"steps": "list", "next_best_action": "string"}

    def execute(self, inputs: dict[str, Any], memory: BrainMemory) -> dict[str, Any]:
        task = _text(inputs.get("task"), "Complete task")
        steps = [
            f"Define done for: {task}",
            "Check constraints and safety requirements",
            "Do the smallest useful first step",
            "Review before continuing",
        ]
        return {"task": task, "steps": steps, "next_best_action": steps[0]}


def create_default_registry(memory: BrainMemory, adaptive_state: AdaptiveState | None = None) -> SkillRegistry:
    registry = SkillRegistry(memory, adaptive_state)
    registry.register(RoomCleanupPlanSkill())
    registry.register(ChecklistBuilderSkill())
    registry.register(MemoryReviewSkill())
    registry.register(MayaBriefingSkill())
    registry.register(TaskBreakdownSkill())
    return registry


def _text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return (text or fallback)[:500]


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text[:120] or None


def _normalize_skill_name(name: str) -> str:
    return name.strip().lower()


def _structured_output(skill_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    next_best_action = _text(data.pop("next_best_action", None), "review skill output")
    return SkillMachineOutput(
        skill=skill_name,
        status="ok",
        next_best_action=next_best_action,
        data=data,
    ).model_dump()


def _optional_state(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    state: dict[str, float] = {}
    for key, item in value.items():
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        state[str(key)] = max(0.0, min(1.0, number))
    return state or None


def _state_value(values: dict[str, float], key: str) -> float:
    try:
        value = float(values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
