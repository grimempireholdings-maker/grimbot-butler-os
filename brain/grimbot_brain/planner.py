from __future__ import annotations

from .schemas import BrainCycleInput, PerceptionResult, RobotIntent


def plan(cycle_input: BrainCycleInput, perception: PerceptionResult) -> RobotIntent:
    """Convert command and perception into a high-level intent."""
    command = cycle_input.user_command.lower()

    if perception.obstacle_detected and any(word in command for word in ("forward", "ahead", "go")):
        return RobotIntent(
            requested_action="stop",
            requested_speed=0,
            reason="Planner sees obstacle risk ahead",
        )

    if any(word in command for word in ("stop", "halt", "freeze")):
        return RobotIntent(requested_action="stop", requested_speed=0, reason="User requested stop")

    if any(word in command for word in ("left", "turn left")):
        return RobotIntent(requested_action="turn_left", requested_speed=0.2, reason="User requested left turn")

    if any(word in command for word in ("right", "turn right")):
        return RobotIntent(requested_action="turn_right", requested_speed=0.2, reason="User requested right turn")

    if any(word in command for word in ("back", "reverse")):
        return RobotIntent(requested_action="reverse", requested_speed=0.15, reason="User requested reverse")

    if any(word in command for word in ("forward", "ahead", "go", "move")):
        return RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="Path appears clear")

    return RobotIntent(requested_action="idle", requested_speed=0, reason="No movement command recognized")
