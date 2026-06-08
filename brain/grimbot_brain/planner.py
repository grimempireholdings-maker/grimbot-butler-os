from __future__ import annotations

import re

from .schemas import BrainCycleInput, PerceptionResult, RobotIntent

FORWARD_WORDS = {"forward", "ahead", "go", "move"}
LEFT_WORDS = {"left"}
RIGHT_WORDS = {"right"}
REVERSE_WORDS = {"back", "reverse"}
STOP_WORDS = {"stop", "halt", "freeze"}


def plan(cycle_input: BrainCycleInput, perception: PerceptionResult) -> RobotIntent:
    """Convert command and perception into a high-level intent."""
    command_words = set(re.findall(r"[a-z]+", cycle_input.user_command.lower()))

    if perception.obstacle_detected and command_words & FORWARD_WORDS:
        return RobotIntent(
            requested_action="stop",
            requested_speed=0,
            reason="Planner sees obstacle risk ahead",
        )

    if command_words & STOP_WORDS:
        return RobotIntent(requested_action="stop", requested_speed=0, reason="User requested stop")

    if command_words & LEFT_WORDS:
        return RobotIntent(requested_action="turn_left", requested_speed=0.2, reason="User requested left turn")

    if command_words & RIGHT_WORDS:
        return RobotIntent(requested_action="turn_right", requested_speed=0.2, reason="User requested right turn")

    if command_words & REVERSE_WORDS:
        return RobotIntent(requested_action="reverse", requested_speed=0.15, reason="User requested reverse")

    if command_words & FORWARD_WORDS:
        return RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="Path appears clear")

    return RobotIntent(requested_action="idle", requested_speed=0, reason="No movement command recognized")
