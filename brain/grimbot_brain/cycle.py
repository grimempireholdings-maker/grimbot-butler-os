from __future__ import annotations

from .memory import BrainMemory
from .perception import perceive
from .planner import plan
from .safety import validate_action
from .schemas import BrainCycleInput, RobotCommand


def execute_cycle(cycle_input: BrainCycleInput, memory: BrainMemory) -> RobotCommand:
    perception = perceive(cycle_input)
    intent = plan(cycle_input, perception)
    command = validate_action(cycle_input, intent)
    memory.log_cycle(cycle_input, perception, intent, command)
    return command
