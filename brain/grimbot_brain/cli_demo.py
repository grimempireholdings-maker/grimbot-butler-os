from __future__ import annotations

import json
import os

from .cycle import execute_cycle
from .memory import BrainMemory
from .schemas import BrainCycleInput, IMUReading


def main() -> None:
    os.environ.setdefault("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory()

    scenarios = [
        ("clear hallway", 120, 82, "move forward slowly"),
        ("chair close ahead", 35, 80, "move forward"),
        ("open room", 90, 76, "turn left"),
        ("open room", 90, 74, "turn right"),
        ("wall very close", 18, 70, "move forward"),
        ("clear floor", 100, 9, "move forward"),
        ("open path", 80, 65, "reverse"),
        ("clear hallway", 150, 60, "stop"),
        ("open path", 140, 55, "dance"),
        ("clear hallway", 130, 50, "go ahead"),
    ]

    for frame, distance_cm, battery, command_text in scenarios:
        cycle_input = BrainCycleInput(
            image_path=None,
            mock_camera_frame=frame,
            imu=IMUReading(),
            battery_percentage=battery,
            distance_cm=distance_cm,
            user_command=command_text,
        )
        command = execute_cycle(cycle_input, memory)
        print(json.dumps(command.model_dump(), separators=(",", ":")))


if __name__ == "__main__":
    main()
