import json

from grimbot_brain.main import run_cycle
from grimbot_brain.schemas import BrainCycleInput
from grimbot_brain.schemas import RobotCommand


def test_robot_command_serializes_to_strict_json_shape() -> None:
    command = RobotCommand(action="stop", speed=0, reason="Obstacle too close")
    payload = json.loads(command.model_dump_json())

    assert payload == {"action": "stop", "speed": 0.0, "reason": "Obstacle too close"}


def test_cycle_returns_command_only() -> None:
    command = run_cycle(
        BrainCycleInput(
            image_path="mock.jpg",
            mock_camera_frame="wall very close",
            battery_percentage=90,
            distance_cm=10,
            user_command="turn left",
        )
    )

    payload = command.model_dump()
    assert set(payload) == {"action", "speed", "reason"}
    assert payload == {"action": "stop", "speed": 0.0, "reason": "Obstacle too close"}
