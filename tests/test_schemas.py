import pytest
from pydantic import ValidationError

from grimbot_brain.schemas import BrainCycleInput, RobotCommand


def test_cycle_input_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        BrainCycleInput(
            battery_percentage=80,
            distance_cm=100,
            user_command="move forward",
            motor_override=True,
        )


def test_robot_command_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RobotCommand(
            action="stop",
            speed=0,
            reason="Obstacle too close",
            raw_motor_pwm=255,
        )


def test_user_command_length_is_bounded() -> None:
    with pytest.raises(ValidationError):
        BrainCycleInput(
            battery_percentage=80,
            distance_cm=100,
            user_command="x" * 501,
        )
