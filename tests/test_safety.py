from grimbot_brain.safety import validate_action
from grimbot_brain.schemas import BrainCycleInput, IMUReading, RobotIntent


def test_obstacle_too_close_forces_stop() -> None:
    cycle_input = BrainCycleInput(
        battery_percentage=80,
        distance_cm=10,
        user_command="move forward",
    )
    intent = RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="test")

    command = validate_action(cycle_input, intent)

    assert command.action == "stop"
    assert command.speed == 0
    assert command.reason == "Obstacle too close"


def test_low_battery_forces_stop() -> None:
    cycle_input = BrainCycleInput(
        battery_percentage=10,
        distance_cm=100,
        user_command="move forward",
    )
    intent = RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="test")

    command = validate_action(cycle_input, intent)

    assert command.action == "stop"
    assert command.speed == 0
    assert command.reason == "Battery too low"


def test_unstable_imu_forces_stop() -> None:
    cycle_input = BrainCycleInput(
        imu=IMUReading(accel_x=8),
        battery_percentage=80,
        distance_cm=100,
        user_command="move forward",
    )
    intent = RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="test")

    command = validate_action(cycle_input, intent)

    assert command.action == "stop"
    assert command.speed == 0
    assert command.reason == "IMU reports unsafe tilt or acceleration"


def test_unsafe_vertical_acceleration_forces_stop() -> None:
    cycle_input = BrainCycleInput(
        imu=IMUReading(accel_z=2),
        battery_percentage=80,
        distance_cm=100,
        user_command="move forward",
    )
    intent = RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="test")

    command = validate_action(cycle_input, intent)

    assert command.action == "stop"
    assert command.reason == "IMU reports unsafe tilt or acceleration"


def test_fast_rotation_forces_stop() -> None:
    cycle_input = BrainCycleInput(
        imu=IMUReading(gyro_z=240),
        battery_percentage=80,
        distance_cm=100,
        user_command="move forward",
    )
    intent = RobotIntent(requested_action="move_forward", requested_speed=0.25, reason="test")

    command = validate_action(cycle_input, intent)

    assert command.action == "stop"
    assert command.reason == "IMU reports unsafe tilt or acceleration"


def test_speed_is_clamped() -> None:
    cycle_input = BrainCycleInput(
        battery_percentage=80,
        distance_cm=100,
        user_command="move forward",
    )
    intent = RobotIntent(requested_action="move_forward", requested_speed=1, reason="test")

    command = validate_action(cycle_input, intent)

    assert command.action == "move_forward"
    assert command.speed == 0.5
