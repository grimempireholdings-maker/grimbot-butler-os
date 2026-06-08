from __future__ import annotations

from .schemas import BrainCycleInput, RobotCommand, RobotIntent

MIN_BATTERY_PERCENTAGE = 10.0
MIN_OBSTACLE_DISTANCE_CM = 25.0
MAX_SAFE_SPEED = 0.5
MAX_SAFE_TILT_ACCEL = 6.0
MOVEMENT_ACTIONS = {"move_forward", "turn_left", "turn_right", "reverse"}


def validate_action(cycle_input: BrainCycleInput, intent: RobotIntent) -> RobotCommand:
    """Validate every planner intent before it can become a robot command."""
    if cycle_input.battery_percentage <= MIN_BATTERY_PERCENTAGE:
        return _stop("Battery too low")

    if cycle_input.distance_cm < MIN_OBSTACLE_DISTANCE_CM and intent.requested_action in MOVEMENT_ACTIONS:
        return _stop("Obstacle too close")

    if _is_unstable(cycle_input):
        return _stop("IMU reports unsafe tilt or acceleration")

    if intent.requested_action not in MOVEMENT_ACTIONS and intent.requested_action not in {"stop", "idle"}:
        return _stop("Unknown action rejected")

    if intent.requested_action in {"stop", "idle"}:
        return RobotCommand(action=intent.requested_action, speed=0, reason=intent.reason)

    safe_speed = min(intent.requested_speed, MAX_SAFE_SPEED)
    return RobotCommand(action=intent.requested_action, speed=safe_speed, reason=intent.reason)


def _is_unstable(cycle_input: BrainCycleInput) -> bool:
    imu = cycle_input.imu
    return abs(imu.accel_x) > MAX_SAFE_TILT_ACCEL or abs(imu.accel_y) > MAX_SAFE_TILT_ACCEL


def _stop(reason: str) -> RobotCommand:
    return RobotCommand(action="stop", speed=0, reason=reason)
