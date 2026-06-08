from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


RobotAction = Literal["stop", "move_forward", "turn_left", "turn_right", "reverse", "idle"]


class IMUReading(BaseModel):
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.8
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0


class BrainCycleInput(BaseModel):
    image_path: str | None = None
    mock_camera_frame: str | None = None
    imu: IMUReading = Field(default_factory=IMUReading)
    battery_percentage: float = Field(ge=0, le=100)
    distance_cm: float = Field(ge=0)
    user_command: str = ""

    @field_validator("user_command")
    @classmethod
    def normalize_command(cls, value: str) -> str:
        return value.strip()


class PerceptionResult(BaseModel):
    mode: Literal["mock", "gemini"]
    scene_summary: str
    obstacle_detected: bool = False
    obstacle_distance_cm: float | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class RobotIntent(BaseModel):
    requested_action: RobotAction
    requested_speed: float = Field(ge=0, le=1)
    reason: str


class RobotCommand(BaseModel):
    action: RobotAction
    speed: float = Field(ge=0, le=1)
    reason: str


class LoggedCycle(BaseModel):
    id: int
    created_at: datetime
    cycle_input: BrainCycleInput
    perception: PerceptionResult
    intent: RobotIntent
    command: RobotCommand
