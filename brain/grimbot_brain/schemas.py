from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RobotAction = Literal["stop", "move_forward", "turn_left", "turn_right", "reverse", "idle"]


class IMUReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.8
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0


class BrainCycleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str | None = Field(default=None, max_length=500)
    mock_camera_frame: str | None = Field(default=None, max_length=2000)
    imu: IMUReading = Field(default_factory=IMUReading)
    battery_percentage: float = Field(ge=0, le=100)
    distance_cm: float = Field(ge=0, le=10000)
    user_command: str = Field(default="", max_length=500)

    @field_validator("user_command")
    @classmethod
    def normalize_command(cls, value: str) -> str:
        return value.strip()


class PerceptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "gemini"]
    scene_summary: str = Field(min_length=1, max_length=2000)
    obstacle_detected: bool = False
    obstacle_distance_cm: float | None = Field(default=None, ge=0, le=10000)
    confidence: float = Field(default=0.5, ge=0, le=1)


class RobotIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_action: RobotAction
    requested_speed: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)


class RobotCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RobotAction
    speed: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
