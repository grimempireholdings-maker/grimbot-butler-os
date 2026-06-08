from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RobotAction = Literal["stop", "move_forward", "turn_left", "turn_right", "reverse", "idle"]
VisionMode = Literal["mock", "gemini"]


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


class RoomScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str | None = Field(default=None, max_length=500)
    mock_camera_frame: str | None = Field(default=None, max_length=2000)
    capture_webcam: bool = False
    camera_index: int = Field(default=0, ge=0, le=10)


class RoomScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_summary: str = Field(min_length=1, max_length=2000)
    visible_objects: list[str] = Field(default_factory=list, max_length=50)
    mess_zones: list[str] = Field(default_factory=list, max_length=20)
    hazards: list[str] = Field(default_factory=list, max_length=20)
    suggested_cleanup_order: list[str] = Field(default_factory=list, max_length=20)
    next_best_action: str = Field(min_length=1, max_length=500)
    mode: VisionMode = "mock"
    image_path: str | None = Field(default=None, max_length=500)

    @field_validator("visible_objects", "mess_zones", "hazards", "suggested_cleanup_order")
    @classmethod
    def bound_list_items(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            item = value.strip()
            if not item:
                continue
            cleaned.append(item[:120])
        return cleaned
