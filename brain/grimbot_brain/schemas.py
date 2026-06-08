from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RobotAction = Literal["stop", "move_forward", "turn_left", "turn_right", "reverse", "idle"]
VisionMode = Literal["mock", "gemini"]
MemoryKind = Literal["observation", "preference", "instruction", "fact"]
AssistantMode = Literal["maya_chief_of_staff", "neutral_robot", "quiet_observer"]
PermissionLevel = Literal["observe", "suggest", "ask_approval", "execute"]
MayaResponseMode = Literal["default", "cleanup_coaching"]
VoiceMode = Literal["mock", "local"]


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

    room_name: str | None = Field(default=None, max_length=120)
    zone_name: str | None = Field(default=None, max_length=120)
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


class RememberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    room_name: str | None = Field(default=None, max_length=120)
    zone_name: str | None = Field(default=None, max_length=120)
    importance: float = Field(default=0.5, ge=0, le=1)
    kind: MemoryKind = "observation"


class RelevantMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    room_name: str | None = Field(default=None, max_length=120)
    zone_name: str | None = Field(default=None, max_length=120)
    limit: int = Field(default=10, ge=1, le=50)


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=120)
    room_name: str | None = Field(default=None, max_length=120)
    zone_name: str | None = Field(default=None, max_length=120)
    count: int = Field(default=1, ge=1)
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)
    first_seen: str
    last_seen: str


class RoomMemorySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: str = Field(min_length=1, max_length=120)
    zones: list[MemoryRecord] = Field(default_factory=list)
    known_objects: list[MemoryRecord] = Field(default_factory=list)
    hazards: list[MemoryRecord] = Field(default_factory=list)
    mess_zones: list[MemoryRecord] = Field(default_factory=list)
    cleanup_tasks: list[MemoryRecord] = Field(default_factory=list)
    episodic_memories: list[dict] = Field(default_factory=list)
    semantic_facts: list[dict] = Field(default_factory=list)
    recommended_first_cleanup_action: str = Field(min_length=1, max_length=500)


class RelevantMemoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    room_name: str | None = Field(default=None, max_length=120)
    hazards: list[MemoryRecord] = Field(default_factory=list)
    mess_zones: list[MemoryRecord] = Field(default_factory=list)
    cleanup_tasks: list[MemoryRecord] = Field(default_factory=list)
    semantic_facts: list[dict] = Field(default_factory=list)
    next_best_action: str = Field(min_length=1, max_length=500)


class RememberResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodic_memory_id: int
    semantic_fact: dict


class MayaComposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_output: dict
    mode: AssistantMode = "maya_chief_of_staff"
    response_mode: MayaResponseMode = "default"
    verified: bool = False
    requested_permission: PermissionLevel = "suggest"
    user_goal: str | None = Field(default=None, max_length=500)


class MayaComposedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: AssistantMode
    permission: PermissionLevel
    verified: bool
    directives_applied: list[str] = Field(max_length=10)
    machine_output: dict
    user_response: str = Field(min_length=1, max_length=2000)


class MayaBriefingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: str | None = Field(default=None, max_length=120)
    zone_name: str | None = Field(default=None, max_length=120)
    verified: bool = False
    mode: AssistantMode = "maya_chief_of_staff"


class MayaBriefing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: AssistantMode
    permission: PermissionLevel
    verified: bool
    priority_items: list[str] = Field(default_factory=list, max_length=10)
    fyi: list[str] = Field(default_factory=list, max_length=10)
    wins: list[str] = Field(default_factory=list, max_length=10)
    hazards: list[str] = Field(default_factory=list, max_length=10)
    next_best_action: str = Field(min_length=1, max_length=500)


class VoiceConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    push_to_talk: bool
    mock_transcript: str | None = Field(default=None, max_length=1000)
    audio_path: str | None = Field(default=None, max_length=500)
    room_name: str | None = Field(default=None, max_length=120)
    zone_name: str | None = Field(default=None, max_length=120)
    assistant_mode: AssistantMode = "maya_chief_of_staff"
    response_mode: MayaResponseMode = "default"
    verified: bool = False


class SpeechToTextResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=1, max_length=1000)
    mode: VoiceMode
    source: str


class TextToSpeechResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    mode: VoiceMode
    audio_path: str | None = Field(default=None, max_length=500)


class VoiceConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=1, max_length=1000)
    memory_context: RelevantMemoryResult
    maya_response: MayaComposedResponse
    speech_output: TextToSpeechResult
    machine_output: dict
