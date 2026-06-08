from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from .adaptive_state import AdaptiveState
from .conversation import run_voice_conversation
from .cycle import execute_cycle
from .maya_core import build_maya_briefing
from .memory import BrainMemory
from .robot_memory import RobotMemory
from .room_scan import run_room_scan
from .skills import create_default_registry
from .response_composer import compose_maya_response
from .schemas import (
    BrainCycleInput,
    MayaBriefing,
    MayaBriefingRequest,
    MayaComposeRequest,
    MayaComposedResponse,
    MemoryRecord,
    RelevantMemoryRequest,
    RelevantMemoryResult,
    RememberRequest,
    RememberResponse,
    RobotCommand,
    RoomMemorySummary,
    RoomScanRequest,
    RoomScanResult,
    SkillInfo,
    SkillRunRequest,
    SkillRunResult,
    StateDecayRequest,
    StateEventRequest,
    StateEventResponse,
    StateResetRequest,
    StateSnapshot,
    VoiceConversationRequest,
    VoiceConversationResponse,
)

load_dotenv()

app = FastAPI(title="GrimBot Butler OS Brain", version="0.7.0")
memory = BrainMemory()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cycle", response_model=RobotCommand)
def run_cycle(cycle_input: BrainCycleInput) -> RobotCommand:
    return execute_cycle(cycle_input, memory)


@app.get("/cycles")
def recent_cycles(limit: int = Query(default=10, ge=1, le=100)) -> list[dict]:
    return memory.recent_cycles(limit=limit)


@app.post("/room-scan", response_model=RoomScanResult)
def room_scan(request: RoomScanRequest) -> RoomScanResult:
    return run_room_scan(request, memory)


@app.get("/room-scans")
def recent_room_scans(limit: int = Query(default=10, ge=1, le=100)) -> list[dict]:
    return memory.recent_room_scans(limit=limit)


@app.post("/memory/remember", response_model=RememberResponse)
def remember(request: RememberRequest) -> dict:
    return RobotMemory(memory).remember(request)


@app.get("/memory/rooms", response_model=list[MemoryRecord])
def memory_rooms() -> list[MemoryRecord]:
    return RobotMemory(memory).list_rooms()


@app.get("/memory/rooms/{room_name}", response_model=RoomMemorySummary)
def memory_room(room_name: str) -> RoomMemorySummary:
    return RobotMemory(memory).room_summary(room_name)


@app.get("/memory/hazards", response_model=list[MemoryRecord])
def memory_hazards(
    room_name: str | None = None,
    zone_name: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[MemoryRecord]:
    return RobotMemory(memory).hazards(room_name, zone_name, limit)


@app.get("/memory/mess-zones", response_model=list[MemoryRecord])
def memory_mess_zones(
    room_name: str | None = None,
    zone_name: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[MemoryRecord]:
    return RobotMemory(memory).mess_zones(room_name, zone_name, limit)


@app.post("/memory/relevant", response_model=RelevantMemoryResult)
def memory_relevant(request: RelevantMemoryRequest) -> RelevantMemoryResult:
    if request.adaptive_state is None:
        request = request.model_copy(update={"adaptive_state": AdaptiveState(memory).values()})
    return RobotMemory(memory).relevant(request)


@app.post("/maya/compose", response_model=MayaComposedResponse)
def maya_compose(request: MayaComposeRequest) -> MayaComposedResponse:
    if request.adaptive_state is None:
        request = request.model_copy(update={"adaptive_state": AdaptiveState(memory).values()})
    return compose_maya_response(request)


@app.post("/maya/briefing", response_model=MayaBriefing)
def maya_briefing(request: MayaBriefingRequest) -> MayaBriefing:
    return build_maya_briefing(request, RobotMemory(memory))


@app.post("/voice/conversation", response_model=VoiceConversationResponse)
def voice_conversation(request: VoiceConversationRequest) -> VoiceConversationResponse:
    try:
        return run_voice_conversation(request, memory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/skills", response_model=list[SkillInfo])
def skills_list(category: str | None = None) -> list[SkillInfo]:
    registry = create_default_registry(memory, AdaptiveState(memory))
    if category:
        return registry.find_by_category(category)
    return registry.list_skills()


@app.get("/skills/{skill_name}", response_model=SkillInfo)
def skills_get(skill_name: str) -> SkillInfo:
    skill = create_default_registry(memory, AdaptiveState(memory)).get(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {skill_name}")
    return skill.info()


@app.post("/skills/{skill_name}/run", response_model=SkillRunResult)
def skills_run(skill_name: str, request: SkillRunRequest) -> SkillRunResult:
    try:
        return create_default_registry(memory, AdaptiveState(memory)).run(skill_name, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/state", response_model=StateSnapshot)
def state_get() -> StateSnapshot:
    return AdaptiveState(memory).snapshot()


@app.post("/state/event", response_model=StateEventResponse)
def state_event(request: StateEventRequest) -> StateEventResponse:
    return AdaptiveState(memory).apply_event(request)


@app.post("/state/decay", response_model=StateSnapshot)
def state_decay(request: StateDecayRequest) -> StateSnapshot:
    return AdaptiveState(memory).decay(request)


@app.post("/state/reset", response_model=StateSnapshot)
def state_reset(request: StateResetRequest) -> StateSnapshot:
    return AdaptiveState(memory).reset(request.reason)
