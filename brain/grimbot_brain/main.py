from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, Query

from .cycle import execute_cycle
from .maya_core import build_maya_briefing
from .memory import BrainMemory
from .robot_memory import RobotMemory
from .room_scan import run_room_scan
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
)

load_dotenv()

app = FastAPI(title="GrimBot Butler OS Brain", version="0.4.0")
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
    return RobotMemory(memory).relevant(request)


@app.post("/maya/compose", response_model=MayaComposedResponse)
def maya_compose(request: MayaComposeRequest) -> MayaComposedResponse:
    return compose_maya_response(request)


@app.post("/maya/briefing", response_model=MayaBriefing)
def maya_briefing(request: MayaBriefingRequest) -> MayaBriefing:
    return build_maya_briefing(request, RobotMemory(memory))
