from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, Query

from .cycle import execute_cycle
from .memory import BrainMemory
from .room_scan import run_room_scan
from .schemas import BrainCycleInput, RobotCommand, RoomScanRequest, RoomScanResult

load_dotenv()

app = FastAPI(title="GrimBot Butler OS Brain", version="0.2.0")
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
