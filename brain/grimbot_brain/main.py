from __future__ import annotations

from fastapi import FastAPI
from dotenv import load_dotenv

from .memory import BrainMemory
from .perception import perceive
from .planner import plan
from .safety import validate_action
from .schemas import BrainCycleInput, RobotCommand

load_dotenv()

app = FastAPI(title="GrimBot Brain v0", version="0.1.0")
memory = BrainMemory()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cycle", response_model=RobotCommand)
def run_cycle(cycle_input: BrainCycleInput) -> RobotCommand:
    perception = perceive(cycle_input)
    intent = plan(cycle_input, perception)
    command = validate_action(cycle_input, intent)
    memory.log_cycle(cycle_input, perception, intent, command)
    return command


@app.get("/cycles")
def recent_cycles(limit: int = 10) -> list[dict]:
    return memory.recent_cycles(limit=limit)
