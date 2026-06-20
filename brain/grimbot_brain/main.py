from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .adaptive_state import AdaptiveState
from .conversation import run_voice_conversation
from .cycle import execute_cycle
from .dreaming.dream_schemas import (
    DreamRunRequest,
    DreamRunResult,
    DreamStatus,
    PromotionQueueItem,
    PromotionReviewRequest,
    SemanticFact,
)
from .dreaming.dreaming_engine import DreamCycleConflictError, DreamingEngine
from .identity.context_schemas import (
    ContextEntry,
    ContextRememberRequest,
    ContextSearchRequest,
    ContextSearchResult,
    ContextSummary,
    PriorityUpdateRequest,
    ProjectContext,
)
from .identity.context_store import ContextStore
from .maya_core import build_maya_briefing
from .memory import BrainMemory
from .procedural_memory.procedure_matcher import ProcedureMatcher
from .procedural_memory.procedure_schemas import (
    PendingProcedure,
    PendingProcedureReview,
    Procedure,
    ProcedureMatchRequest,
    ProcedureMatchResult,
)
from .procedural_memory.procedure_store import ProcedureStore
from .robot_memory import RobotMemory
from .room_scan import run_room_scan
from .web_search import SearchUsage, search_usage
from .skills import create_default_registry
from .response_composer import compose_maya_response
from .workspace.workspace_inspector import WorkspaceInspector
from .workspace.workspace_schemas import (
    WorkspaceDocument,
    WorkspaceOverview,
    WorkspaceSearchRequest,
    WorkspaceSearchResult,
)
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

CONSOLE_DIR = Path(__file__).resolve().parent / "console"

app = FastAPI(title="GrimBot Butler OS Brain", version="0.12.0")
app.mount("/console/assets", StaticFiles(directory=CONSOLE_DIR), name="console-assets")
memory = BrainMemory()
workspace = WorkspaceInspector()


@app.get("/console", response_class=FileResponse, include_in_schema=False)
def console_page() -> FileResponse:
    return FileResponse(
        CONSOLE_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search/usage", response_model=SearchUsage)
def get_search_usage() -> SearchUsage:
    return search_usage()


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


@app.get("/context", response_model=ContextSummary)
def context_get() -> ContextSummary:
    return ContextStore(memory).summary()


@app.get("/context/projects", response_model=list[ProjectContext])
def context_projects() -> list[ProjectContext]:
    return ContextStore(memory).projects()


@app.get("/context/priorities", response_model=list[ContextEntry])
def context_priorities() -> list[ContextEntry]:
    return ContextStore(memory).priorities()


@app.get("/context/relationships", response_model=list[ContextEntry])
def context_relationships() -> list[ContextEntry]:
    return ContextStore(memory).relationships()


@app.post("/context/search", response_model=ContextSearchResult)
def context_search(request: ContextSearchRequest) -> ContextSearchResult:
    return ContextStore(memory).search(request)


@app.post("/context/remember", response_model=ContextEntry)
def context_remember(request: ContextRememberRequest) -> ContextEntry:
    return ContextStore(memory).remember(request)


@app.post("/context/update-priority", response_model=ProjectContext)
def context_update_priority(request: PriorityUpdateRequest) -> ProjectContext:
    try:
        return ContextStore(memory).update_priority(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc


@app.post("/maya/compose", response_model=MayaComposedResponse)
def maya_compose(request: MayaComposeRequest) -> MayaComposedResponse:
    if request.adaptive_state is None:
        request = request.model_copy(update={"adaptive_state": AdaptiveState(memory).values()})
    return compose_maya_response(request)


@app.post("/maya/briefing", response_model=MayaBriefing)
def maya_briefing(request: MayaBriefingRequest) -> MayaBriefing:
    return build_maya_briefing(request, RobotMemory(memory), ContextStore(memory))


@app.post("/voice/conversation", response_model=VoiceConversationResponse)
def voice_conversation(request: VoiceConversationRequest) -> VoiceConversationResponse:
    try:
        return run_voice_conversation(request, memory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/workspace", response_model=WorkspaceOverview)
def workspace_get() -> WorkspaceOverview:
    return workspace.overview()


@app.get("/workspace/docs", response_model=list[WorkspaceDocument])
def workspace_docs() -> list[WorkspaceDocument]:
    return workspace.documents()


@app.post("/workspace/search", response_model=WorkspaceSearchResult)
def workspace_search(request: WorkspaceSearchRequest) -> WorkspaceSearchResult:
    return workspace.search(request)


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


@app.post("/dream/run", response_model=DreamRunResult)
def dream_run(request: DreamRunRequest) -> DreamRunResult:
    try:
        return DreamingEngine(memory).run(request)
    except DreamCycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/dream/status", response_model=DreamStatus)
def dream_status() -> DreamStatus:
    return DreamingEngine(memory).status()


@app.get("/dream/facts", response_model=list[SemanticFact])
def dream_facts() -> list[SemanticFact]:
    return DreamingEngine(memory).facts()


@app.get("/dream/promotions", response_model=list[PromotionQueueItem])
def dream_promotions() -> list[PromotionQueueItem]:
    return DreamingEngine(memory).promotions()


@app.post("/dream/promotions/{promotion_id}/approve", response_model=PromotionQueueItem)
def dream_approve(promotion_id: int, request: PromotionReviewRequest) -> PromotionQueueItem:
    try:
        return DreamingEngine(memory).approve(promotion_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/dream/promotions/{promotion_id}/reject", response_model=PromotionQueueItem)
def dream_reject(promotion_id: int, request: PromotionReviewRequest) -> PromotionQueueItem:
    try:
        return DreamingEngine(memory).reject(promotion_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/procedures", response_model=list[Procedure])
def procedures_list() -> list[Procedure]:
    return ProcedureStore(memory).list_procedures(active_only=True)


@app.get("/procedures/pending", response_model=list[PendingProcedure])
def procedures_pending() -> list[PendingProcedure]:
    return ProcedureStore(memory).list_pending(pending_only=True)


@app.post("/procedures/pending/{pending_id}/approve", response_model=PendingProcedure)
def procedures_pending_approve(
    pending_id: int,
    request: PendingProcedureReview,
) -> PendingProcedure:
    try:
        return ProcedureStore(memory).approve_pending(pending_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/procedures/pending/{pending_id}/reject", response_model=PendingProcedure)
def procedures_pending_reject(
    pending_id: int,
    request: PendingProcedureReview,
) -> PendingProcedure:
    try:
        return ProcedureStore(memory).reject_pending(pending_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/procedures/match", response_model=ProcedureMatchResult)
def procedures_match(request: ProcedureMatchRequest) -> ProcedureMatchResult:
    return ProcedureMatcher(ProcedureStore(memory)).match(request)


@app.get("/procedures/{procedure_id}", response_model=Procedure)
def procedures_get(procedure_id: int) -> Procedure:
    procedure = ProcedureStore(memory).get(procedure_id)
    if not procedure:
        raise HTTPException(status_code=404, detail=f"Unknown procedure: {procedure_id}")
    return procedure
