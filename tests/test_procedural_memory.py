import ast
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import grimbot_brain.main as main_module
from grimbot_brain.memory import BrainMemory
from grimbot_brain.procedural_memory.procedure_matcher import ProcedureMatcher
from grimbot_brain.procedural_memory.procedure_schemas import (
    PendingProcedure,
    PendingProcedureCreate,
    PendingProcedureReview,
    ProcedureCreate,
    ProcedureMatchRequest,
    ProcedureMatchResult,
    ProcedureUpdate,
)
from grimbot_brain.procedural_memory.procedure_store import ProcedureStore


def test_v08_database_migrates_all_procedural_tables(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    memory = BrainMemory(db_path)
    with sqlite3.connect(memory.db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        procedure_columns = {row[1] for row in connection.execute("PRAGMA table_info(procedures)")}
        pending_columns = {row[1] for row in connection.execute("PRAGMA table_info(pending_procedures)")}

    assert {"procedures", "procedure_executions", "pending_procedures"}.issubset(tables)
    assert {
        "procedure_confidence",
        "source",
        "status",
        "version",
        "definition_json",
    }.issubset(procedure_columns)
    assert {"status", "proposal_json", "approved_procedure_id"}.issubset(pending_columns)


@pytest.mark.parametrize(
    "change",
    [
        {"name": "   "},
        {"steps": []},
        {
            "steps": [
                {"step_id": "same", "name": "One", "instruction": "First"},
                {"step_id": "same", "name": "Two", "instruction": "Second"},
            ]
        },
        {
            "steps": [
                {"step_id": "Step-One", "name": "One", "instruction": "First"},
                {"step_id": "step one", "name": "Two", "instruction": "Second"},
            ]
        },
        {
            "branches": [
                {
                    "from_step_id": "inspect",
                    "condition": "if clear",
                    "target_step_id": "missing",
                }
            ]
        },
        {
            "branches": [
                {
                    "from_step_id": " ",
                    "condition": " ",
                    "target_step_id": " ",
                }
            ]
        },
        {"procedure_confidence": 1.1},
        {"source": "untrusted_source"},
        {"unexpected": "field"},
        {"trigger_phrases": ["Reset my desk", "reset-my-desk"]},
        {"preconditions": {"required_objects": ["Desk", "desk"], "notes": []}},
        {"preconditions": {"required_permissions": ["suggest", "suggest"]}},
    ],
)
def test_procedure_schema_rejects_malformed_procedures(change) -> None:
    payload = _procedure_payload()
    payload.update(change)

    with pytest.raises(ValidationError):
        ProcedureCreate.model_validate(payload)


def test_procedure_store_creates_and_retrieves_active_procedure(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))

    created = store.create(_procedure())
    retrieved = store.get(created.procedure_id)
    listed = store.list_procedures()

    assert retrieved == created
    assert listed == [created]
    assert created.version == 1
    assert created.status == "active"
    assert created.source == "human_defined"
    assert created.stats.execution_count == 0


def test_unicode_procedure_names_remain_distinct_and_matchable(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    first = store.create(
        ProcedureCreate.model_validate(
            {**_procedure_payload(), "name": "Cafe Reset", "trigger_phrases": ["reset cafe"]}
        )
    )
    second = store.create(
        ProcedureCreate.model_validate(
            {**_procedure_payload(), "name": "Café Reset", "trigger_phrases": ["reset café"]}
        )
    )

    result = ProcedureMatcher(store).match(ProcedureMatchRequest(query="Café Reset"))

    assert first.procedure_id != second.procedure_id
    assert result.procedure_id == second.procedure_id
    assert result.match_type == "exact_name"


def test_update_archives_old_version_and_creates_new_version(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    first = store.create(_procedure())
    update = ProcedureUpdate.model_validate(
        {
            **_procedure_payload(),
            "description": "Updated desk reset",
            "procedure_confidence": 0.95,
        }
    )

    second = store.update(first.procedure_id, update)
    old = store.get(first.procedure_id)
    active = store.list_procedures()
    rollback = store.rollback_lookup("desk reset", 1)

    assert old is None
    assert second.version == 2
    assert second.status == "active"
    assert active == [second]
    assert rollback is not None
    assert rollback.status == "archived"
    assert rollback.archived_at is not None


def test_failed_update_rolls_back_archive_change(tmp_path, monkeypatch) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    procedure = store.create(_procedure())

    def fail_create(connection, request):
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(store, "_create_in_transaction", fail_create)

    with pytest.raises(RuntimeError, match="injected update failure"):
        store.update(procedure.procedure_id, ProcedureUpdate.model_validate(_procedure_payload()))

    unchanged = store.get(procedure.procedure_id)
    assert unchanged is not None
    assert unchanged.status == "active"
    assert unchanged.version == 1


def test_rollback_lookup_rejects_invalid_versions_and_normalizes_name(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    procedure = store.create(_procedure())

    assert store.rollback_lookup(" DESK-reset!! ", 1) == procedure
    assert store.rollback_lookup("Desk Reset", 0) is None
    assert store.rollback_lookup("Desk Reset", -1) is None
    assert store.rollback_lookup("Desk Reset", 99) is None


def test_execution_records_update_stats_without_executing_steps(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    procedure = store.create(_procedure())

    execution = store.record_execution(procedure.procedure_id, status="completed", outcome="historical record")
    refreshed = store.get(procedure.procedure_id)

    assert execution.status == "completed"
    assert execution.procedure_version == 1
    assert refreshed is not None
    assert refreshed.stats.execution_count == 1
    assert refreshed.stats.success_count == 1
    assert not hasattr(store, "execute")
    assert not hasattr(store, "run")
    with pytest.raises(ValueError, match="Invalid execution status"):
        store.record_execution(procedure.procedure_id, status="running")


def test_pending_proposals_require_explicit_approval_or_rejection(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    pending_approval = store.create_pending(PendingProcedureCreate(proposal=_procedure()))
    pending_rejection = store.create_pending(
        PendingProcedureCreate(
            proposal=ProcedureCreate.model_validate(
                {**_procedure_payload(), "name": "Kitchen Reset", "trigger_phrases": ["reset kitchen"]}
            )
        )
    )

    assert store.list_procedures() == []
    approved = store.approve_pending(
        pending_approval.pending_id,
        PendingProcedureReview(note="Human approved"),
    )
    rejected = store.reject_pending(
        pending_rejection.pending_id,
        PendingProcedureReview(note="Needs revision"),
    )

    assert approved.status == "approved"
    assert approved.approved_procedure_id is not None
    assert rejected.status == "rejected"
    assert rejected.approved_procedure_id is None
    assert len(store.list_procedures()) == 1
    with pytest.raises(ValueError, match="already been reviewed"):
        store.approve_pending(pending_rejection.pending_id, PendingProcedureReview())


def test_pending_approval_conflict_rolls_back_review_state(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    store.create(_procedure())
    pending = store.create_pending(PendingProcedureCreate(proposal=_procedure()))

    with pytest.raises(ValueError, match="Active procedure already exists"):
        store.approve_pending(pending.pending_id, PendingProcedureReview(note="Conflicting approval"))

    still_pending = store.list_pending()
    assert len(still_pending) == 1
    assert still_pending[0].pending_id == pending.pending_id
    assert still_pending[0].status == "pending"
    assert len(store.list_procedures()) == 1


def test_pending_and_match_result_models_reject_inconsistent_shapes() -> None:
    with pytest.raises(ValidationError):
        PendingProcedure.model_validate(
            {
                "pending_id": 1,
                "status": "approved",
                "proposal": _procedure_payload(),
                "submitted_at": "2026-06-14",
                "approved_procedure_id": None,
            }
        )
    with pytest.raises(ValidationError):
        ProcedureMatchResult(
            matched=False,
            procedure_id=1,
            name="Desk Reset",
            confidence=0.5,
            match_type="exact_name",
            required_permission="suggest",
        )


def test_match_request_rejects_ambiguous_id_and_query() -> None:
    with pytest.raises(ValidationError, match="not both"):
        ProcedureMatchRequest(procedure_id=1, query="Desk Reset")


def test_matcher_exact_id_and_normalized_name_lookup(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    procedure = store.create(_procedure())
    matcher = ProcedureMatcher(store)

    by_id = matcher.match(ProcedureMatchRequest(procedure_id=procedure.procedure_id))
    by_name = matcher.match(ProcedureMatchRequest(query="  DESK-reset!! "))

    assert by_id.matched is True
    assert by_id.match_type == "procedure_id"
    assert by_id.procedure_id == procedure.procedure_id
    assert by_name.matched is True
    assert by_name.match_type == "exact_name"
    assert by_name.confidence == 1.0


def test_matcher_fuzzy_trigger_lookup_uses_top_active_match(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    procedure = store.create(_procedure())

    result = ProcedureMatcher(store).match(
        ProcedureMatchRequest(query="please reset my desk area", minimum_confidence=0.6)
    )

    assert result.matched is True
    assert result.procedure_id == procedure.procedure_id
    assert result.match_type == "fuzzy_trigger"
    assert result.required_permission == "suggest"
    assert result.confidence >= 0.6


def test_low_confidence_match_returns_structured_no_match(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    store.create(_procedure())

    result = ProcedureMatcher(store).match(
        ProcedureMatchRequest(query="calibrate orbital telescope", minimum_confidence=0.8)
    )

    assert result.model_dump() == {
        "matched": False,
        "procedure_id": None,
        "name": None,
        "confidence": 0.0,
        "match_type": None,
        "required_permission": None,
    }


def test_matcher_cannot_be_forced_to_guess_with_zero_threshold(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    store.create(_procedure())
    matcher = ProcedureMatcher(store)

    unrelated = matcher.match(
        ProcedureMatchRequest(query="calibrate orbital telescope", minimum_confidence=0)
    )
    too_short = matcher.match(ProcedureMatchRequest(query="x", minimum_confidence=0))

    assert unrelated.matched is False
    assert too_short.matched is False


def test_matcher_does_not_overmatch_generic_one_word_trigger(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    store.create(
        ProcedureCreate.model_validate(
            {
                **_procedure_payload(),
                "name": "General Clean",
                "trigger_phrases": ["clean"],
            }
        )
    )

    result = ProcedureMatcher(store).match(
        ProcedureMatchRequest(query="clean install dependencies", minimum_confidence=0)
    )

    assert result.matched is False


def test_archived_procedures_are_not_match_candidates(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    procedure = store.create(_procedure())
    store.archive(procedure.procedure_id)

    result = ProcedureMatcher(store).match(ProcedureMatchRequest(query="reset my desk"))

    assert result.matched is False


def test_archived_procedure_is_hidden_from_default_id_endpoint(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    store = ProcedureStore(memory)
    procedure = store.create(_procedure())
    store.archive(procedure.procedure_id)
    monkeypatch.setattr(main_module, "memory", memory)

    with pytest.raises(HTTPException) as exc_info:
        main_module.procedures_get(procedure.procedure_id)

    assert exc_info.value.status_code == 404
    assert store.rollback_lookup("Desk Reset", 1) is not None


def test_flagged_procedures_are_not_listed_or_matched(tmp_path) -> None:
    store = ProcedureStore(BrainMemory(tmp_path / "memory.sqlite3"))
    procedure = store.create(_procedure())

    flagged = store.flag(procedure.procedure_id)
    result = ProcedureMatcher(store).match(ProcedureMatchRequest(query="reset my desk"))

    assert flagged.status == "flagged"
    assert store.list_procedures() == []
    assert result.matched is False


def test_procedure_endpoints_have_consistent_shapes_and_no_execution_route(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    store = ProcedureStore(memory)
    procedure = store.create(_procedure())
    pending = store.create_pending(
        PendingProcedureCreate(
            proposal=ProcedureCreate.model_validate(
                {**_procedure_payload(), "name": "Office Close", "trigger_phrases": ["close office"]}
            )
        )
    )
    monkeypatch.setattr(main_module, "memory", memory)

    listed = main_module.procedures_list()
    pending_list = main_module.procedures_pending()
    matched = main_module.procedures_match(ProcedureMatchRequest(query="desk reset"))
    fetched = main_module.procedures_get(procedure.procedure_id)
    approved = main_module.procedures_pending_approve(
        pending.pending_id,
        PendingProcedureReview(note="Approved manually"),
    )
    paths = {route.path for route in main_module.app.routes}

    assert set(listed[0].model_dump()) == {
        "name",
        "description",
        "trigger_phrases",
        "required_permission",
        "source",
        "procedure_confidence",
        "preconditions",
        "steps",
        "branches",
        "procedure_id",
        "version",
        "status",
        "created_at",
        "archived_at",
        "stats",
    }
    assert pending_list[0].status == "pending"
    assert matched.matched is True
    assert fetched == procedure
    assert approved.status == "approved"
    assert "/procedures/pending" in paths
    assert "/procedures/match" in paths
    assert not any(
        path.startswith("/procedures") and any(word in path for word in ("/run", "/execute"))
        for path in paths
    )


def test_procedural_modules_do_not_import_execution_or_external_systems() -> None:
    package_dir = Path(main_module.__file__).parent / "procedural_memory"
    prohibited = {
        "adaptive_state",
        "cycle",
        "hardware",
        "planner",
        "safety",
        "skills",
        "voice",
    }
    imported_modules: set[str] = set()

    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[-1])

    assert imported_modules.isdisjoint(prohibited)


def _procedure() -> ProcedureCreate:
    return ProcedureCreate.model_validate(_procedure_payload())


def _procedure_payload() -> dict:
    return {
        "name": "Desk Reset",
        "description": "Prepare and organize the desk in a reviewable sequence.",
        "trigger_phrases": ["reset my desk", "tidy desk area"],
        "required_permission": "suggest",
        "source": "human_defined",
        "procedure_confidence": 0.9,
        "preconditions": {
            "required_room": "Office",
            "required_objects": ["desk"],
            "required_permissions": ["suggest"],
            "notes": ["Confirm current hazards first"],
        },
        "steps": [
            {
                "step_id": "inspect",
                "name": "Inspect",
                "instruction": "Review the desk and nearby floor for hazards.",
                "required_permission": "observe",
            },
            {
                "step_id": "plan",
                "name": "Plan",
                "instruction": "Describe the suggested cleanup order.",
                "required_permission": "suggest",
            },
        ],
        "branches": [
            {
                "from_step_id": "inspect",
                "condition": "hazard is present",
                "target_step_id": "plan",
            }
        ],
    }
