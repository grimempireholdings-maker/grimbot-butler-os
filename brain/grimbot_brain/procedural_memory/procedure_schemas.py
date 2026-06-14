from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schemas import PermissionLevel


ProcedureSource = Literal["human_defined", "observed_pattern", "dream_inferred", "skill_composed"]
ProcedureStatus = Literal["active", "archived", "flagged"]
PendingProcedureStatus = Literal["pending", "approved", "rejected"]
ProcedureMatchType = Literal["procedure_id", "exact_name", "fuzzy_trigger"]
ExecutionStatus = Literal["recorded", "completed", "failed", "cancelled"]


class ProcedureStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)
    instruction: str = Field(min_length=1, max_length=1000)
    required_permission: PermissionLevel = "suggest"

    @field_validator("step_id", "name", "instruction")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("step fields cannot be blank")
        return cleaned


class ProcedureBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_step_id: str = Field(min_length=1, max_length=80)
    condition: str = Field(min_length=1, max_length=500)
    target_step_id: str = Field(min_length=1, max_length=80)

    @field_validator("from_step_id", "condition", "target_step_id")
    @classmethod
    def clean_branch_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("branch fields cannot be blank")
        return cleaned


class ProcedurePreconditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_room: str | None = Field(default=None, max_length=120)
    required_objects: list[str] = Field(default_factory=list, max_length=30)
    required_permissions: list[PermissionLevel] = Field(default_factory=list, max_length=4)
    notes: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("required_room")
    @classmethod
    def clean_optional_room(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("required_objects", "notes")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip()[:200] for value in values if value.strip()]
        normalized = [_normalize_text(value) for value in cleaned]
        if len(normalized) != len(set(normalized)):
            raise ValueError("list values must be unique")
        return cleaned

    @field_validator("required_permissions")
    @classmethod
    def unique_permissions(cls, values: list[PermissionLevel]) -> list[PermissionLevel]:
        if len(values) != len(set(values)):
            raise ValueError("required permissions must be unique")
        return values


class ProcedureStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_count: int = Field(default=0, ge=0, le=1_000_000)
    success_count: int = Field(default=0, ge=0, le=1_000_000)
    failure_count: int = Field(default=0, ge=0, le=1_000_000)
    last_executed_at: str | None = None

    @model_validator(mode="after")
    def validate_totals(self) -> "ProcedureStats":
        if self.success_count + self.failure_count > self.execution_count:
            raise ValueError("success and failure counts cannot exceed execution count")
        return self


class ProcedureDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=1000)
    trigger_phrases: list[str] = Field(min_length=1, max_length=30)
    required_permission: PermissionLevel = "suggest"
    source: ProcedureSource
    procedure_confidence: float = Field(ge=0, le=1)
    preconditions: ProcedurePreconditions = Field(default_factory=ProcedurePreconditions)
    steps: list[ProcedureStep] = Field(min_length=1, max_length=100)
    branches: list[ProcedureBranch] = Field(default_factory=list, max_length=100)

    @field_validator("name", "description")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("procedure name and description cannot be blank")
        return cleaned

    @field_validator("trigger_phrases")
    @classmethod
    def clean_triggers(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip()[:200] for value in values if value.strip()]
        if not cleaned:
            raise ValueError("at least one trigger phrase is required")
        normalized = [_normalize_text(value) for value in cleaned]
        if len(normalized) != len(set(normalized)):
            raise ValueError("trigger phrases must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_step_graph(self) -> "ProcedureDefinition":
        step_ids = [_normalize_text(step.step_id) for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("step IDs must be unique")
        valid_ids = {step.step_id for step in self.steps}
        for branch in self.branches:
            if branch.from_step_id not in valid_ids or branch.target_step_id not in valid_ids:
                raise ValueError("branches must reference valid step IDs")
        return self


class Procedure(ProcedureDefinition):
    procedure_id: int = Field(ge=1)
    version: int = Field(ge=1)
    status: ProcedureStatus
    created_at: str
    archived_at: str | None = None
    stats: ProcedureStats = Field(default_factory=ProcedureStats)


class ProcedureCreate(ProcedureDefinition):
    pass


class ProcedureUpdate(ProcedureDefinition):
    pass


class ProcedureExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: int = Field(ge=1)
    procedure_id: int = Field(ge=1)
    procedure_version: int = Field(ge=1)
    status: ExecutionStatus
    started_at: str
    completed_at: str | None = None
    outcome: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_completion_state(self) -> "ProcedureExecution":
        if self.status == "recorded" and self.completed_at is not None:
            raise ValueError("recorded executions cannot have a completion timestamp")
        if self.status != "recorded" and self.completed_at is None:
            raise ValueError("terminal execution records require a completion timestamp")
        return self


class PendingProcedure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pending_id: int = Field(ge=1)
    status: PendingProcedureStatus
    proposal: ProcedureDefinition
    submitted_at: str
    reviewed_at: str | None = None
    review_note: str | None = Field(default=None, max_length=500)
    approved_procedure_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_review_state(self) -> "PendingProcedure":
        if self.status == "approved" and self.approved_procedure_id is None:
            raise ValueError("approved proposals require an approved procedure ID")
        if self.status != "approved" and self.approved_procedure_id is not None:
            raise ValueError("only approved proposals may reference a procedure")
        return self


class PendingProcedureCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: ProcedureDefinition


class PendingProcedureReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str | None = Field(default=None, max_length=500)


class ProcedureMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    procedure_id: int | None = Field(default=None, ge=1)
    query: str | None = Field(default=None, max_length=500)
    minimum_confidence: float = Field(default=0.65, ge=0, le=1)

    @model_validator(mode="after")
    def require_lookup_value(self) -> "ProcedureMatchRequest":
        has_query = bool(self.query and self.query.strip())
        if self.procedure_id is None and not has_query:
            raise ValueError("procedure_id or query is required")
        if self.procedure_id is not None and has_query:
            raise ValueError("provide procedure_id or query, not both")
        return self


class ProcedureMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched: bool
    procedure_id: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, max_length=160)
    confidence: float = Field(ge=0, le=1)
    match_type: ProcedureMatchType | None = None
    required_permission: PermissionLevel | None = None

    @model_validator(mode="after")
    def validate_match_shape(self) -> "ProcedureMatchResult":
        details = (self.procedure_id, self.name, self.match_type, self.required_permission)
        if self.matched and any(value is None for value in details):
            raise ValueError("matched results require procedure details")
        if self.matched and self.confidence <= 0:
            raise ValueError("matched results require positive confidence")
        if not self.matched and any(value is not None for value in details):
            raise ValueError("unmatched results cannot include procedure details")
        if not self.matched and self.confidence != 0:
            raise ValueError("unmatched results must have zero confidence")
        return self


def _normalize_text(value: str) -> str:
    return re.sub(r"_+", " ", re.sub(r"[^\w]+", " ", value.casefold())).strip()
