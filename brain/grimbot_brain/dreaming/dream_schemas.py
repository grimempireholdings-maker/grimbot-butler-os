from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DreamProvider = Literal["rule_based", "mock"]
DreamStatusValue = Literal["running", "completed", "failed"]
FactTier = Literal["semantic", "core"]
PromotionStatus = Literal["pending", "approved", "rejected", "anchor"]


class DreamRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: DreamProvider = "rule_based"
    episode_limit: int = Field(default=500, ge=1, le=2000)
    run_forgetting: bool = True


class SemanticFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: int = Field(ge=1)
    content: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)
    created_at: str
    last_reinforced: str
    tags: list[str] = Field(default_factory=list, max_length=30)
    tier: FactTier


class PromotionQueueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    fact_id: int = Field(ge=1)
    status: PromotionStatus
    created_at: str
    reviewed_at: str | None = None
    review_note: str | None = Field(default=None, max_length=500)
    fact: SemanticFact


class PromotionReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str | None = Field(default=None, max_length=500)
    anchor: bool = False


class DreamCycle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dream_id: int = Field(ge=1)
    started_at: str
    completed_at: str | None = None
    episodes_processed: int = Field(ge=0)
    facts_created: int = Field(ge=0)
    facts_forgotten: int = Field(ge=0)
    contradictions_flagged: int = Field(ge=0)
    status: DreamStatusValue
    error_message: str | None = Field(default=None, max_length=1000)


class DreamRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle: DreamCycle
    candidate_facts: list[SemanticFact] = Field(default_factory=list)
    promotions_created: int = Field(ge=0)


class DreamStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_policy: Literal["manual_only"] = "manual_only"
    active: bool
    latest_cycle: DreamCycle | None = None
