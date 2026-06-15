from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ContextType = Literal[
    "person_profile",
    "mission",
    "venture",
    "project",
    "priority",
    "relationship",
    "decision",
    "constraint",
    "protocol",
    "belief",
    "current_bottleneck",
    "next_action",
]
ContextSource = Literal["julian_prime", "maya", "grimbot", "board", "portfolio_seed"]
ContextWriteSource = Literal["julian_prime", "maya", "grimbot", "board"]
ProjectStatus = Literal["active", "building", "experiment", "archived", "paused"]


class ContextEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    context_type: ContextType
    name: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=2000)
    priority: int = Field(default=50, ge=0, le=100)
    source: ContextSource
    verified: bool
    created_at: str
    last_updated: str


class ProjectContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=160)
    status: ProjectStatus
    priority: int = Field(ge=0, le=100)
    current_bottleneck: str = Field(min_length=1, max_length=1000)
    next_action: str = Field(min_length=1, max_length=1000)
    last_updated: str
    related_entities: list[str] = Field(default_factory=list, max_length=30)
    source: ContextSource
    verified: bool


class ContextSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_profile: list[ContextEntry] = Field(default_factory=list)
    missions: list[ContextEntry] = Field(default_factory=list)
    ventures: list[ContextEntry] = Field(default_factory=list)
    projects: list[ProjectContext] = Field(default_factory=list)
    priorities: list[ContextEntry] = Field(default_factory=list)
    relationships: list[ContextEntry] = Field(default_factory=list)
    bottlenecks: list[ContextEntry] = Field(default_factory=list)
    next_actions: list[ContextEntry] = Field(default_factory=list)
    protocols: list[ContextEntry] = Field(default_factory=list)


class ContextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    context_types: list[ContextType] | None = Field(default=None, max_length=12)
    limit: int = Field(default=10, ge=1, le=50)


class ContextSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    entries: list[ContextEntry] = Field(default_factory=list)
    projects: list[ProjectContext] = Field(default_factory=list)
    next_best_action: str = Field(min_length=1, max_length=1000)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=500)


class ContextRememberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_type: ContextType
    name: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=2000)
    priority: int = Field(default=50, ge=0, le=100)
    source: ContextWriteSource = "julian_prime"
    verified: bool = False

    @model_validator(mode="after")
    def verified_requires_julian_prime(self) -> "ContextRememberRequest":
        if self.verified and self.source != "julian_prime":
            raise ValueError("verified context must come from julian_prime")
        return self


class PriorityUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    priority: int = Field(ge=0, le=100)
    status: ProjectStatus | None = None
    current_bottleneck: str | None = Field(default=None, max_length=1000)
    next_action: str | None = Field(default=None, max_length=1000)
    verified: bool = False
