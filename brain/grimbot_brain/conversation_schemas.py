from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ConversationIntent = Literal[
    "casual_chat",
    "chief_of_staff_briefing",
    "project_recall",
    "memory_search",
    "skill_request",
    "procedure_request",
    "dream_review",
    "room_or_physical_request",
    "unclear",
]


class ConversationSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1)
    required_permission: str = Field(min_length=1, max_length=40)
    reason: str = Field(min_length=1, max_length=500)


class ConversationalAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: ConversationIntent
    user_response: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)
    retrieved_context: list[dict] = Field(default_factory=list, max_length=20)
    suggested_skill: ConversationSuggestion | None = None
    suggested_procedure: ConversationSuggestion | None = None
    machine_output: dict
    verified: bool
