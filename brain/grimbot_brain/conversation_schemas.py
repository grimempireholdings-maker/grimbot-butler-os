from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ConversationIntent = Literal[
    "casual_chat",
    "chief_of_staff_briefing",
    "project_recall",
    "memory_search",
    "skill_request",
    "procedure_request",
    "dream_review",
    "workspace_awareness",
    "room_or_physical_request",
    "unclear",
]

ConversationMode = Literal[
    "ambient_companion",
    "morning_ramp",
    "evening_winddown",
    "casual_presence",
    "approval_review",
    "gentle_orientation",
    "casual",
    "morning_orientation",
    "work_focus",
    "personal_support",
    "business_strategy",
    "project_context",
    "workspace_awareness",
    "physical_environment",
    "feedback_about_maya",
    "capability_question",
    "unclear",
]


class ConversationClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ConversationMode
    needs_web_search: bool = False
    search_query: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_search_decision(self) -> "ConversationClassification":
        query = self.search_query.strip() if self.search_query else None
        if self.needs_web_search and not query:
            raise ValueError("search_query is required when needs_web_search is true")
        if not self.needs_web_search and query:
            raise ValueError("search_query must be null when needs_web_search is false")
        self.search_query = query
        return self


class ConversationSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1)
    required_permission: str = Field(min_length=1, max_length=40)
    reason: str = Field(min_length=1, max_length=500)


class ConversationalAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: ConversationIntent
    user_response: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0, le=1)
    retrieved_context: list[dict] = Field(default_factory=list, max_length=20)
    suggested_skill: ConversationSuggestion | None = None
    suggested_procedure: ConversationSuggestion | None = None
    machine_output: dict
    verified: bool
