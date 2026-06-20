from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkspaceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    relative_path: str = Field(min_length=1, max_length=500)
    kind: Literal["file", "directory"]


class WorkspaceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=160)
    relative_path: str = Field(min_length=1, max_length=500)
    preview: str = Field(max_length=500)


class WorkspaceOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_root: str = Field(min_length=1, max_length=1000)
    repo_name: str = Field(min_length=1, max_length=160)
    branch: str | None = Field(default=None, max_length=160)
    status_summary: list[str] = Field(default_factory=list, max_length=100)
    recent_commits: list[str] = Field(default_factory=list, max_length=5)
    version: str | None = Field(default=None, max_length=80)
    top_level_items: list[WorkspaceItem] = Field(default_factory=list, max_length=100)
    docs_detected: list[str] = Field(default_factory=list, max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=20)


class WorkspaceSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    max_results: int = Field(default=20, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query cannot be blank")
        return cleaned


class WorkspaceSearchMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1, max_length=500)
    line_number: int = Field(ge=1)
    snippet: str = Field(min_length=1, max_length=300)


class WorkspaceSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    results: list[WorkspaceSearchMatch] = Field(default_factory=list, max_length=50)
    files_scanned: int = Field(ge=0, le=500)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list, max_length=20)
