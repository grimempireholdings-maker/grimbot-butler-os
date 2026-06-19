from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from .memory import BrainMemory


TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
SEARCH_TIMEOUT_SECONDS = 5.0
SEARCH_CACHE_TTL_SECONDS = 60 * 60
TAVILY_MONTHLY_LIMIT = int(os.getenv("TAVILY_MONTHLY_LIMIT", "1000"))

_NEWS_KEYWORDS = frozenset({"news", "latest", "happening", "headlines", "current events", "breaking"})
_WEATHER_KEYWORDS = frozenset({"weather", "forecast", "temperature", "humidity", "rain", "snow"})


class SearchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    snippet: str = Field(default="", max_length=4000)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=240)
    success: bool
    results: list[SearchItem] = Field(default_factory=list, max_length=10)
    answer: str | None = Field(default=None, max_length=4000)
    reason: str | None = Field(default=None, max_length=500)
    cached: bool = False
    topic: str = "general"
    days: int | None = None


class SearchUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    month: str
    count: int
    limit: int
    remaining: int


@dataclass(frozen=True)
class _CacheEntry:
    created_at: float
    result: SearchResult


_SEARCH_CACHE: dict[tuple[str, int, str, int | None], _CacheEntry] = {}
_CACHE_LOCK = Lock()
_USAGE_LOCK = Lock()


def search_web(
    query: str,
    max_results: int = 5,
    *,
    topic: str = "general",
    days: int | None = None,
    memory: BrainMemory | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SearchResult:
    """Run one bounded Tavily snippet search without fetching result URLs."""
    normalized_query = _normalize_query(query)
    safe_max_results = max(1, min(int(max_results), 10))
    safe_topic = topic if topic in ("general", "news") else "general"
    safe_days = max(1, min(int(days), 30)) if days is not None else None

    if not normalized_query:
        result = SearchResult(
            query="invalid query",
            success=False,
            reason="Search query was empty.",
            topic=safe_topic,
            days=safe_days,
        )
        _log_search(memory, result)
        return result

    cache_key = (normalized_query.casefold(), safe_max_results, safe_topic, safe_days)
    now = clock()
    with _CACHE_LOCK:
        entry = _SEARCH_CACHE.get(cache_key)
        if entry and now - entry.created_at < SEARCH_CACHE_TTL_SECONDS:
            cached = entry.result.model_copy(update={"cached": True})
            _log_search(memory, cached)
            return cached
        if entry:
            _SEARCH_CACHE.pop(cache_key, None)

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        result = SearchResult(
            query=normalized_query,
            success=False,
            reason="TAVILY_API_KEY is not configured.",
            topic=safe_topic,
            days=safe_days,
        )
        _log_search(memory, result)
        return result

    try:
        payload = _request_search(normalized_query, safe_max_results, api_key, safe_topic, safe_days)
        items = _parse_items(payload, safe_max_results)
        answer = _optional_text(payload.get("answer"), 4000)
        result = SearchResult(
            query=normalized_query,
            success=True,
            results=items,
            answer=answer,
            topic=safe_topic,
            days=safe_days,
        )
        with _CACHE_LOCK:
            _SEARCH_CACHE[cache_key] = _CacheEntry(created_at=now, result=result)
        _increment_usage()
    except Exception as exc:
        result = SearchResult(
            query=normalized_query,
            success=False,
            reason=_failure_reason(exc),
            topic=safe_topic,
            days=safe_days,
        )

    _log_search(memory, result)
    return result


def search_usage() -> SearchUsage:
    """Return current month's Tavily API call count against the monthly cap."""
    month = datetime.now().strftime("%Y-%m")
    data = _read_usage_file()
    count = data.get(month, 0)
    return SearchUsage(
        month=month,
        count=count,
        limit=TAVILY_MONTHLY_LIMIT,
        remaining=max(0, TAVILY_MONTHLY_LIMIT - count),
    )


def clear_search_cache() -> None:
    """Clear process-local search cache; intended for tests and operator resets."""
    with _CACHE_LOCK:
        _SEARCH_CACHE.clear()


def topic_for_query(query: str) -> tuple[str, int | None]:
    """Infer Tavily topic and days lookback from a search query string."""
    lower = query.lower()
    tokens = set(re.split(r"\W+", lower))
    if tokens & _NEWS_KEYWORDS:
        return "news", 3
    if tokens & _WEATHER_KEYWORDS:
        return "general", 1
    return "general", None


def _request_search(
    query: str,
    max_results: int,
    api_key: str,
    topic: str = "general",
    days: int | None = None,
) -> dict:
    body_dict: dict = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": "basic",
        "include_raw_content": False,
        "topic": topic,
    }
    if days is not None:
        body_dict["days"] = days
    body = json.dumps(body_dict).encode("utf-8")
    request = urllib.request.Request(
        TAVILY_SEARCH_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Tavily HTTP {exc.code}: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Tavily returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Tavily returned an invalid response shape.")
    return payload


def _parse_items(payload: dict, max_results: int) -> list[SearchItem]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []
    items: list[SearchItem] = []
    for raw in raw_results[:max_results]:
        if not isinstance(raw, dict):
            continue
        title = _optional_text(raw.get("title"), 500)
        url = _optional_text(raw.get("url"), 2000)
        if not title or not url or not url.startswith(("http://", "https://")):
            continue
        snippet = _optional_text(raw.get("content"), 4000) or ""
        items.append(SearchItem(title=title, url=url, snippet=snippet))
    return items


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query)).strip()[:240]


def _optional_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:limit] or None


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return "Web search timed out or could not reach Tavily within 5 seconds."
    return f"Web search failed: {str(exc)[:420]}"


def _usage_counter_path() -> Path:
    db_path = Path(os.getenv("GRIMBOT_DB_PATH", "memory/grimbot_brain.sqlite3"))
    return db_path.parent / "tavily_usage.json"


def _read_usage_file() -> dict:
    path = _usage_counter_path()
    with _USAGE_LOCK:
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            return {}


def _increment_usage() -> None:
    path = _usage_counter_path()
    month = datetime.now().strftime("%Y-%m")
    with _USAGE_LOCK:
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            data = {}
        data[month] = data.get(month, 0) + 1
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass


def _log_search(memory: BrainMemory | None, result: SearchResult) -> None:
    if memory is None:
        return
    event = {
        "query": result.query,
        "success": result.success,
        "result_count": len(result.results),
        "cached": result.cached,
        "topic": result.topic,
        "days": result.days,
        "reason": result.reason,
    }
    try:
        memory.log_episode(
            kind="web_search",
            content=json.dumps(event, ensure_ascii=True, sort_keys=True),
            importance=0.45 if result.success else 0.6,
        )
    except Exception:
        # Search availability must never depend on secondary observability writes.
        return
