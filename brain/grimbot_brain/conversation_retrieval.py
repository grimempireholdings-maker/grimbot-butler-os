from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_RETRIEVAL_QUERY_LENGTH = 480

_FILLER_WORDS = {
    "actually",
    "anyway",
    "basically",
    "etc",
    "haha",
    "hahaha",
    "honestly",
    "idk",
    "just",
    "kinda",
    "kind",
    "like",
    "literally",
    "lol",
    "lmao",
    "maybe",
    "probably",
    "really",
    "sort",
    "stuff",
    "things",
    "uh",
    "um",
    "very",
    "whatever",
    "you",
    "know",
}

_STOPWORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "with",
    "would",
}

_PREFERRED_PHRASES = (
    "grim empire",
    "grimbot butler os",
    "grim bot",
    "grimbot",
    "maya console",
    "maya",
    "real estate",
    "land flipping",
    "apex acquisitions",
    "autoshift",
    "bird dash",
    "birddash",
    "grim curriculum",
    "architecture",
    "jarvis",
    "optimus",
    "openclaw",
    "codex",
    "chief of staff",
    "procedural memory",
    "adaptive state",
    "dreaming",
    "memory",
    "openrouter",
    "voice",
    "skills",
    "safety",
)

_PREFERRED_TOKENS = {
    "adaptive",
    "apex",
    "architecture",
    "autoshift",
    "birddash",
    "briefing",
    "body",
    "butler",
    "clawdbot",
    "console",
    "context",
    "codex",
    "curriculum",
    "dreaming",
    "estate",
    "grim",
    "grimbot",
    "hardware",
    "jarvis",
    "land",
    "maya",
    "memory",
    "openrouter",
    "openclaw",
    "optimus",
    "planner",
    "priority",
    "procedure",
    "procedural",
    "project",
    "real",
    "revenue",
    "robot",
    "safety",
    "skills",
    "voice",
}

_DEFAULT_FALLBACK_TERMS = ("active projects", "top priorities", "recent context")


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    source_terms: list[str] = field(default_factory=list)
    truncated: bool = False
    fallback_used: bool = False

    def machine_output(self) -> dict:
        return {
            "query": self.query,
            "query_length": len(self.query),
            "max_length": MAX_RETRIEVAL_QUERY_LENGTH,
            "source_terms": self.source_terms[:20],
            "truncated": self.truncated,
            "fallback_used": self.fallback_used,
        }


def build_retrieval_query(message: str, max_length: int = MAX_RETRIEVAL_QUERY_LENGTH) -> RetrievalQuery:
    safe_max = max(1, min(max_length, MAX_RETRIEVAL_QUERY_LENGTH))
    normalized = _normalize(message)
    tokens = normalized.split()
    terms: list[str] = []

    for phrase in _PREFERRED_PHRASES:
        if phrase in normalized:
            terms.append(phrase)

    preferred_tokens = [
        token
        for token in tokens
        if token in _PREFERRED_TOKENS and token not in _FILLER_WORDS
    ]
    general_tokens = [
        token
        for token in tokens
        if _is_searchable_token(token) and token not in _PREFERRED_TOKENS
    ]
    terms.extend(preferred_tokens)
    terms.extend(general_tokens[:80])
    terms = _dedupe(terms)

    fallback_used = False
    if not terms:
        terms = list(_DEFAULT_FALLBACK_TERMS)
        fallback_used = True

    query, truncated = _truncate_terms(terms, safe_max)
    if not query:
        query = "active projects top priorities recent context"[:safe_max]
        terms = query.split()
        fallback_used = True

    return RetrievalQuery(
        query=query,
        source_terms=terms,
        truncated=truncated,
        fallback_used=fallback_used,
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _is_searchable_token(token: str) -> bool:
    return (
        len(token) > 2
        and token not in _FILLER_WORDS
        and token not in _STOPWORDS
        and not token.isdigit()
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _truncate_terms(terms: list[str], max_length: int) -> tuple[str, bool]:
    query = ""
    truncated = False
    for term in terms:
        candidate = term if not query else f"{query} {term}"
        if len(candidate) > max_length:
            truncated = True
            break
        query = candidate
    return query, truncated
