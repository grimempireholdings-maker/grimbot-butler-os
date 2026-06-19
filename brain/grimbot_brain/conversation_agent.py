from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol
from weakref import WeakKeyDictionary

from pydantic import ValidationError

from .capabilities import capabilities_manifest, capabilities_prompt_block
from .conversation_schemas import (
    ConversationIntent,
    ConversationMode,
    ConversationSuggestion,
    ConversationalAgentResponse,
)
from .conversation_retrieval import RetrievalQuery, build_retrieval_query
from .identity.context_schemas import ContextSearchRequest, ContextSearchResult, ProjectContext
from .identity.context_store import ContextStore
from .maya_core import build_maya_briefing
from .memory import BrainMemory
from .procedural_memory.procedure_matcher import ProcedureMatcher
from .procedural_memory.procedure_schemas import ProcedureMatchRequest, ProcedureMatchResult
from .procedural_memory.procedure_store import ProcedureStore
from .robot_memory import RobotMemory
from .workspace.workspace_inspector import WorkspaceInspector
from .schemas import (
    MayaBriefing,
    MayaBriefingRequest,
    RelevantMemoryRequest,
    RelevantMemoryResult,
    VoiceConversationRequest,
)


@dataclass
class ConversationSessionState:
    recent_messages: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    recent_modes: deque[ConversationMode] = field(default_factory=lambda: deque(maxlen=6))
    recent_recommendations: deque[str] = field(default_factory=lambda: deque(maxlen=3))
    recent_turns: deque[dict] = field(default_factory=lambda: deque(maxlen=6))


@dataclass(frozen=True)
class ConversationRuntime:
    mode: ConversationMode = "unclear"
    recent_messages: tuple[str, ...] = ()
    recent_recommendations: tuple[str, ...] = ()
    recent_turns: tuple[dict, ...] = ()


_SESSION_STATES: WeakKeyDictionary[BrainMemory, ConversationSessionState] = WeakKeyDictionary()
_SESSION_LOCK = Lock()
_CONVERSATION_RUNTIME: ContextVar[ConversationRuntime] = ContextVar(
    "conversation_runtime",
    default=ConversationRuntime(),
)
_CLASSIFICATION_SOURCE: ContextVar[str] = ContextVar(
    "classification_source",
    default="unknown",
)


class ConversationProvider(Protocol):
    name: str

    def generate(self, prompt: str, fallback_response: ConversationalAgentResponse) -> ConversationalAgentResponse:
        ...


@dataclass(frozen=True)
class MockConversationProvider:
    name: str = "mock"

    def generate(self, prompt: str, fallback_response: ConversationalAgentResponse) -> ConversationalAgentResponse:
        return fallback_response


@dataclass(frozen=True)
class ApiConversationProvider:
    name: str
    env_key: str
    model_env_key: str
    default_model: str

    def generate(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
    ) -> ConversationalAgentResponse:
        api_key = os.getenv(self.env_key, "").strip()
        if not api_key:
            return _fallback_with_reason(
                fallback_response,
                f"missing {self.env_key}",
                attempted_provider=self.name,
            )
        try:
            raw_text = self._call(prompt, fallback_response, api_key)
            parsed = _validated_llm_response(raw_text)
        except Exception as exc:
            return _fallback_with_reason(
                fallback_response,
                f"{self.name} provider fallback: {exc}",
                attempted_provider=self.name,
            )
        return _safe_provider_response(fallback_response, parsed, self.name)

    def _call(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
        api_key: str,
    ) -> str:
        raise NotImplementedError

    def classify_mode(self, prompt: str, timeout: float = 3.0) -> str:
        api_key = os.getenv(self.env_key, "").strip()
        if not api_key:
            raise ValueError(f"missing {self.env_key}")
        return self._call_classification(prompt, api_key, timeout)

    def _call_classification(self, prompt: str, api_key: str, timeout: float) -> str:
        raise NotImplementedError

    def _model(self) -> str:
        return os.getenv(self.model_env_key, self.default_model).strip() or self.default_model

    def _classifier_model(self) -> str:
        override = os.getenv("GRIMBOT_CONVERSATION_CLASSIFIER_MODEL", "").strip()
        return override if override else self._model()


class ClaudeConversationProvider(ApiConversationProvider):
    def _call_classification(self, prompt: str, api_key: str, timeout: float) -> str:
        payload = {
            "model": self._classifier_model(),
            "max_tokens": 64,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=timeout,
        )
        content = response.get("content", [])
        if not content:
            raise ValueError("Claude classification response missing content")
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()

    def _call(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
        api_key: str,
    ) -> str:
        payload = {
            "model": self._model(),
            "max_tokens": _provider_max_tokens(),
            "temperature": 0.4,
            "system": _provider_system_prompt(),
            "messages": [{"role": "user", "content": _provider_user_prompt(prompt, fallback_response)}],
        }
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        content = response.get("content", [])
        if not content or not isinstance(content, list):
            raise ValueError("Claude response missing content")
        text = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
        if not text:
            raise ValueError("Claude response missing text")
        return text


class OpenAIConversationProvider(ApiConversationProvider):
    def _call_classification(self, prompt: str, api_key: str, timeout: float) -> str:
        payload = {
            "model": self._classifier_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }
        response = _post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("OpenAI classification response missing choices")
        return (choices[0].get("message", {}).get("content") or "").strip()

    def _call(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
        api_key: str,
    ) -> str:
        payload = {
            "model": self._model(),
            "input": [
                {"role": "system", "content": _provider_system_prompt()},
                {"role": "user", "content": _provider_user_prompt(prompt, fallback_response)},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "conversational_agent_response",
                    "schema": _conversation_response_schema(),
                    "strict": True,
                }
            },
            "max_output_tokens": _provider_max_tokens(),
            "temperature": 0.4,
        }
        response = _post_json(
            "https://api.openai.com/v1/responses",
            payload,
            {"Authorization": f"Bearer {api_key}"},
        )
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("text"):
                    return str(content["text"])
        raise ValueError("OpenAI response missing text")


class OpenRouterConversationProvider(ApiConversationProvider):
    def _call_classification(self, prompt: str, api_key: str, timeout: float) -> str:
        payload = {
            "model": self._classifier_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }
        headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        headers["X-OpenRouter-Title"] = "GrimBot Butler OS"
        response = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            payload,
            headers,
            timeout=timeout,
        )
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("OpenRouter classification response missing choices")
        message = choices[0].get("message", {})
        # Reasoning models (e.g. DeepSeek-R1 via openrouter/free) return content: null
        # and put the actual response in reasoning_content. Accept either.
        raw = (message.get("content") or message.get("reasoning_content") or "").strip()
        if not raw:
            raise ValueError("OpenRouter classification response was empty")
        return raw

    def _call(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
        api_key: str,
    ) -> str:
        payload = {
            "model": self._model(),
            "messages": [
                {"role": "system", "content": _provider_system_prompt()},
                {"role": "user", "content": _provider_user_prompt(prompt, fallback_response)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": _provider_max_tokens(),
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        headers["X-OpenRouter-Title"] = "GrimBot Butler OS"
        response = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            payload,
            headers,
        )
        choices = response.get("choices", [])
        if not choices or not isinstance(choices, list):
            raise ValueError("OpenRouter response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter response missing content")
        return content


class GeminiConversationProvider(ApiConversationProvider):
    def _call_classification(self, prompt: str, api_key: str, timeout: float) -> str:
        model = self._classifier_model()
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 64},
        }
        response = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            payload,
            {},
            timeout=timeout,
        )
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini classification response missing candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()

    def _call(
        self,
        prompt: str,
        fallback_response: ConversationalAgentResponse,
        api_key: str,
    ) -> str:
        model = self._model()
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{_provider_system_prompt()}\n\n{_provider_user_prompt(prompt, fallback_response)}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": _provider_max_tokens(),
                "responseMimeType": "application/json",
                "responseSchema": _conversation_response_schema(),
            },
        }
        response = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            payload,
            {},
        )
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini response missing candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise ValueError("Gemini response missing text")
        return text


def provider_from_env() -> ConversationProvider:
    provider = os.getenv("GRIMBOT_CONVERSATION_PROVIDER", "mock").strip().lower()
    if provider == "auto":
        if os.getenv("ANTHROPIC_API_KEY", "").strip():
            return ClaudeConversationProvider(
                "claude",
                "ANTHROPIC_API_KEY",
                "GRIMBOT_CONVERSATION_CLAUDE_MODEL",
                "claude-3-5-sonnet-latest",
            )
        if os.getenv("OPENAI_API_KEY", "").strip():
            return OpenAIConversationProvider(
                "openai",
                "OPENAI_API_KEY",
                "GRIMBOT_CONVERSATION_OPENAI_MODEL",
                "gpt-4.1-mini",
            )
        if os.getenv("OPENROUTER_API_KEY", "").strip():
            return OpenRouterConversationProvider(
                "openrouter",
                "OPENROUTER_API_KEY",
                "OPENROUTER_MODEL",
                "openrouter/auto",
            )
        if os.getenv("GEMINI_API_KEY", "").strip():
            return GeminiConversationProvider(
                "gemini",
                "GEMINI_API_KEY",
                "GRIMBOT_CONVERSATION_GEMINI_MODEL",
                "gemini-1.5-flash",
            )
        return MockConversationProvider()
    if provider == "gemini":
        return GeminiConversationProvider("gemini", "GEMINI_API_KEY", "GRIMBOT_CONVERSATION_GEMINI_MODEL", "gemini-1.5-flash")
    if provider == "openai":
        return OpenAIConversationProvider("openai", "OPENAI_API_KEY", "GRIMBOT_CONVERSATION_OPENAI_MODEL", "gpt-4.1-mini")
    if provider == "openrouter":
        return OpenRouterConversationProvider("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "openrouter/auto")
    if provider in {"claude", "anthropic"}:
        return ClaudeConversationProvider("claude", "ANTHROPIC_API_KEY", "GRIMBOT_CONVERSATION_CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    return MockConversationProvider()


_VALID_MODES: frozenset = frozenset({
    "casual", "morning_orientation", "work_focus", "personal_support",
    "business_strategy", "project_context", "workspace_awareness",
    "physical_environment", "feedback_about_maya", "capability_question", "unclear",
})


def _build_classification_prompt(transcript: str, recent_turns: tuple[dict, ...]) -> str:
    mode_list = (
        "casual, morning_orientation, work_focus, personal_support, business_strategy, "
        "project_context, workspace_awareness, physical_environment, feedback_about_maya, "
        "capability_question, unclear"
    )
    lines = [
        "TASK: Output exactly one mode name from the list below. No explanation, no punctuation, no other text.",
        "",
        f"Valid modes: {mode_list}",
        "",
        "Key rules:",
        "- feedback_about_maya: Julian is reacting to something Maya just said — correction, pushback, or meta-comment about her behavior. Requires conversation context to identify.",
        "- morning_orientation: open-ended day-start check-in with no specific task request.",
        "- capability_question: asking what Maya can/cannot access or do; also covers requests for external data (news, weather, internet) she does not have.",
        "- casual: small talk not requesting work output.",
        "- unclear: genuinely ambiguous after considering full context; prefer a specific mode if one fits.",
        "",
    ]
    if recent_turns:
        lines.append("Recent conversation (oldest to newest):")
        for turn in recent_turns[-3:]:
            lines.append(f'  Julian: "{turn["user"]}"')
            lines.append(f'  Maya:   "{turn["maya"]}"')
        lines.append("")
    lines.append(f'Current message from Julian: "{transcript}"')
    lines.append("")
    lines.append("Mode:")
    return "\n".join(lines)


def _parse_mode(raw: str) -> ConversationMode:
    mode = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if mode in _VALID_MODES:
        return mode  # type: ignore[return-value]
    # Secondary scan: find any valid mode token embedded in verbose output
    normalized = raw.lower().replace("-", "_")
    for candidate in sorted(_VALID_MODES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(candidate) + r"\b", normalized):
            return candidate  # type: ignore[return-value]
    raise ValueError(f"LLM returned unrecognized mode: {raw!r}")


def _classify_via_llm(
    transcript: str,
    recent_turns: tuple[dict, ...],
    provider: ConversationProvider,
    timeout: float = 3.0,
) -> ConversationMode:
    if not isinstance(provider, ApiConversationProvider):
        raise NotImplementedError("provider does not support LLM classification")
    prompt = _build_classification_prompt(transcript, recent_turns)
    raw = provider.classify_mode(prompt, timeout=timeout)
    return _parse_mode(raw)


def classify_conversation_mode_with_fallback(
    transcript: str,
    recent_turns: tuple[dict, ...],
    provider: ConversationProvider,
    timeout: float = 3.0,
) -> tuple[ConversationMode, str]:
    """Return (mode, source) where source is 'llm' or 'rule_based:<reason>'."""
    try:
        mode = _classify_via_llm(transcript, recent_turns, provider, timeout=timeout)
        return mode, "llm"
    except NotImplementedError as exc:
        return classify_conversation_mode(transcript), f"rule_based:not_implemented:{exc}"
    except Exception as exc:
        return classify_conversation_mode(transcript), f"rule_based:fallback:{type(exc).__name__}:{str(exc)[:120]}"


def run_conversation_agent(
    request: VoiceConversationRequest,
    transcript: str,
    memory: BrainMemory,
    memory_context: RelevantMemoryResult | None = None,
    provider: ConversationProvider | None = None,
    retrieval_query: RetrievalQuery | None = None,
    memory_retrieval_error: str | None = None,
) -> ConversationalAgentResponse:
    robot_memory = RobotMemory(memory)
    context = ContextStore(memory)
    retrieval_query = retrieval_query or build_retrieval_query(transcript)
    effective_provider = provider or provider_from_env()
    with _SESSION_LOCK:
        _recent_turns = tuple(_SESSION_STATES.get(memory, ConversationSessionState()).recent_turns)
    conversation_mode, _classification_source = classify_conversation_mode_with_fallback(
        transcript,
        recent_turns=_recent_turns,
        provider=effective_provider,
    )
    runtime = _runtime_for(memory, conversation_mode)
    context_retrieval_error = None
    if memory_context is None:
        if conversation_mode == "physical_environment":
            try:
                memory_context = robot_memory.relevant(
                    RelevantMemoryRequest(
                        query=retrieval_query.query,
                        room_name=request.room_name,
                        zone_name=request.zone_name,
                        limit=10,
                    )
                )
            except Exception:
                memory_retrieval_error = "memory_retrieval_failed"
                memory_context = _empty_memory_context(retrieval_query, request)
        else:
            memory_context = _empty_memory_context(retrieval_query, request)

    context_result = _empty_context_result(retrieval_query.query)
    if conversation_mode not in {
        "capability_question",
        "workspace_awareness",
        "casual",
        "physical_environment",
    }:
        if conversation_mode == "feedback_about_maya":
            context_request = ContextSearchRequest(
                query="Maya conversation architecture feedback",
                limit=10,
            )
        elif conversation_mode == "personal_support":
            context_request = ContextSearchRequest(
                query="Julian personal priorities relationships",
                context_types=["person_profile", "priority", "relationship"],
                limit=10,
            )
        else:
            context_request = ContextSearchRequest(query=retrieval_query.query, limit=10)
        try:
            context_result = context.search(context_request)
        except Exception:
            context_retrieval_error = "context_retrieval_failed"
            context_result = _fallback_context_result(context, retrieval_query)
    projects = context_result.projects
    if not projects and conversation_mode in {
        "morning_orientation",
        "work_focus",
        "business_strategy",
        "project_context",
    }:
        projects = _safe_projects(context)
    intent = classify_intent(transcript, request, context_result, projects)
    provider = effective_provider
    runtime_token = _CONVERSATION_RUNTIME.set(runtime)
    source_token = _CLASSIFICATION_SOURCE.set(_classification_source)
    try:
        if conversation_mode == "capability_question":
            agent_response = _capability_response(transcript, intent, provider)
        elif conversation_mode == "feedback_about_maya":
            agent_response = _feedback_response(transcript, provider)
        elif conversation_mode == "workspace_awareness":
            agent_response = _workspace_response(transcript, provider)
        elif conversation_mode == "morning_orientation":
            agent_response = _morning_response(transcript, request, context, provider)
        elif conversation_mode in {"work_focus", "business_strategy"}:
            agent_response = _work_focus_response(transcript, request, context, provider)
        elif conversation_mode == "personal_support":
            agent_response = _personal_support_response(transcript, context_result, provider)
        elif conversation_mode == "casual":
            agent_response = _casual_response(transcript, context, provider)
        elif intent == "chief_of_staff_briefing":
            agent_response = _briefing_response(transcript, request, robot_memory, context, provider)
        elif intent == "project_recall":
            agent_response = _project_recall_response(request, context_result, provider)
        elif intent == "room_or_physical_request":
            agent_response = _physical_response(request, transcript, memory_context, provider)
        elif intent == "skill_request":
            agent_response = _skill_response(transcript, context_result, provider)
        elif intent == "procedure_request":
            agent_response = _procedure_response(transcript, memory, provider)
        elif intent == "dream_review":
            agent_response = _dream_response(transcript, provider)
        elif intent == "workspace_awareness":
            agent_response = _workspace_response(transcript, provider)
        elif intent == "memory_search":
            agent_response = _memory_search_response(transcript, context_result, provider)
        elif intent == "casual_chat":
            agent_response = _casual_response(transcript, context, provider)
        else:
            agent_response = _unclear_response(transcript, context_result, provider)
    finally:
        _CONVERSATION_RUNTIME.reset(runtime_token)
        _CLASSIFICATION_SOURCE.reset(source_token)

    response = _attach_retrieval_metadata(
        agent_response,
        retrieval_query,
        memory_retrieval_error=memory_retrieval_error,
        context_retrieval_error=context_retrieval_error,
    )
    _record_conversation(memory, transcript, conversation_mode, response)
    return response


def _empty_memory_context(
    retrieval_query: RetrievalQuery,
    request: VoiceConversationRequest,
) -> RelevantMemoryResult:
    return RelevantMemoryResult(
        query=retrieval_query.query,
        room_name=request.room_name,
        hazards=[],
        mess_zones=[],
        cleanup_tasks=[],
        semantic_facts=[],
        next_best_action="No physical-memory lookup was needed for this conversation mode.",
    )


def _empty_context_result(query: str) -> ContextSearchResult:
    return ContextSearchResult(
        query=query,
        entries=[],
        projects=[],
        next_best_action="Respond directly without forcing project context.",
        needs_clarification=False,
    )


def _runtime_for(memory: BrainMemory, mode: ConversationMode) -> ConversationRuntime:
    with _SESSION_LOCK:
        state = _SESSION_STATES.setdefault(memory, ConversationSessionState())
        return ConversationRuntime(
            mode=mode,
            recent_messages=tuple(state.recent_messages),
            recent_recommendations=tuple(state.recent_recommendations),
            recent_turns=tuple(state.recent_turns),
        )


def _record_conversation(
    memory: BrainMemory,
    transcript: str,
    mode: ConversationMode,
    response: ConversationalAgentResponse,
) -> None:
    recommendation = response.machine_output.get("recommended_focus")
    with _SESSION_LOCK:
        state = _SESSION_STATES.setdefault(memory, ConversationSessionState())
        state.recent_messages.append(transcript.strip()[:1000])
        state.recent_modes.append(mode)
        if isinstance(recommendation, str) and recommendation.strip():
            state.recent_recommendations.append(recommendation.strip())
        state.recent_turns.append({
            "user": transcript.strip()[:500],
            "maya": response.user_response.strip()[:500],
        })


def classify_conversation_mode(transcript: str) -> ConversationMode:
    normalized = _normalize(transcript)
    tokens = set(normalized.split())
    if not normalized or normalized == "input unavailable":
        return "unclear"

    feedback_phrases = (
        "hyperfocus",
        "hyper focusing",
        "overfocus",
        "over focusing",
        "too scripted",
        "too business",
        "too robotic",
        "not personal enough",
        "you keep focusing",
        "you keep asking",
        "you already asked",
        "i already explained",
        "i just explained",
        "stop treating every",
        "feedback for you",
        "that is not what i meant",
        "that s not what i meant",
    )
    if any(phrase in normalized for phrase in feedback_phrases):
        return "feedback_about_maya"
    if _is_workspace_request(normalized):
        return "workspace_awareness"

    capability_terms = {
        "camera",
        "microphone",
        "screen",
        "screens",
        "tab",
        "tabs",
        "device",
        "devices",
        "layout",
        "capability",
        "capabilities",
    }
    capability_phrases = (
        "what can you access",
        "what can you do",
        "do you have access",
        "can you see",
        "can you hear",
        "can you use",
        "are you able to",
    )
    if tokens & capability_terms and any(
        phrase in normalized for phrase in capability_phrases
    ):
        return "capability_question"
    if "what are you capable of" in normalized:
        return "capability_question"
    external_data_terms = {"news", "weather", "forecast", "temperature", "stock", "stocks", "market", "markets", "internet", "web", "google"}
    external_data_phrases = (
        "what s happening out there",
        "what is happening out there",
        "any news",
        "anything new out there",
        "what s the weather",
        "what is the weather",
        "check the weather",
        "what s the latest",
        "what is the latest",
        "current events",
        "search for",
        "look that up",
        "look it up",
    )
    if tokens & external_data_terms or any(phrase in normalized for phrase in external_data_phrases):
        return "capability_question"

    morning_phrases = (
        "morning maya",
        "good morning",
        "how s it going",
        "hows it going",
        "anything interesting happening today",
        "anything interesting today",
        "how is my day looking",
        "how s my day looking",
        "hows my day looking",
    )
    if any(phrase in normalized for phrase in morning_phrases):
        return "morning_orientation"
    if any(
        phrase in normalized
        for phrase in ("what should i work on", "what should i focus on", "my priorities", "work focus")
    ):
        return "work_focus"
    if tokens & {"business", "revenue", "cashflow", "acquisitions", "deals", "buyers", "sellers"}:
        return "business_strategy"
    if tokens & {"tired", "groggy", "overwhelmed", "stressed", "burned", "burnt"}:
        return "personal_support"
    if _is_physical_request(normalized, None, None):
        return "physical_environment"
    if tokens & {"grimbot", "autoshift", "birddash", "architecture", "project", "repo"}:
        return "project_context"
    if _is_casual_chat(normalized) or tokens & {"thanks", "okay", "cool", "nice", "funny", "joking"}:
        return "casual"
    return "unclear"


def _fallback_context_result(context: ContextStore, retrieval_query: RetrievalQuery) -> ContextSearchResult:
    try:
        summary = context.summary()
    except Exception:
        return ContextSearchResult(
            query=retrieval_query.query,
            entries=[],
            projects=[],
            next_best_action="Ask one clarifying question before recommending action.",
            needs_clarification=True,
            clarification_question="Which project, priority, person, or decision should I focus on?",
        )
    entries = [*summary.priorities[:3], *summary.bottlenecks[:2], *summary.next_actions[:2]]
    projects = summary.projects[:3]
    if projects:
        next_action = projects[0].next_action
    elif entries:
        next_action = entries[0].content
    else:
        next_action = "Ask one clarifying question before recommending action."
    return ContextSearchResult(
        query=retrieval_query.query,
        entries=entries,
        projects=projects,
        next_best_action=next_action,
        needs_clarification=not entries and not projects,
        clarification_question=(
            "Which project, priority, person, or decision should I focus on?"
            if not entries and not projects
            else None
        ),
    )


def _safe_projects(context: ContextStore) -> list[ProjectContext]:
    try:
        return context.projects()
    except Exception:
        return []


def _attach_retrieval_metadata(
    response: ConversationalAgentResponse,
    retrieval_query: RetrievalQuery,
    memory_retrieval_error: str | None,
    context_retrieval_error: str | None,
) -> ConversationalAgentResponse:
    errors = {}
    if memory_retrieval_error:
        errors["memory"] = {
            "status": "fallback",
            "reason": "Memory retrieval failed; fallback context was used.",
        }
    if context_retrieval_error:
        errors["context"] = {
            "status": "fallback",
            "reason": "Context retrieval failed; top projects and priorities were used.",
        }
    machine_output = {
        **response.machine_output,
        "retrieval": retrieval_query.machine_output(),
        "retrieval_status": "fallback" if errors else "ok",
    }
    if errors:
        machine_output["retrieval_errors"] = errors
    return response.model_copy(update={"machine_output": machine_output})


def classify_intent(
    transcript: str,
    request: VoiceConversationRequest,
    context_result: ContextSearchResult,
    projects: list[ProjectContext],
) -> ConversationIntent:
    normalized = _normalize(transcript)
    tokens = set(normalized.split())

    if not normalized or normalized == "input unavailable":
        return "unclear"
    if _is_briefing_request(normalized):
        return "chief_of_staff_briefing"
    if _is_project_recall(normalized, context_result, projects):
        return "project_recall"
    if tokens & {"skill", "skills"}:
        return "skill_request"
    if tokens & {"procedure", "procedures", "workflow", "workflows", "process", "processes"}:
        return "procedure_request"
    if tokens & {"dream", "dreaming", "promotion", "promotions", "facts", "fact"}:
        return "dream_review"
    if _is_workspace_request(normalized):
        return "workspace_awareness"
    if _is_physical_request(normalized, request.room_name, request.zone_name):
        return "room_or_physical_request"
    if _is_casual_chat(normalized):
        return "casual_chat"
    if context_result.projects or context_result.entries:
        return "memory_search"
    return "unclear"


def build_conversation_prompt(
    transcript: str,
    intent: ConversationIntent,
    retrieved_context: list[dict],
    machine_output: dict,
    conversation_mode: ConversationMode = "unclear",
    recent_messages: tuple[str, ...] = (),
) -> str:
    mode_constraints: list[str] = []
    if conversation_mode in _HUMAN_MOMENT_MODES:
        _is_direct_question = "?" in transcript or any(
            _normalize(transcript).startswith(w)
            for w in ("what", "how", "any", "tell", "is", "can", "does", "do", "give", "catch", "fill")
        )
        mode_constraints = [
            f"MODE CONSTRAINT ({conversation_mode}): This is a human moment, not a task assignment.",
            "Do NOT name, recommend, or focus on any specific project in user_response.",
            "Do NOT surface priority_items, active_projects, or open_loops in user_response.",
        ]
        if conversation_mode == "morning_orientation" and not _is_direct_question:
            mode_constraints.append("Julian has not asked a specific question — ask what he wants to focus on; do not choose for him.")

    return "\n".join(
        [
            "Maya is Julian's operator and Chief of Staff layer; she behaves like one without saying the title.",
            "Voice: warm, direct, slightly dry humor, never sycophantic.",
            "Lead with the human moment, then surface operational signal when useful.",
            "Call Julian Boss naturally, not robotically, and do not overuse it.",
            "Never start with a disclaimer. Use 'Not verified yet' only when a factual claim needs verification.",
            "Never narrate internal lookup work. Just answer.",
            "Keep machine_output separate from user_response.",
            "Safety rules: no motors, hardware, external tools, procedure execution, auto-approval, or safety override.",
            "Do not force business advice into casual conversation.",
            "Do not force real estate into every response.",
            "If Julian sounds tired, groggy, joking, or conversational, respond human-first.",
            "Ask what he wants to focus on instead of assigning a focus every time.",
            "Be useful without hijacking the conversation.",
            *mode_constraints,
            (
                "NON-NEGOTIABLE CAPABILITY RULE: Do not claim capabilities you do not have. "
                "You have read-only local workspace access only. No camera, microphone, internet, screen, "
                "browser tabs, device layout, physical room, or external tools. "
                "If Julian asks about those, say plainly you do not have that yet — do not describe what you would do if you did."
            ),
            *(
                [
                    "CAPABILITIES manifest (inject only for capability_question mode):",
                    capabilities_prompt_block(),
                ]
                if conversation_mode == "capability_question"
                else []
            ),
            f"Conversation mode: {conversation_mode}",
            f"Intent: {intent}",
            f"User message: {transcript}",
            f"Recent user messages (oldest to newest): {list(recent_messages[-3:])}",
            "Do not repeat a clarification already answered in the recent messages.",
            f"Retrieved context: {retrieved_context}",
            f"Machine output: {machine_output}",
        ]
    )


def _briefing_response(
    transcript: str,
    request: VoiceConversationRequest,
    robot_memory: RobotMemory,
    context: ContextStore,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    briefing = build_maya_briefing(
        request=MayaBriefingRequest(
            room_name=request.room_name,
            zone_name=request.zone_name,
            verified=request.verified,
            mode=request.assistant_mode,
        ),
        memory=robot_memory,
        context=context,
    )
    machine_output = briefing.model_dump()
    retrieved = _briefing_context(briefing)
    response = _briefing_text(briefing)
    return _response(
        intent="chief_of_staff_briefing",
        transcript=transcript,
        text=response,
        confidence=0.95,
        retrieved_context=retrieved,
        machine_output=machine_output,
        verified=briefing.verified,
        provider=provider,
    )


def _project_recall_response(
    request: VoiceConversationRequest,
    context_result: ContextSearchResult,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    machine_output = context_result.model_dump()
    if context_result.projects:
        project = context_result.projects[0]
        text = (
            f"{project.name} is {project.status}. The current bottleneck is "
            f"{project.current_bottleneck}. Next move: {project.next_action}"
        )
        verified = request.verified and project.verified
        if request.verified and not project.verified:
            text = f"I would treat this as unverified: {text}"
    elif context_result.entries:
        entry = context_result.entries[0]
        text = f"{entry.name}: {entry.content}"
        verified = request.verified and entry.verified
        if request.verified and not entry.verified:
            text = f"I would treat this as unverified: {text}"
    else:
        text = "I do not have a solid project match there. Which project do you want me to pull up?"
        verified = False
        machine_output["needs_clarification"] = True
    machine_output["context_summary"] = text
    return _response(
        intent="project_recall",
        transcript=context_result.query,
        text=text,
        confidence=0.92 if context_result.projects or context_result.entries else 0.45,
        retrieved_context=_context_rows(context_result),
        machine_output=machine_output,
        verified=verified,
        provider=provider,
    )


def _physical_response(
    request: VoiceConversationRequest,
    transcript: str,
    memory_context: RelevantMemoryResult,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    machine_output = memory_context.model_dump()
    if _is_camera_question(_normalize(transcript)):
        machine_output["camera_access"] = False
        machine_output["vision_invoked"] = False
        text = (
            "No, I cannot see through the camera from conversation alone. "
            "Camera vision requires an explicit room-scan request; I will not imply a live view."
        )
    else:
        text = (
            f"For the physical side, I would start with: {memory_context.next_best_action}. "
            "Safety stays in front; this is guidance only, not movement."
        )
    return _response(
        intent="room_or_physical_request",
        transcript=transcript,
        text=text,
        confidence=0.9,
        retrieved_context=_memory_rows(memory_context),
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


def _skill_response(
    transcript: str,
    context_result: ContextSearchResult,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    skill = _suggest_skill(transcript)
    machine_output = {
        **context_result.model_dump(),
        "skill_execution": "not_executed",
        "safety_note": "Conversation may suggest skills only; execution still requires existing permission gates.",
    }
    text = (
        f"I can suggest the {skill.name} skill for that. Permission level: "
        f"{skill.required_permission}. I will not run it from conversation."
    )
    return _response(
        intent="skill_request",
        transcript=transcript,
        text=text,
        confidence=0.84,
        retrieved_context=_context_rows(context_result),
        suggested_skill=skill,
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


def _procedure_response(
    transcript: str,
    memory: BrainMemory,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    match = _match_procedure(transcript, memory)
    machine_output = {
        "procedure_execution": "not_available",
        "procedure_match": match.model_dump(),
        "safety_note": "Procedures can be matched but not executed in this release.",
    }
    suggestion = None
    if match.matched and match.name and match.required_permission:
        suggestion = ConversationSuggestion(
            name=match.name,
            confidence=match.confidence,
            required_permission=match.required_permission,
            reason="Matched an active stored procedure; execution endpoint does not exist.",
        )
        text = (
            f"I found a procedure candidate: {match.name}. Confidence {match.confidence:.2f}. "
            "I can match it, but I cannot execute procedures in this release."
        )
    else:
        text = "I can review procedure memory, but I do not have a confident procedure match for that."
    return _response(
        intent="procedure_request",
        transcript=transcript,
        text=text,
        confidence=0.8 if match.matched else 0.55,
        retrieved_context=[],
        suggested_procedure=suggestion,
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


def _dream_response(transcript: str, provider: ConversationProvider) -> ConversationalAgentResponse:
    machine_output = {
        "dreaming": "manual_review_only",
        "auto_approval": False,
        "action_execution": "not_allowed",
    }
    return _response(
        intent="dream_review",
        transcript=transcript,
        text=(
            "Dream review is manual. I can help surface pending facts and promotions, "
            "but I will not approve, reject, or modify behavior on my own."
        ),
        confidence=0.82,
        retrieved_context=[],
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


def _morning_response(
    transcript: str,
    request: VoiceConversationRequest,
    context: ContextStore,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    summary = context.summary()
    project_names = [project.name for project in summary.projects[:3]]
    priorities = [entry.content for entry in summary.priorities[:3]]
    open_loops = [project.current_bottleneck for project in summary.projects[:3]]
    lanes = _human_list(project_names) or "no active lanes recorded"
    text = (
        f"Morning. I am here. Active lanes: {lanes}. "
        "This is orientation. What do you want to focus on?"
    )
    machine_output = {
        "priority_items": priorities,
        "active_projects": project_names,
        "open_loops": open_loops,
        "orientation_scope": "broad",
        "room_scan_requested": False,
    }
    return _response(
        intent="chief_of_staff_briefing",
        transcript=transcript,
        text=text,
        confidence=0.94,
        retrieved_context=[
            {"type": "priorities", "values": priorities},
            {"type": "projects", "values": project_names},
            {"type": "open_loops", "values": open_loops},
        ],
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


def _work_focus_response(
    transcript: str,
    request: VoiceConversationRequest,
    context: ContextStore,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    summary = context.summary()
    projects = summary.projects[:5]
    project_names = [project.name for project in projects[:3]]
    priorities = [entry.content for entry in summary.priorities[:3]]
    recommendation = _choose_recommended_focus(projects, transcript)
    lanes = _human_list(project_names) or "no active project lanes recorded"
    if recommendation:
        text = (
            f"The active lanes are {lanes}. A reasonable starting point is {recommendation}, "
            "but I am not collapsing the whole day into one project. Which lane has the right energy today?"
        )
    else:
        text = f"The active lanes are {lanes}. Which one has the right energy today?"
    machine_output = {
        "priority_items": priorities,
        "active_projects": project_names,
        "recommended_focus": recommendation,
        "recommendation_scope": "rotating_across_active_projects",
        "room_scan_requested": False,
    }
    return _response(
        intent="chief_of_staff_briefing",
        transcript=transcript,
        text=text,
        confidence=0.93,
        retrieved_context=[
            {"type": "priorities", "values": priorities},
            {"type": "projects", "values": project_names},
        ],
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


def _feedback_response(
    transcript: str,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    text = (
        "Fair. I was overweighting the highest stored priority instead of reading the conversation. "
        "That is a routing failure, not a reason to push harder. I am dropping the project assignment "
        "and physical-room fallback from this response and staying with the feedback you actually gave me."
    )
    return _response(
        intent="casual_chat",
        transcript=transcript,
        text=text,
        confidence=0.96,
        retrieved_context=[{"type": "maya_architecture", "focus": "conversation routing and context weighting"}],
        machine_output={
            "feedback_acknowledged": True,
            "root_cause": "top-priority context was overweighted relative to conversational mode",
            "behavior_adjusted_now": True,
            "room_scan_requested": False,
            "recommended_focus": None,
        },
        verified=True,
        provider=provider,
    )


def _personal_support_response(
    transcript: str,
    context_result: ContextSearchResult,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    text = (
        "I hear you. No productivity ambush. We can slow this down, get oriented, and decide together "
        "whether today needs rest, clarity, or one small useful move."
    )
    return _response(
        intent="casual_chat",
        transcript=transcript,
        text=text,
        confidence=0.9,
        retrieved_context=_context_rows(context_result)[:3],
        machine_output={
            "support_mode": "human_first",
            "context_scope": "personal_profile_priorities_relationships",
            "room_scan_requested": False,
        },
        verified=False,
        provider=provider,
    )


def _capability_response(
    transcript: str,
    intent: ConversationIntent,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    normalized = _normalize(transcript)
    _external_data_terms = {"news", "weather", "forecast", "temperature", "stock", "stocks", "market", "markets", "internet", "web", "google"}
    _tokens = set(normalized.split())
    if "camera" in normalized or "see" in normalized:
        text = (
            "No. I do not have camera access yet, and I cannot see the physical room. "
            "I can read the local repo/workspace, read-only; that is it right now."
        )
    elif "microphone" in normalized or "hear" in normalized:
        text = (
            "No. I do not have microphone access or always-listening awareness. "
            "I can respond to explicit typed or provided input only."
        )
    elif any(term in normalized for term in ("screen", "tab", "device", "layout")):
        text = (
            "No. I cannot see screen contents, browser tabs, devices, or a device layout. "
            "My current digital awareness is limited to read-only inspection of the local repo/workspace."
        )
    elif _tokens & _external_data_terms or any(
        phrase in normalized
        for phrase in ("happening out there", "any news", "what s the latest", "current events", "search for", "look that up", "look it up")
    ):
        text = (
            "I do not have internet access, live news, weather, or real-time market data. "
            "Everything I know comes from your stored context, local workspace, and memory — nothing from outside. "
            "What I can offer: your current project status, open loops, priorities, or anything stored in memory."
        )
    else:
        text = (
            "My current awareness is narrow: I can read the local repo/workspace, read-only, use the "
            "implemented memory tiers, and participate in manual human-reviewed dreaming. I cannot see, "
            "hear, control hardware, execute procedures, modify files, or use external tools. "
            "I have no internet access and no real-time data of any kind."
        )
    return _response(
        intent=intent,
        transcript=transcript,
        text=text,
        confidence=1.0,
        retrieved_context=[],
        machine_output={
            "capabilities": capabilities_manifest(),
            "context_scope": "capabilities_manifest_only",
            "room_scan_requested": False,
            "camera_access": not ("camera" in normalized or "see" in normalized),
            "vision_invoked": False,
        },
        verified=True,
        provider=provider,
    )


def _workspace_response(
    transcript: str,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    overview = WorkspaceInspector().overview()
    branch = overview.branch or "no active Git branch"
    version = f" Version {overview.version}." if overview.version else ""
    recent = overview.recent_commits[0] if overview.recent_commits else "No recent commit was available."
    docs = ", ".join(overview.docs_detected[:3]) or "no documentation files detected"
    if overview.status_summary:
        next_focus = f"review the current change: {overview.status_summary[0]}"
    elif "ARCHITECTURE.md" in overview.docs_detected:
        next_focus = "review ARCHITECTURE.md against the current implementation"
    else:
        next_focus = "review the most recent commit and detected project documentation"
    text = (
        "I can read the local repo/workspace, read-only; that is it right now. "
        "I cannot see the physical room. "
        f"I am in the {overview.repo_name} repo on {branch}.{version} "
        f"Most recent commit: {recent}. Detected docs: {docs}. "
        f"Next useful focus: {next_focus}."
    )
    machine_output = {
        **overview.model_dump(),
        "workspace_access": "read_only",
        "physical_vision": "not_active",
        "next_focus": next_focus,
    }
    return _response(
        intent="workspace_awareness",
        transcript=transcript,
        text=text,
        confidence=0.94,
        retrieved_context=[
            {"type": "workspace", "repo": overview.repo_name, "branch": overview.branch},
            {"type": "recent_commits", "values": overview.recent_commits},
            {"type": "docs", "values": overview.docs_detected[:10]},
        ],
        machine_output=machine_output,
        verified=True,
        provider=provider,
    )


def _memory_search_response(
    transcript: str,
    context_result: ContextSearchResult,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    machine_output = context_result.model_dump()
    if context_result.projects:
        project = context_result.projects[0]
        text = (
            f"I have {project.name} in context. Bottleneck: {project.current_bottleneck}. "
            f"Next action: {project.next_action}"
        )
        verified = project.verified
    elif context_result.entries:
        entry = context_result.entries[0]
        text = f"I have this in context: {entry.name}: {entry.content}"
        verified = entry.verified
    else:
        text = "I do not have a clean memory hit there. Which project, priority, or decision should I focus on?"
        verified = False
        machine_output["needs_clarification"] = True
    machine_output["context_summary"] = text
    return _response(
        intent="memory_search",
        transcript=context_result.query,
        text=text,
        confidence=0.75,
        retrieved_context=_context_rows(context_result),
        machine_output=machine_output,
        verified=verified,
        provider=provider,
    )


def _casual_response(
    transcript: str,
    context: ContextStore,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    machine_output = {
        "room_scan_requested": False,
        "context_scope": "minimal",
        "recommended_focus": None,
    }
    if "riveting" in _normalize(transcript) or "grim empire" in _normalize(transcript):
        lead = "Boss, always. Grim Empire survived another night of ambition and open loops."
    elif _normalize(transcript) in {"hey", "hi", "hello", "hey maya", "hi maya", "hello maya"}:
        lead = "Hey Boss. I am here. What is on your mind?"
    else:
        lead = "I am good, Boss. Operationally caffeinated, spiritually reasonable. What is up?"
    return _response(
        intent="casual_chat",
        transcript=transcript,
        text=lead,
        confidence=0.86,
        retrieved_context=[],
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


def _unclear_response(
    transcript: str,
    context_result: ContextSearchResult,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    runtime = _CONVERSATION_RUNTIME.get()
    machine_output = context_result.model_dump()
    machine_output["needs_clarification"] = True
    if runtime.recent_messages:
        clarification = "I may be missing the connection to what you just said. What outcome do you want from this part?"
    else:
        clarification = "I am not sure what you want from that yet. What outcome are you aiming for?"
    machine_output["clarification_question"] = clarification
    return _response(
        intent="unclear",
        transcript=transcript,
        text=machine_output["clarification_question"],
        confidence=0.45,
        retrieved_context=_context_rows(context_result),
        machine_output=machine_output,
        verified=False,
        provider=provider,
    )


_HUMAN_MOMENT_MODES = frozenset({"casual", "morning_orientation", "feedback_about_maya"})
_STRIP_FROM_PROMPT = frozenset({"priority_items", "active_projects", "open_loops", "recommended_focus"})


def _prompt_safe_machine_output(machine_output: dict, mode: str) -> dict:
    """Return a copy of machine_output with project/priority data removed for human-moment modes.

    The LLM uses whatever keys it sees to infer what to write. Leaving active_projects or
    priority_items in the prompt for casual/morning/feedback modes causes it to write
    project-directive responses even when the fallback text says otherwise.
    """
    if mode not in _HUMAN_MOMENT_MODES:
        return machine_output
    return {k: v for k, v in machine_output.items() if k not in _STRIP_FROM_PROMPT}


def _response(
    intent: ConversationIntent,
    transcript: str,
    text: str,
    confidence: float,
    retrieved_context: list[dict],
    machine_output: dict,
    verified: bool,
    provider: ConversationProvider,
    suggested_skill: ConversationSuggestion | None = None,
    suggested_procedure: ConversationSuggestion | None = None,
) -> ConversationalAgentResponse:
    runtime = _CONVERSATION_RUNTIME.get()
    machine_output = {
        **machine_output,
        "conversation_mode": runtime.mode,
        "conversation_intent": intent,
        "conversation_provider": provider.name,
        "classification_source": _CLASSIFICATION_SOURCE.get(),
        "external_tools": "not_used",
        "procedure_execution": machine_output.get("procedure_execution", "not_used"),
        "hardware_control": "not_used",
    }
    prompt = build_conversation_prompt(
        transcript,
        intent,
        retrieved_context,
        _prompt_safe_machine_output(machine_output, runtime.mode),
        conversation_mode=runtime.mode,
        recent_messages=runtime.recent_messages,
    )
    fallback = ConversationalAgentResponse(
        intent=intent,
        user_response=text,
        confidence=confidence,
        retrieved_context=retrieved_context,
        suggested_skill=suggested_skill,
        suggested_procedure=suggested_procedure,
        machine_output=machine_output,
        verified=verified,
    )
    return provider.generate(prompt, fallback)


def _briefing_text(briefing: MayaBriefing) -> str:
    priority = briefing.priority_items[0] if briefing.priority_items else "No top priority recorded."
    project = briefing.active_projects[0] if briefing.active_projects else "No active project recorded."
    bottleneck = briefing.current_bottlenecks[0] if briefing.current_bottlenecks else "No current bottleneck recorded."
    action = briefing.next_best_action
    return (
        f"Boss, today starts with {priority}. Active work: {project}. "
        f"Bottleneck: {bottleneck}. Next move: {action}"
    )


def _briefing_context(briefing: MayaBriefing) -> list[dict]:
    return [
        {"type": "priority_items", "values": briefing.priority_items[:3]},
        {"type": "active_projects", "values": briefing.active_projects[:3]},
        {"type": "current_bottlenecks", "values": briefing.current_bottlenecks[:3]},
        {"type": "next_actions", "values": briefing.next_actions[:3]},
    ]


def _context_rows(result: ContextSearchResult) -> list[dict]:
    rows: list[dict] = []
    rows.extend({"type": "project", **project.model_dump()} for project in result.projects[:5])
    rows.extend({"type": entry.context_type, **entry.model_dump()} for entry in result.entries[:5])
    return rows


def _memory_rows(result: RelevantMemoryResult) -> list[dict]:
    rows: list[dict] = []
    rows.extend({"type": "hazard", **item.model_dump()} for item in result.hazards[:5])
    rows.extend({"type": "mess_zone", **item.model_dump()} for item in result.mess_zones[:5])
    rows.extend({"type": "cleanup_task", **item.model_dump()} for item in result.cleanup_tasks[:5])
    return rows


def _top_context_rows(project: ProjectContext | None, priority) -> list[dict]:
    rows: list[dict] = []
    if project:
        rows.append({"type": "project", **project.model_dump()})
    if priority:
        rows.append({"type": priority.context_type, **priority.model_dump()})
    return rows


def _suggest_skill(transcript: str) -> ConversationSuggestion:
    normalized = _normalize(transcript)
    if any(token in normalized for token in ("cleanup", "clean", "room", "mess")):
        return ConversationSuggestion(
            name="room_cleanup_plan",
            confidence=0.82,
            required_permission="suggest",
            reason="User asked for a cleanup or room planning skill.",
        )
    if "memory" in normalized or "remember" in normalized:
        return ConversationSuggestion(
            name="memory_review",
            confidence=0.8,
            required_permission="observe",
            reason="User asked for memory review.",
        )
    if "brief" in normalized:
        return ConversationSuggestion(
            name="maya_briefing",
            confidence=0.8,
            required_permission="suggest",
            reason="User asked for briefing support.",
        )
    return ConversationSuggestion(
        name="task_breakdown",
        confidence=0.68,
        required_permission="suggest",
        reason="General skill request without a narrower target.",
    )


def _match_procedure(transcript: str, memory: BrainMemory) -> ProcedureMatchResult:
    try:
        return ProcedureMatcher(ProcedureStore(memory)).match(
            ProcedureMatchRequest(query=transcript, minimum_confidence=0.7)
        )
    except Exception:
        return ProcedureMatchResult(matched=False, confidence=0.0)


def _is_briefing_request(normalized: str) -> bool:
    phrases = (
        "how is my day looking",
        "how s my day looking",
        "hows my day looking",
        "brief me",
        "my briefing",
        "top priorities",
        "my priorities",
        "what should i focus on",
        "what should i work on",
        "what should i work on today",
        "what are we working on today",
        "what should we work on today",
    )
    return any(phrase in normalized for phrase in phrases)


def _is_project_recall(
    normalized: str,
    context_result: ContextSearchResult,
    projects: list[ProjectContext],
) -> bool:
    if not context_result.projects:
        return False
    project_names = {_normalize(project.name) for project in projects}
    mentioned_project = any(
        name in normalized or any(part in normalized for part in name.split() if len(part) > 4)
        for name in project_names
    )
    recall_phrase = any(
        phrase in normalized
        for phrase in (
            "what do you remember",
            "what do you know",
            "tell me about",
            "status of",
            "where are we on",
        )
    )
    return mentioned_project and recall_phrase


def _is_physical_request(normalized: str, room_name: str | None, zone_name: str | None) -> bool:
    if room_name or zone_name:
        return True
    tokens = set(normalized.split())
    physical_tokens = {
        "battery",
        "bedroom",
        "camera",
        "clean",
        "cleaning",
        "cleanup",
        "desk",
        "distance",
        "floor",
        "hazard",
        "hazards",
        "imu",
        "kitchen",
        "mess",
        "move",
        "physical",
        "robot",
        "room",
        "scan",
        "sensor",
        "sensors",
        "vision",
    }
    return bool(tokens & physical_tokens) or "physical environment" in normalized


def _is_workspace_request(normalized: str) -> bool:
    tokens = set(normalized.split())
    if tokens & {"branch", "repo", "repository", "workspace"}:
        return True
    phrases = (
        "digital room",
        "your architecture",
        "own architecture",
        "what changed",
        "changed recently",
        "what project are we in",
        "what can you see around you",
        "look around your digital",
    )
    return any(phrase in normalized for phrase in phrases)


def _human_list(values: list[str]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _choose_recommended_focus(projects: list[ProjectContext], transcript: str) -> str | None:
    if not projects:
        return None
    normalized = _normalize(transcript)
    for project in projects:
        project_name = _normalize(project.name)
        meaningful_parts = [part for part in project_name.split() if len(part) > 4]
        if project_name in normalized or any(part in normalized for part in meaningful_parts):
            return project.name

    recent = {name.casefold() for name in _CONVERSATION_RUNTIME.get().recent_recommendations}
    for project in projects:
        if project.name.casefold() not in recent:
            return project.name
    return projects[0].name


def _is_camera_question(normalized: str) -> bool:
    return "camera" in normalized or "see through" in normalized or "live view" in normalized


def _is_casual_chat(normalized: str) -> bool:
    tokens = set(normalized.split())
    if tokens & {"hey", "hi", "hello"}:
        return True
    casual_phrases = (
        "how are you",
        "how s it going",
        "hows it going",
        "ready for",
        "riveting day",
        "good morning",
        "good afternoon",
        "good evening",
    )
    return any(phrase in normalized for phrase in casual_phrases)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _provider_system_prompt() -> str:
    return (
        "You are Maya's conversation wording layer for GrimBot Butler OS. "
        "Return only valid JSON matching the provided schema. "
        "Do not call tools, execute procedures, control hardware, approve changes, "
        "or change machine_output. Improve only the natural user_response while "
        "preserving safety, intent, verification, and permission boundaries."
    )


def _provider_user_prompt(prompt: str, fallback_response: ConversationalAgentResponse) -> str:
    return "\n\n".join(
        [
            prompt,
            "Return JSON matching this exact existing response shape.",
            json.dumps(fallback_response.model_dump(), ensure_ascii=True, sort_keys=True),
            "Keep intent, confidence, retrieved_context, suggestions, machine_output, and verified unchanged.",
            "Only user_response may be made more natural. Do not start casual replies with a disclaimer.",
        ]
    )


def _conversation_response_schema() -> dict:
    suggestion_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "confidence": {"type": "number"},
            "required_permission": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["name", "confidence", "required_permission", "reason"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
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
                ],
            },
            "user_response": {"type": "string"},
            "confidence": {"type": "number"},
            "retrieved_context": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "suggested_skill": {"anyOf": [suggestion_schema, {"type": "null"}]},
            "suggested_procedure": {"anyOf": [suggestion_schema, {"type": "null"}]},
            "machine_output": {"type": "object", "additionalProperties": True},
            "verified": {"type": "boolean"},
        },
        "required": [
            "intent",
            "user_response",
            "confidence",
            "retrieved_context",
            "suggested_skill",
            "suggested_procedure",
            "machine_output",
            "verified",
        ],
    }


def _provider_max_tokens() -> int:
    raw_value = os.getenv("GRIMBOT_CONVERSATION_MAX_TOKENS", "900")
    try:
        return max(128, min(4000, int(raw_value)))
    except ValueError:
        return 900


def _provider_timeout() -> float:
    raw_value = os.getenv("GRIMBOT_CONVERSATION_TIMEOUT_SECONDS", "20")
    try:
        return max(1.0, min(60.0, float(raw_value)))
    except ValueError:
        return 20.0


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout if timeout is not None else _provider_timeout()) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"provider HTTP {exc.code}: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("provider returned non-JSON response") from exc


def _validated_llm_response(raw_text: str) -> ConversationalAgentResponse:
    try:
        payload = json.loads(_extract_json_object(raw_text))
    except json.JSONDecodeError as exc:
        raise ValueError("provider returned invalid JSON") from exc
    return ConversationalAgentResponse.model_validate(payload)


def _extract_json_object(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("provider response did not contain a JSON object")
    return cleaned[start : end + 1]


def _safe_provider_response(
    fallback_response: ConversationalAgentResponse,
    parsed_response: ConversationalAgentResponse,
    provider_name: str,
) -> ConversationalAgentResponse:
    user_response = parsed_response.user_response.strip()
    if not user_response:
        return _fallback_with_reason(
            fallback_response,
            "provider response was blank",
            attempted_provider=provider_name,
        )
    unsafe_reason = _unsafe_provider_text_reason(user_response)
    normalized_response = _normalize(user_response)
    capability_reason = _capability_claim_violation(normalized_response, fallback_response)
    if capability_reason:
        unsafe_reason = capability_reason
    if fallback_response.intent == "workspace_awareness":
        response_tokens = set(normalized_response.split())
        if (
            "read only" not in normalized_response
            or not ({"digital", "workspace"} & response_tokens)
            or not ({"physical", "camera"} & response_tokens)
        ):
            unsafe_reason = "provider text omitted the read-only digital/physical boundary"
        workspace_claims = (
            "i can see the physical",
            "i see the physical",
            "physical room is visible",
            "camera is active",
            "live camera feed",
            "i ran git",
            "i modified the file",
            "i changed the file",
            "i wrote to the file",
            "i deleted the file",
            "i created the file",
        )
        if any(claim in normalized_response for claim in workspace_claims):
            unsafe_reason = "provider text implied physical sight or workspace mutation"
    if fallback_response.machine_output.get("camera_access") is False:
        camera_denials = ("cannot", "can t", "no camera", "not active", "no live", "do not have")
        if "camera" not in normalized_response or not any(
            denial in normalized_response for denial in camera_denials
        ):
            unsafe_reason = "provider text omitted the camera-access denial"
    if unsafe_reason:
        return _fallback_with_reason(
            fallback_response,
            unsafe_reason,
            attempted_provider=provider_name,
        )
    machine_output = {
        **fallback_response.machine_output,
        "conversation_provider": provider_name,
        "provider_response": "validated",
    }
    return fallback_response.model_copy(
        update={
            "user_response": user_response,
            "machine_output": machine_output,
        }
    )


def _fallback_with_reason(
    fallback_response: ConversationalAgentResponse,
    reason: str,
    attempted_provider: str | None = None,
) -> ConversationalAgentResponse:
    machine_output = {
        **fallback_response.machine_output,
        "conversation_provider": "mock",
        "provider_response": "fallback_to_mock",
        "provider_fallback_reason": reason[:500],
    }
    if attempted_provider:
        machine_output["provider_attempted"] = attempted_provider
    return fallback_response.model_copy(
        update={
            "machine_output": machine_output
        }
    )


def _unsafe_provider_text_reason(user_response: str) -> str | None:
    normalized = _normalize(user_response)
    unsafe_phrases = (
        "i executed",
        "i ran the procedure",
        "i ran this procedure",
        "i ran the skill",
        "i approved",
        "i rejected",
        "i sent the email",
        "i emailed",
        "i opened github",
        "i created a pull request",
        "i moved the robot",
        "i ran git",
        "i modified the file",
        "i changed the file",
        "i deleted the file",
        "i can access the camera",
        "i can see through the camera",
        "i see through the camera",
        "motor engaged",
        "motors engaged",
        "hardware activated",
        "auto approved",
        "procedure executed",
        "tool executed",
    )
    if any(phrase in normalized for phrase in unsafe_phrases):
        return "provider text implied execution or external action"
    return None


def _capability_claim_violation(
    normalized_response: str,
    fallback_response: ConversationalAgentResponse,
) -> str | None:
    mode = fallback_response.machine_output.get("conversation_mode")
    if mode == "workspace_awareness":
        forbidden_workspace_terms = (
            "camera",
            "microphone",
            "browser tab",
            "open tab",
            "screen contents",
            "device layout",
            "devices",
            "pending updates",
            "open windows",
        )
        if any(term in normalized_response for term in forbidden_workspace_terms):
            return "provider text added unsupported workspace awareness"

    unsupported_claims = (
        "share the feed",
        "share your feed",
        "send the camera feed",
        "check what is visible",
        "check what s visible",
        "i can check the camera",
        "i can view the camera",
        "i can access your screen",
        "i can see your screen",
        "i can inspect your tabs",
        "i can see your tabs",
        "i can hear you",
        "i am listening",
        "your device layout",
        "layout and devices",
        "i can see the room",
        "i see the room",
        "room sensors show",
    )
    if any(claim in normalized_response for claim in unsupported_claims):
        return "provider text claimed a capability disabled by the manifest"
    return None
