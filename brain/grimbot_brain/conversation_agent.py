from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from .conversation_schemas import (
    ConversationIntent,
    ConversationSuggestion,
    ConversationalAgentResponse,
)
from .identity.context_schemas import ContextSearchRequest, ContextSearchResult, ProjectContext
from .identity.context_store import ContextStore
from .maya_core import build_maya_briefing
from .memory import BrainMemory
from .procedural_memory.procedure_matcher import ProcedureMatcher
from .procedural_memory.procedure_schemas import ProcedureMatchRequest, ProcedureMatchResult
from .procedural_memory.procedure_store import ProcedureStore
from .robot_memory import RobotMemory
from .schemas import (
    MayaBriefing,
    MayaBriefingRequest,
    RelevantMemoryRequest,
    RelevantMemoryResult,
    VoiceConversationRequest,
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

    def _model(self) -> str:
        return os.getenv(self.model_env_key, self.default_model).strip() or self.default_model


class ClaudeConversationProvider(ApiConversationProvider):
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


def run_conversation_agent(
    request: VoiceConversationRequest,
    transcript: str,
    memory: BrainMemory,
    memory_context: RelevantMemoryResult | None = None,
    provider: ConversationProvider | None = None,
) -> ConversationalAgentResponse:
    robot_memory = RobotMemory(memory)
    context = ContextStore(memory)
    memory_context = memory_context or robot_memory.relevant(
        RelevantMemoryRequest(
            query=transcript,
            room_name=request.room_name,
            zone_name=request.zone_name,
            limit=10,
        )
    )
    context_result = context.search(ContextSearchRequest(query=transcript, limit=10))
    intent = classify_intent(transcript, request, context_result, context.projects())
    provider = provider or provider_from_env()

    if intent == "chief_of_staff_briefing":
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
    elif intent == "memory_search":
        agent_response = _memory_search_response(transcript, context_result, provider)
    elif intent == "casual_chat":
        agent_response = _casual_response(transcript, context, provider)
    else:
        agent_response = _unclear_response(transcript, context_result, provider)

    return agent_response


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
) -> str:
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
            f"Intent: {intent}",
            f"User message: {transcript}",
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
    projects = context.projects()
    priorities = context.priorities()
    top_project = projects[0] if projects else None
    top_priority = priorities[0] if priorities else None
    machine_output = {
        "conversation_mode": "casual",
        "room_scan_requested": False,
        "priority": top_priority.model_dump() if top_priority else None,
        "project": top_project.model_dump() if top_project else None,
    }
    if "riveting" in _normalize(transcript) or "grim empire" in _normalize(transcript):
        lead = "Boss, always. Grim Empire survived another night of ambition and open loops."
    elif _normalize(transcript) in {"hey", "hi", "hello", "hey maya", "hi maya", "hello maya"}:
        lead = "Hey Boss. I am here."
    else:
        lead = "I am good, Boss. Operationally caffeinated, spiritually reasonable."

    if top_project:
        text = f"{lead} If we are working, I would start with {top_project.name}: {top_project.next_action}"
    elif top_priority:
        text = f"{lead} The first useful thread is {top_priority.content}"
    else:
        text = f"{lead} Tell me which lane you want to hit first."
    return _response(
        intent="casual_chat",
        transcript=transcript,
        text=text,
        confidence=0.86,
        retrieved_context=_top_context_rows(top_project, top_priority),
        machine_output=machine_output,
        verified=bool((top_project and top_project.verified) or (top_priority and top_priority.verified)),
        provider=provider,
    )


def _unclear_response(
    transcript: str,
    context_result: ContextSearchResult,
    provider: ConversationProvider,
) -> ConversationalAgentResponse:
    machine_output = context_result.model_dump()
    machine_output["needs_clarification"] = True
    machine_output["clarification_question"] = (
        "Which project or lane do you mean: strategy, memory, skills, procedures, dreams, or the physical room?"
    )
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
    machine_output = {
        **machine_output,
        "conversation_intent": intent,
        "conversation_provider": provider.name,
        "external_tools": "not_used",
        "procedure_execution": machine_output.get("procedure_execution", "not_used"),
        "hardware_control": "not_used",
    }
    prompt = build_conversation_prompt(transcript, intent, retrieved_context, machine_output)
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
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
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
                    "room_or_physical_request",
                    "unclear",
                ],
            },
            "user_response": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
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


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
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
        with urllib.request.urlopen(request, timeout=_provider_timeout()) as response:
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
        "motors engaged",
        "hardware activated",
        "auto approved",
        "procedure executed",
        "tool executed",
    )
    if any(phrase in normalized for phrase in unsafe_phrases):
        return "provider text implied execution or external action"
    return None
