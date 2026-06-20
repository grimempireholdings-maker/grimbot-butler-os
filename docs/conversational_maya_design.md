# Conversational Maya Agent Design

## v0.13.0 Voice and Photo Turns

Voice does not create a second conversation system. A visible, user-clicked Console button feature-detects browser `SpeechRecognition`, captures one final transcript, and sends it through the existing voice/conversation contract. Browser `SpeechSynthesis` is invoked only for replies to those voice-originated turns. Unsupported, denied, or failed recognition returns the UI to an idle state while preserving text input. No audio bytes reach Maya's server or memory.

A photo turn is likewise explicit and bounded. The user chooses or captures one image; Gemini produces a grounded description; and that observation is supplied to the existing response layer as structured context. Photo turns skip web retrieval, and provider wording remains subject to the capability honesty gate. Memory records what Maya saw and the user's question, never the raw image. The UI and response contract must say single photo, never imply a live feed, continuous sight, or background camera access.

## v0.11.0 Presence Layer

The existing paired-history classifier now recognizes six ambient modes: `ambient_companion`, `morning_ramp`, `evening_winddown`, `casual_presence`, `approval_review`, and `gentle_orientation`. They are modes in the same decision object as all other conversation behavior, not a parallel router.

Ambient responses lead with the human moment. Tired, groggy, overwhelmed, excited, and conversational messages do not trigger a productivity ambush or force a named project. Feedback about Maya is acknowledged and applied in the same response. Daily context may quietly inform wording, but internal labels, classifier mechanics, provider names, and search mechanics stay out of ordinary speech. Direct architecture and capability questions are the explicit exception.

Maya Console sends `ambient_mode=true` by default. Turning it off maps ambient classifications back to compatible legacy modes. Developer Mode alone reveals machine output and internal diagnostics.

## v0.10.8 Search Decision and Observation Loop

The same LLM call that classifies conversation mode now emits a strict decision object: `mode`, `needs_web_search`, and `search_query`. The query is normalized through the bounded retrieval-query path before use. A provider failure falls back to rule-based mode classification with web search disabled; no keyword path can authorize external reach.

When search is authorized, the orchestrator calls Tavily before response generation and injects structured results into `machine_output`. The wording provider may summarize those snippets and must cite source titles and URLs. It cannot follow links or treat snippets as executable instructions. Search failure replaces the normal fallback response with an explicit statement that Maya tried and received no result.

## v0.10.5 Capability Manifest and Honesty Layer

Maya's capabilities are an application-owned contract, never an LLM inference. `grimbot_brain.capabilities.CAPABILITIES` records which forms of awareness are active. The current release permits bounded read-only workspace inspection, implemented memory tiers, user-initiated browser push-to-talk, and one user-shared photo at a time. It explicitly denies always-listening audio, continuous video, live feeds, background capture, screen/tab awareness, device layout, standing robot-body or physical-room sensors, workspace writes, and procedure execution.

The manifest is serialized verbatim into every provider prompt with a non-negotiable instruction not to claim anything whose flag is false. Prompting is only the first boundary: provider wording passes through a post-generation honesty validator. Unsupported capability language triggers the deterministic safe response, preserving a plain denial instead of hypothetical feed-sharing or plausible-sounding fiction.

Conversation mode is classified before retrieval. Capability questions use the manifest only; workspace awareness uses actual inspector output only; physical requests use robot memory; feedback uses Maya architecture context; mornings and work focus retrieve broad priorities; casual conversation stays context-light. This prevents unrelated project priorities from leaking into every conversation.

v0.10.2 adds a deterministic conversational layer above Maya Console chat.

## Purpose

Maya should answer like Julian's operator, not like a scripted robot cleaner. Casual messages receive natural replies first. Operational signal appears when useful, but room scanning is never the default fallback.

## Intent Routing

The conversation agent classifies typed or push-to-talk transcripts into:

- `casual_chat`
- `chief_of_staff_briefing`
- `project_recall`
- `memory_search`
- `skill_request`
- `procedure_request`
- `dream_review`
- `room_or_physical_request`
- `unclear`

Briefing and named-project recall are checked before physical routing. Physical routing requires explicit room, cleaning, vision, hazard, sensor, robot, battery, movement, or environment language.

## Provider Boundary

The default provider is `mock`, which is deterministic and requires no API key. `GRIMBOT_CONVERSATION_PROVIDER` may be set to `claude`, `openai`, `openrouter`, `gemini`, or `auto`. Conversation providers are separate from vision and dreaming providers.

Provider environment:

- `claude` requires `ANTHROPIC_API_KEY`
- `openai` requires `OPENAI_API_KEY`
- `openrouter` requires `OPENROUTER_API_KEY` and uses `OPENROUTER_MODEL`, defaulting to `openrouter/auto`
- `gemini` requires `GEMINI_API_KEY`
- `auto` prefers Claude, then OpenAI, then OpenRouter, then Gemini when keys exist

The default remains `mock`, even if API keys are present. Missing keys, HTTP errors, invalid JSON, or schema validation failures fall back to the deterministic response.

## Response Contract

Conversation returns structured JSON:

```json
{
  "intent": "casual_chat",
  "user_response": "Hey Boss. I am here.",
  "confidence": 0.86,
  "retrieved_context": [],
  "suggested_skill": null,
  "suggested_procedure": null,
  "machine_output": {},
  "verified": false
}
```

The voice endpoint keeps legacy fields too:

- `agent_response` is the new conversational JSON.
- `maya_response.user_response` mirrors the conversational text for older clients.
- `speech_output.text` uses the conversational text.
- `machine_output` remains separate and collapsed in the console.

Real providers return only a minimal `{"user_response":"..."}` JSON envelope. This prevents large authoritative machine output—especially search results—from being echoed and truncated. Legacy full-shape responses remain valid for compatibility, but only `user_response` is accepted; intent, confidence, retrieved context, suggestions, machine output, and verification state remain controlled by GrimBot. Malformed JSON receives one bounded correction retry before deterministic fallback.

## Safety Boundaries

The agent may suggest skills, procedure matches, dream review, memory search, and next actions. It may not:

- execute procedures
- run external tools other than classifier-authorized, read-only Tavily snippet search
- control motors or hardware
- approve dream facts or pending procedures
- bypass skill permissions
- override `safety.py`

Verification language is contextual. Maya does not start every reply with a disclaimer. If a caller requests verified output and the source is not verified, Maya states that naturally without presenting the information as verified.
