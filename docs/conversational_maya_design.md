# Conversational Maya Agent Design

## v0.10.5 Capability Manifest and Honesty Layer

Maya's capabilities are an application-owned contract, never an LLM inference. `grimbot_brain.capabilities.CAPABILITIES` records which forms of awareness are active. The current release permits bounded, read-only local repository/workspace inspection and the implemented memory tiers. It explicitly denies camera, microphone, screen/tab, device-layout, robot-body, physical-room sensor, workspace-write, procedure-execution, and external-tool access.

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

For real providers, the model is asked to return the same structured JSON shape. The response is validated with the existing Pydantic schema. After validation, only `user_response` is accepted from the provider; intent, confidence, retrieved context, suggestions, machine output, and verification state remain controlled by GrimBot.

## Safety Boundaries

The agent may suggest skills, procedure matches, dream review, memory search, and next actions. It may not:

- execute procedures
- run external tools
- control motors or hardware
- approve dream facts or pending procedures
- bypass skill permissions
- override `safety.py`

Verification language is contextual. Maya does not start every reply with a disclaimer. If a caller requests verified output and the source is not verified, Maya states that naturally without presenting the information as verified.
