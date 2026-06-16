# Conversational Maya Agent Design

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

The default provider is `mock`, which is deterministic and requires no API key. `GRIMBOT_CONVERSATION_PROVIDER` may be set to `gemini`, `openai`, or `claude`; those hooks are intentionally inert unless future provider clients are added. Conversation providers are separate from vision and dreaming providers.

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

## Safety Boundaries

The agent may suggest skills, procedure matches, dream review, memory search, and next actions. It may not:

- execute procedures
- run external tools
- control motors or hardware
- approve dream facts or pending procedures
- bypass skill permissions
- override `safety.py`

Verification language is contextual. Maya does not start every reply with a disclaimer. If a caller requests verified output and the source is not verified, Maya states that naturally without presenting the information as verified.
