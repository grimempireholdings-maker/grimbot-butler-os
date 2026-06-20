# GrimBot Butler OS

An open, modular AI-powered robotic butler platform.

## Mission

Create a safe, affordable personal robotic assistant capable of perception, memory, planning, conversation, automation, and eventually physical household assistance.

## Roadmap

- [x] Brain Simulation
- [x] Vision
- [x] Long-Term Memory
- [x] Maya Core
- [x] Push-to-Talk Voice
- [x] Safe Skills Registry
- [x] Adaptive State Engine
- [x] Dreaming Foundation
- [x] Procedural Memory Foundation
- [x] Maya Console
- [x] Chief of Staff Context
- [x] Conversational Maya Agent
- [x] Read-Only Workspace Awareness
- [x] Classifier-Authorized Web Search
- [x] Ambient Companion Mode
- [x] Three-Mode Maya Console
- [x] Real Browser Voice and Single-Photo Vision
- [ ] External Tool Use
- [ ] Rover Platform
- [ ] Object Manipulation
- [ ] Household Assistance
- [ ] Humanoid Platform

## Current Status

Phase 0 includes a runnable Python/FastAPI robot brain simulator. It accepts sensor input, runs mock or Gemini-backed perception, creates planner intent, validates all movement through a safety layer, and logs cycles to SQLite.

Phase 1 adds GrimBot Vision v0.2: safe local webcam frame capture, approved-image validation, and Gemini or mock room scans that return structured JSON.

Phase 2 adds GrimBot Robot Memory v0.3: structured SQLite memory for rooms, zones, known objects, hazards, mess observations, cleanup tasks, episodic memories, and semantic facts.

Phase 3 adds GrimBot Maya Core v0.4: assistant modes, permission logic, Maya-style response composition, cleanup coaching, and structured briefings.

Phase 4 adds GrimBot Conversational Voice v0.5: push-to-talk speech-to-text, Maya response composition, robot memory retrieval, and mock text-to-speech output.

Phase 5 adds GrimBot Skills Registry v0.6: safe internal butler skills with permission gates, Maya responses, and memory-backed planning.

Phase 6 adds GrimBot Adaptive State v0.7: SQLite-backed state signals that influence attention, memory priority, skill suggestion, and Maya response style without adding emotions, consciousness, ML training, motors, or autonomous execution.

Phase 7 adds GrimBot Dreaming Foundation v0.8: manual rule-based reflection, protected forgetting, candidate semantic facts, human promotion review, and auditable dream-cycle logs.

Phase 8 adds GrimBot Procedural Memory v0.9: strict procedure schemas, immutable version history, pending proposal review, passive execution statistics, and deterministic matching without procedure execution.

Phase 9 adds Maya Console v0.10: a local operator interface for conversation, briefings, adaptive state, skills, dreaming review, procedural memory review, and robot memory inspection.

Phase 10 adds Maya Chief of Staff Context v0.10.1: structured personal and business context for Julian, his ventures, active projects, priorities, relationships, bottlenecks, protocols, and next actions.

Phase 10.2 adds Conversational Maya Agent v0.10.2: deterministic intent routing, natural Maya responses, provider hooks, and console chat integration without defaulting to room scans.

Phase 10.4 adds Maya Workspace Awareness: bounded read-only inspection of the active repository, branch, status, recent commits, documentation, version, and safe project text search. Version `0.10.3` remains the existing OpenRouter provider release, so workspace awareness advances to `0.10.4` rather than reusing that tag.

Phase 10.5 adds capability honesty and conversation modes. A hardcoded capability manifest is included in every provider prompt, unsupported awareness claims are rejected after generation, and casual, morning, feedback, work-focus, workspace, physical, and capability conversations retrieve only mode-appropriate context. Maya no longer treats Real Estate Acquisitions as the universal fallback.

Phase 10.8 adds Maya's first bounded external-world capability: classifier-authorized Tavily web search. Search is read-only snippet retrieval with a five-second timeout, one-hour cache, episodic usage logging, structured machine output, and honest failure behavior. It does not browse pages, scrape arbitrary URLs, follow links, or execute result content.

Phase 11 adds Ambient Companion Mode v0.11.0. Six new modes flow through the existing paired-history LLM classifier, daily orientation assembles read-only context behind the scenes, and ordinary wording is protected from internal/debug vocabulary. Ambient Mode is on by default in Maya Console; Developer Mode remains the explicit place for architecture and search diagnostics. A morning greeting may perform one cached weather lookup. This is the only autonomous, non-question-triggered search; proactive news is not enabled.

Phase 12 adds the Three-Mode Maya Console v0.12.0. Conversation is the default and contains only live status tokens, chat, Ambient Mode, persona, and send controls. Briefing is generated only after an explicit request. Developer Mode dynamically mounts the full context, workspace, state, skill, dreaming, procedure, memory, and conversation-diagnostic panels, then removes them from the live DOM when disabled. The console remains a local FastAPI-served static application with no frontend build step.

Phase 13 adds real, user-initiated sensory input in v0.13.0. Maya Console push-to-talk uses the browser Web Speech API and speaks voice-originated replies with SpeechSynthesis. A mobile capture control submits exactly one selected photo to Gemini 2.5 Flash Lite for a conversational response. Neither capability is ambient: there is no background audio, live camera feed, continuous video, or persistent raw-media library.

Maya Console is designed for phone and desktop use over a trusted local network or private Tailscale connection. It is not deployed publicly and should not be exposed directly to the public internet.

LLM output is never connected directly to motors. Every movement command must pass through `brain/grimbot_brain/safety.py`.

By default, local cycle logs are stored at `memory/grimbot_brain.sqlite3`.
Approved room-scan images are stored under `vision/images` by default. User photo uploads are validated, analyzed from a temporary file, and deleted in all outcomes; episodic memory retains only the description and context.

## Setup

Create a free Tavily API key at [tavily.com](https://tavily.com), copy `.env.example` to `.env`, and set:

```env
TAVILY_API_KEY=your_key_here
GRIMBOT_WEATHER_LOCATION=Dayton, Ohio
```

Without this key, classifier-authorized searches fail closed and Maya states that live search did not return rather than fabricating current information.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

To enable direct Gemini perception and single-photo analysis:

```powershell
pip install -e ".[gemini,dev]"
copy .env.example .env
```

Then set `GEMINI_API_KEY` and `GRIMBOT_MOCK_PERCEPTION=false` in `.env`. Single-photo analysis can instead reuse an existing `OPENROUTER_API_KEY`; it remains pinned to the real `google/gemini-2.5-flash-lite` model through `OPENROUTER_VISION_MODEL` and never falls back to a mock description.

To enable real webcam capture:

```powershell
pip install -e ".[vision]"
```

The CLI voice path remains mockable. Maya Console uses browser-native push-to-talk and speech synthesis with no cloud voice key.

## Run Brain Server

```powershell
uvicorn grimbot_brain.main:app --reload
```

Local API docs:

```text
http://127.0.0.1:8000/docs
```

## Run Maya Console

Start the existing brain server:

```powershell
uvicorn grimbot_brain.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/console
```

The console initially loads Conversation Mode only. Briefing is explicit, and Chief of Staff context, workspace awareness, adaptive state, skills, dreaming, procedural memory, robot memory, and diagnostics are absent from the live DOM until Developer Mode is enabled. Chat, photo capture, briefings, skill runs, dream cycles, review decisions, procedure matching, workspace search, and memory recall occur only after an explicit operator action.

The console does not add autonomous execution, procedure execution, motors, hardware control, or automatic approval. Conversation may perform only classifier-authorized, read-only Tavily snippet search; skill permissions and all other safety boundaries remain authoritative.

Console chat uses the v0.10.2 conversational Maya agent. Casual chat stays conversational, day/planning questions route to Chief of Staff briefing, named-project questions use context recall, and room scanning appears only for explicit physical, cleaning, vision, hazard, sensor, or robot requests.

## Workspace Awareness

Maya can inspect the local project without modifying it:

```text
GET /workspace
GET /workspace/docs
POST /workspace/search
```

Workspace inspection reports the current repository root and name, branch, short status, last five commits, detected version, safe top-level items, documentation files, and warnings. Search is literal text matching across a bounded set of small text files. It skips `.git`, virtual environments, `node_modules`, caches, SQLite databases, `.env` files, binary files, large files, secret-looking filenames, and symlink escapes. Git inspection disables optional locks, and likely API-key patterns are redacted from returned text.

Only these Git commands are available to workspace inspection, always with `shell=False` and a timeout:

```text
git rev-parse --show-toplevel
git branch --show-current
git status --short
git log --oneline -5
```

Workspace awareness is digital, local, and read-only. It remains separate from web search and is not physical camera vision, command execution, or file editing.

## Run CLI Demo

```powershell
python -m grimbot_brain.cli_demo
```

The demo simulates 10 robot brain cycles and prints strict JSON commands only.

## Run Room Scan

Mock room scan, no webcam or API key required:

```powershell
python -m grimbot_brain.room_scan_cli --mock-camera-frame "laundry, dishes, and a loose cord"
```

Webcam room scan:

```powershell
python -m grimbot_brain.room_scan_cli --capture-webcam
```

Room scan output is structured JSON with:

```json
{
  "room_summary": "Mock room scan completed with simulated visual context.",
  "visible_objects": ["floor", "table", "chair"],
  "mess_zones": ["general surfaces"],
  "hazards": [],
  "suggested_cleanup_order": ["general surfaces"],
  "next_best_action": "general surfaces",
  "mode": "mock",
  "image_path": null
}
```

## Robot Memory

Room scans automatically update structured robot memory:

- visible objects become known objects
- mess zones become recurring mess observations
- hazards update hazard counts and confidence
- suggested cleanup order becomes cleanup task memory

Memory endpoints:

```text
POST /memory/remember
GET /memory/rooms
GET /memory/rooms/{room_name}
GET /memory/hazards
GET /memory/mess-zones
POST /memory/relevant
```

Example recall shape:

```json
{
  "query": "what should I clean first?",
  "room_name": "Office",
  "hazards": [{"name": "loose cord on floor", "count": 2}],
  "mess_zones": [{"name": "notebooks on desk"}],
  "cleanup_tasks": [{"name": "loose cord on floor"}],
  "semantic_facts": [],
  "next_best_action": "clear hazard: loose cord on floor"
}
```

Memory can inform planning context later, but it never overrides `safety.py`.

## Maya Core

Maya Core keeps machine commands separate from user-facing assistant responses.

Assistant modes:

- `maya_chief_of_staff`
- `neutral_robot`
- `quiet_observer`

Maya directives:

- Protect the Asset
- Buy Back Time
- Ensure Profitability
- Verify before acting
- Clarity over cleverness

Permission levels:

- `observe`
- `suggest`
- `ask_approval`
- `execute`

Permission labels are advisory inside Maya Core. v0.4 does not add motors, voice, or external agentic tools.

Maya endpoints:

```text
POST /maya/compose
POST /maya/briefing
```

Example composed response shape:

```json
{
  "mode": "maya_chief_of_staff",
  "permission": "suggest",
  "verified": false,
  "machine_output": {"action": "stop", "speed": 0, "reason": "Obstacle too close"},
  "user_response": "Not verified yet. Safety wins: stop: Obstacle too close"
}
```

Maya never overrides `safety.py` and never presents unverified information as verified.

## Voice I/O

Maya Console voice is push-to-talk only. Tapping the microphone begins one browser `SpeechRecognition` session; tapping again stops it. A final transcript uses the existing `/voice/conversation` endpoint, and only replies originating from that voice action are read with browser `SpeechSynthesis`.

Chrome exposes Web Speech recognition through `SpeechRecognition` or its prefixed form. Safari has supported Web Speech recognition since Safari 14.1, but actual availability, permissions, and whether recognition uses an online service remain browser/OS dependent. The console feature-detects at runtime: unsupported or denied recognition leaves text chat fully usable and shows a friendly status rather than a raw browser error. Test the actual phone/browser combination; private-network microphone access may require a secure Tailscale HTTPS URL rather than plain LAN HTTP.

- No always-listening mode
- No wake word
- No autonomous action trigger
- No motors
- No external agentic tools

Cloud TTS such as ElevenLabs or OpenAI TTS is a future quality decision, not part of v0.13.0.

## Single-Photo Vision

The Conversation view uses the mobile-native pattern `<input type="file" accept="image/*" capture="environment">`. The user explicitly opens the picker/camera and submits one image to `POST /vision/photo`. The server validates type, size, and signature; writes only a temporary approved file; calls real Gemini; logs the resulting description and conversational context; and deletes the bytes in a `finally` block. There is no `getUserMedia`, live preview, continuous video, background capture, or silent camera activation.

Configure either `GEMINI_API_KEY` for Google AI directly or `OPENROUTER_API_KEY` for the pinned Google Gemini model. Camera permission and picker behavior are controlled by the phone browser. If analysis fails, Maya reports a friendly failure and does not invent a visual description.

Voice endpoint:

```text
POST /voice/conversation
```

CLI mock demo:

```powershell
python -m grimbot_brain.voice_cli --push-to-talk --mock-transcript "what should I clean first?"
```

Voice conversation responses keep command and memory JSON separate from speech output:

```json
{
  "transcript": "what should I clean first?",
  "agent_response": {
    "intent": "room_or_physical_request",
    "user_response": "For the physical side, I would start with: clear hazard: loose cord on floor. Safety stays in front; this is guidance only, not movement.",
    "verified": false
  },
  "machine_output": {"next_best_action": "clear hazard: loose cord on floor"},
  "speech_output": {
    "text": "For the physical side, I would start with: clear hazard: loose cord on floor. Safety stays in front; this is guidance only, not movement.",
    "mode": "mock",
    "audio_path": null
  }
}
```

## Conversational Maya

`POST /voice/conversation` now returns an `agent_response` object with intent, natural `user_response`, retrieved context, skill/procedure suggestions, machine output, and verification state.

Supported intents:

- `casual_chat`
- `chief_of_staff_briefing`
- `project_recall`
- `memory_search`
- `skill_request`
- `procedure_request`
- `dream_review`
- `room_or_physical_request`
- `unclear`

The default conversation provider is mock/deterministic. Optional provider selection uses `GRIMBOT_CONVERSATION_PROVIDER`, separate from vision and dreaming providers. v0.10.2 does not require Gemini, OpenAI, Claude, or any paid API for conversation tests.

Real conversation providers are conversation-only and never receive tool access. Set one provider explicitly:

```powershell
$env:GRIMBOT_CONVERSATION_PROVIDER="claude"  # requires ANTHROPIC_API_KEY
$env:GRIMBOT_CONVERSATION_PROVIDER="openai"  # requires OPENAI_API_KEY
$env:GRIMBOT_CONVERSATION_PROVIDER="openrouter"  # requires OPENROUTER_API_KEY
$env:GRIMBOT_CONVERSATION_PROVIDER="gemini"  # requires GEMINI_API_KEY
```

OpenRouter uses `OPENROUTER_MODEL`, defaulting to `openrouter/auto`, and may include `OPENROUTER_SITE_URL` as the optional HTTP referer.

Or use `GRIMBOT_CONVERSATION_PROVIDER=auto` to prefer Claude, then OpenAI, then OpenRouter, then Gemini when keys exist. The default remains `mock`, even if API keys are present.

LLM conversation wording uses a minimal JSON envelope containing only `user_response`. Legacy full-shape responses remain accepted for compatibility, but provider output can never replace intent, machine output, verification state, skill/procedure suggestions, or safety metadata. Malformed JSON receives one bounded correction retry before Maya falls back to deterministic wording.

Maya may suggest skills, procedures, reviews, and next actions. The only implemented external call is classifier-authorized Tavily snippet search; she may not execute procedures, follow search links, call arbitrary external tools, control hardware, approve changes, or override `safety.py`.

## Skills Registry

Skills are safe internal modules. v0.6 and v0.7 do not add motors, autonomous actions, email, calendar, GitHub, external tools, or arbitrary filesystem writes.

Built-in skills:

- `room_cleanup_plan`
- `checklist_builder`
- `memory_review`
- `maya_briefing`
- `task_breakdown`

Skill endpoints:

```text
GET /skills
GET /skills/{skill_name}
POST /skills/{skill_name}/run
```

Skill results keep machine output separate from Maya text:

```json
{
  "allowed": true,
  "permission": "suggest",
  "machine_output": {
    "skill": "room_cleanup_plan",
    "next_best_action": "clear hazard: loose cord on floor"
  },
  "maya_response": {
    "user_response": "Boss, I can run the room cleanup planning skill. Permission level: suggest."
  }
}
```

## Adaptive State

Adaptive state is a lightweight state-weighting system, not emotions, consciousness, or machine learning. It tracks bounded SQLite-backed signals:

- `attention`
- `urgency`
- `novelty`
- `confidence`
- `reward`
- `friction`
- `fatigue`
- `curiosity`

State endpoints:

```text
GET /state
POST /state/event
POST /state/decay
POST /state/reset
```

Room scans and memory frequency can update state. State can influence Maya response style and skill suggestion ranking, but it never overrides `safety.py`, executes skills directly, controls motors, or bypasses permission gates.

Example state-informed response:

```json
{
  "machine_output": {"next_best_action": "clear hazard: loose cord on floor"},
  "user_response": "Not verified yet. First: Urgency is elevated, so I will keep this concise. clear hazard: loose cord on floor. Then reassess."
}
```

Safety remains authoritative. Skills can plan and suggest, but they cannot execute movement.

## Dreaming Foundation

Dreaming is reflection, not reaction. v0.8 uses deterministic rule-based clustering only; it does not use an LLM provider or autonomous learning.

Dream endpoints:

```text
POST /dream/run
GET /dream/status
GET /dream/facts
GET /dream/promotions
POST /dream/promotions/{id}/approve
POST /dream/promotions/{id}/reject
```

Dream cycles are manual only. They read episodic memories and may write candidate semantic facts, promotion records, and dream logs. Pending and rejected facts are quarantined from active robot-memory retrieval. Only human-approved or anchored facts become available to normal recall.

Dreaming never modifies `safety.py`, adaptive state, skills, actions, motors, or live episodic memory. There is no idle-time trigger, automatic dreaming, or automatic promotion.

## Procedural Memory

Procedural memory stores ordered, reviewable sequences:

- A skill is one atomic capability.
- A procedure is an ordered sequence of steps.
- A workflow is a procedure with branches.

v0.9 stores and matches procedures but cannot execute them. It has no procedure execution endpoint and cannot invoke skills, actions, motors, external tools, adaptive state, or safety changes.

Procedure endpoints:

```text
GET /procedures
GET /procedures/{procedure_id}
GET /procedures/pending
POST /procedures/pending/{pending_id}/approve
POST /procedures/pending/{pending_id}/reject
POST /procedures/match
```

Active procedures are returned by default. Updates archive the previous version and create a new immutable version. Pending proposals require explicit human approval or rejection. Matching uses exact IDs, normalized names, and conservative standard-library fuzzy trigger matching.

Design details: [`docs/procedural_memory_design.md`](docs/procedural_memory_design.md)

## Maya Console

Maya Console is a lightweight FastAPI-served HTML, CSS, and JavaScript interface. It uses existing structured API contracts and keeps Maya text, machine output, review decisions, and passive memory inspection visibly separate.

Design details: [`docs/maya_console_design.md`](docs/maya_console_design.md)

## Chief of Staff Context

Maya is Julian's Chief of Staff first and a room assistant second. v0.10.1 adds SQLite-backed context for:

- Julian's profile, mission, beliefs, constraints, and protocols
- Grim Empire Holdings LLC and the active venture portfolio
- project status, priority, bottleneck, next action, and related entities
- source separation between Julian Prime, Maya, GrimBot, the Board, and the portfolio seed
- explicit verified and unverified context

Context endpoints:

```text
GET /context
GET /context/projects
GET /context/priorities
GET /context/relationships
POST /context/search
POST /context/remember
POST /context/update-priority
```

Maya briefings rank life and business priorities, active projects, bottlenecks, and next actions before room information. Conversation only defaults to robot memory for explicitly physical questions such as rooms, cleaning, vision, hazards, sensors, or movement.

Design details: [`docs/chief_of_staff_context_design.md`](docs/chief_of_staff_context_design.md)

## Test

```powershell
pytest
```
