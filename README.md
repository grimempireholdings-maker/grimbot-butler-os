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

LLM output is never connected directly to motors. Every movement command must pass through `brain/grimbot_brain/safety.py`.

By default, local cycle logs are stored at `memory/grimbot_brain.sqlite3`.
Approved room-scan images are stored under `vision/images` by default. The API does not accept arbitrary file uploads.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

To enable Gemini perception:

```powershell
pip install -e ".[gemini,dev]"
copy .env.example .env
```

Then set `GEMINI_API_KEY` and `GRIMBOT_MOCK_PERCEPTION=false` in `.env`.

To enable real webcam capture:

```powershell
pip install -e ".[vision]"
```

Voice I/O works in mock mode by default and does not require microphone hardware.

## Run Brain Server

```powershell
uvicorn grimbot_brain.main:app --reload
```

Local API docs:

```text
http://127.0.0.1:8000/docs
```

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

Voice is push-to-talk only.

- No always-listening mode
- No wake word
- No autonomous action trigger
- No motors
- No external agentic tools

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
  "machine_output": {"next_best_action": "clear hazard: loose cord on floor"},
  "speech_output": {
    "text": "Not verified yet. Here is the signal: Next best action is clear hazard: loose cord on floor",
    "mode": "mock",
    "audio_path": null
  }
}
```

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

## Test

```powershell
pytest
```
