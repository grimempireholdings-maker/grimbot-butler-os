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
- [ ] Agentic Tool Use
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

## Test

```powershell
pytest
```
