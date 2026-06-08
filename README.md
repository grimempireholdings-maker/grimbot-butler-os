# GrimBot Butler OS

An open, modular AI-powered robotic butler platform.

## Mission

Create a safe, affordable personal robotic assistant capable of perception, memory, planning, conversation, automation, and eventually physical household assistance.

## Roadmap

- [x] Brain Simulation
- [x] Vision
- [ ] Voice
- [ ] Long-Term Memory
- [ ] Agentic Tool Use
- [ ] Rover Platform
- [ ] Object Manipulation
- [ ] Household Assistance
- [ ] Humanoid Platform

## Current Status

Phase 0 includes a runnable Python/FastAPI robot brain simulator. It accepts sensor input, runs mock or Gemini-backed perception, creates planner intent, validates all movement through a safety layer, and logs cycles to SQLite.

Phase 1 adds GrimBot Vision v0.2: safe local webcam frame capture, approved-image validation, and Gemini or mock room scans that return structured JSON.

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

## Test

```powershell
pytest
```
