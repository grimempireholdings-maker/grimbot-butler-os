# GrimBot Butler OS

An open, modular AI-powered robotic butler platform.

## Mission

Create a safe, affordable personal robotic assistant capable of perception, memory, planning, conversation, automation, and eventually physical household assistance.

## Roadmap

- [x] Brain Simulation
- [ ] Vision
- [ ] Voice
- [ ] Long-Term Memory
- [ ] Agentic Tool Use
- [ ] Rover Platform
- [ ] Object Manipulation
- [ ] Household Assistance
- [ ] Humanoid Platform

## Current Status

Phase 0 includes a runnable Python/FastAPI robot brain simulator. It accepts sensor input, runs mock or Gemini-backed perception, creates planner intent, validates all movement through a safety layer, and logs cycles to SQLite.

LLM output is never connected directly to motors. Every movement command must pass through `brain/grimbot_brain/safety.py`.

By default, local cycle logs are stored at `memory/grimbot_brain.sqlite3`.

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

## Test

```powershell
pytest
```
