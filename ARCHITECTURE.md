# Architecture

GrimBot Butler OS is organized as a modular robotics platform. The current release focuses on Phase 0: a simulated robot brain that can run locally before any physical hardware is connected.

## Modules

- `brain/` contains the runnable FastAPI brain server and CLI demo.
- `memory/` is reserved for long-term memory systems and retrieval.
- `perception/` is reserved for camera, sensor, and multimodal perception adapters.
- `planner/` is reserved for higher-level task and behavior planning.
- `safety/` is reserved for shared safety policies, hardware limits, and validation gates.
- `skills/` is reserved for agentic tool use and household automation skills.
- `voice/` is reserved for speech input and output.
- `vision/` is reserved for webcam and computer vision pipelines.
- `hardware/` is reserved for rover, actuator, motor, and sensor integrations.
- `docs/` contains project documentation.
- `tests/` contains automated tests.

## Safety Principle

Planning and perception may suggest intent, but they do not directly control hardware. Any movement command must be validated by the safety layer before it can become an executable robot command.

The current command contract is strict JSON:

```json
{"action":"stop","speed":0,"reason":"Obstacle too close"}
```

## Current Brain Cycle

1. Accept image path or mock camera frame, IMU readings, battery percentage, distance sensor reading, and user command.
2. Run perception in mock mode by default, with optional Gemini support.
3. Convert perception and user command into a high-level robot intent.
4. Validate intent through safety rules.
5. Log the full cycle to SQLite.
6. Return only the final safe command.
