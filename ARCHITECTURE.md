# Architecture

GrimBot Butler OS is organized as a modular robotics platform. The current release focuses on Phase 0: a simulated robot brain that can run locally before any physical hardware is connected.

## Modules

- `brain/` contains the runnable FastAPI brain server and CLI demo.
- `memory/` stores SQLite robot memory and local runtime databases.
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

Room scan output is also structured JSON:

```json
{
  "room_summary": "Short summary",
  "visible_objects": ["floor"],
  "mess_zones": ["general surfaces"],
  "hazards": [],
  "suggested_cleanup_order": ["general surfaces"],
  "next_best_action": "general surfaces",
  "mode": "mock",
  "image_path": null
}
```

## Current Brain Cycle

1. Accept image path or mock camera frame, IMU readings, battery percentage, distance sensor reading, and user command.
2. Run perception in mock mode by default, with optional Gemini support.
3. Convert perception and user command into a high-level robot intent.
4. Validate intent through safety rules.
5. Log the full cycle to SQLite.
6. Return only the final safe command.

## Current Room Scan Flow

1. Accept either a webcam capture request, an approved local image path, or mock camera text.
2. Save webcam frames only into the configured safe image directory, `vision/images` by default.
3. Reject arbitrary image paths outside the safe image directory.
4. Run Gemini room scanning only for approved image files when mock mode is disabled and an API key is present.
5. Fall back to mock room scanning without API keys or webcam hardware.
6. Store the structured room scan result in SQLite.
7. Extract visible objects, hazards, mess zones, and cleanup tasks into robot memory.

## Current Robot Memory Flow

SQLite remains the source of truth. v0.3 adds structured tables for:

- rooms
- room_zones
- known_objects
- hazards
- mess_observations
- cleanup_tasks
- episodic_memories
- semantic_facts

Memory writes use normalized room, zone, and item keys. Repeated observations update count, confidence, importance, and last-seen timestamps instead of creating endless duplicates.

Robot memory supports:

- remembering a user-provided fact or observation
- recalling what is known about a room
- listing hazards by room or zone
- listing recurring mess zones by room or zone
- returning relevant cleanup context and the next best cleanup action

Memory may provide planning context, but safety remains authoritative. A remembered fact that a hallway is usually clear cannot override a live obstacle stop from `safety.py`.
