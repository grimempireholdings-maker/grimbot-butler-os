# Architecture

GrimBot Butler OS is organized as a modular robotics platform. The current release focuses on Phase 0: a simulated robot brain that can run locally before any physical hardware is connected.

## Modules

- `brain/` contains the runnable FastAPI brain server and CLI demo.
- `memory/` stores SQLite robot memory and local runtime databases.
- `perception/` is reserved for camera, sensor, and multimodal perception adapters.
- `planner/` is reserved for higher-level task and behavior planning.
- `safety/` is reserved for shared safety policies, hardware limits, and validation gates.
- `skills/` is reserved for safe internal butler skills and future external tool integrations.
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

## Current Maya Core Flow

v0.4 adds Maya as a personality, judgment, and communication layer. Maya does not create machine commands. She composes user-facing responses around existing structured output.

Assistant modes:

- `maya_chief_of_staff` applies Protect the Asset, Buy Back Time, Ensure Profitability, Verify before acting, and Clarity over cleverness.
- `neutral_robot` keeps responses plain and procedural.
- `quiet_observer` observes only and does not recommend execution.

Permission levels are explicit:

- `observe`
- `suggest`
- `ask_approval`
- `execute`

These are response-layer permission labels. v0.4 does not add execution machinery, motor control, voice, or external agentic tools.

Response composition returns both fields separately:

```json
{
  "machine_output": {"action": "stop", "speed": 0, "reason": "Obstacle too close"},
  "user_response": "Verified. Safety wins: stop: Obstacle too close"
}
```

Maya briefing summarizes priority items, FYI, wins, hazards, and the next best action from robot memory. Verification status is explicit; Maya must not label information verified unless the caller marks it verified.

## Current Voice Flow

v0.5 adds conversational voice as a push-to-talk I/O layer. It does not add always-listening behavior, wake words, motors, autonomous actions, or external tools.

1. The caller must send `push_to_talk=true`.
2. Speech-to-text runs in mock mode by default, using an explicit transcript.
3. Optional audio paths must resolve inside the configured safe audio directory.
4. The transcript queries robot memory for relevant context.
5. Maya composes a user-facing response.
6. Text-to-speech returns mock speech output by default.
7. Machine output remains separate from speech text.

Safety remains authoritative. Voice context can inform the conversation, but it cannot execute motion or override `safety.py`.

## Current Skills Flow

v0.6 adds a safe internal skills registry. Skills are Python modules with a fixed interface:

- name
- description
- category
- required_permission
- inputs_schema
- outputs_schema
- can_execute()
- execute()

The registry can register skills, list skills, find skills by name/category, validate permission before execution, and return structured JSON results.

Built-in v0.6 skills:

- `room_cleanup_plan`
- `checklist_builder`
- `memory_review`
- `maya_briefing`
- `task_breakdown`

Skills may read robot memory and compose Maya responses. They do not add motors, autonomous action, email, calendar, GitHub, external tools, or arbitrary filesystem writes.

Skill responses keep `machine_output` separate from `maya_response`. Permission gates are enforced before skill execution. Safety remains authoritative if any skill output is later used as context for movement planning.

## Current Adaptive State Flow

v0.7 adds a lightweight adaptive state system inspired by pheromone-style pressure signals. It is not emotions, consciousness, or machine learning. SQLite remains the source of truth.

Tracked signals:

- attention
- urgency
- novelty
- confidence
- reward
- friction
- fatigue
- curiosity

Each signal stores its current value, min/max bounds, baseline, decay rate, last update time, source, and reason. Event updates are bounded between 0 and 1. Decay drifts values back toward baseline over time.

State events include:

- repeated hazards raising urgency and attention
- successful cleanup raising reward and confidence
- ignored recommendations raising friction
- new rooms or objects raising novelty and curiosity
- low battery or unsafe sensors raising fatigue and urgency

Adaptive state can influence Maya response style, skill suggestion ranking, memory relevance pressure, and `next_best_action` wording. It cannot execute skills, bypass permission gates, control motors, or override `safety.py`.

State endpoints:

```text
GET /state
POST /state/event
POST /state/decay
POST /state/reset
```

## Current Dreaming Flow

v0.8 adds a manual reflection pipeline. Dreaming is not a live reaction system, an autonomous learner, or a self-modification mechanism.

1. A human explicitly calls `POST /dream/run`.
2. The rule-based provider reads a bounded snapshot of unconsolidated episodic memories.
3. The consolidator clusters repeated observations by normalized content, room, zone, action, outcome, object, and hazard tags.
4. Candidate semantic facts are created or reinforced.
5. New candidates enter `promotion_queue` as `pending`.
6. Pending and rejected candidates remain excluded from active robot-memory retrieval.
7. A human approves, rejects, or anchors a candidate through a review endpoint.
8. Every run writes an auditable `dream_cycles` record.

The forgetting subsystem scores semantic facts using importance, frequency, and recency. It never removes core facts, safety-related facts, approved facts, anchored facts, or facts awaiting review. Episodic memories are read-only during dream cycles.

Dreaming writes only:

- `semantic_facts`
- `promotion_queue`
- `dream_cycles`

Dreaming does not modify adaptive state, execute skills, issue actions, control motors, or bypass `safety.py`. There are no automatic, scheduled, or idle-time dream triggers.
