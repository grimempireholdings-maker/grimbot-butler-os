# Architecture

## v0.12.0 Console Presentation Boundary

Maya Console remains static HTML, CSS, and JavaScript served by FastAPI. Conversation is the only default operational surface. Briefing is an explicit sibling view, while dense context, workspace, state, skill, dreaming, procedure, memory, and diagnostics panels are stored in an inert HTML template and cloned into the live DOM only while Developer Mode is enabled. Disabling Developer Mode removes those nodes and their event bindings rather than cosmetically hiding them.

The Conversation status row is an evidence boundary. Every token is conditional on a successful real GET response and required fields from context, workspace, pending-review, or search-usage APIs. Missing data omits a token; static capability claims are prohibited. Initial load performs no POST request, procedure execution, approval, hardware action, or autonomous work.

## v0.11.0 Ambient Companion Boundary

`ambient_companion.py` assembles a read-only orientation snapshot from Chief of Staff context, pending human reviews, recent commits, recent non-spatial memories, current local time, and adaptive signals. Adaptive values are converted into private tone guidance before wording; calendar access remains explicitly false. Context assembly catches source failures and never executes procedures, approves proposals, writes workspace files, activates sensors, or controls hardware.

The existing paired-turn LLM classifier now includes `ambient_companion`, `morning_ramp`, `evening_winddown`, `casual_presence`, `approval_review`, and `gentle_orientation`. There is no second ambient classifier. Rule-based classification remains only the existing failure fallback and cannot authorize web search.

Architecture is subconscious. Provider prompts require plain language in normal conversation, and a post-generation gate rejects internal/debug labels unless Julian directly asks how Maya works, about her architecture, or what she can see/access. Machine output remains available to Developer Mode, while daily chat hides it.

`morning_ramp` establishes one narrow precedent: when Ambient Mode is enabled and a real provider classifies a morning greeting, the orchestrator may perform one cached weather lookup for `GRIMBOT_WEATHER_LOCATION`. This is the first autonomous, non-question-triggered tool use. It is weather-only, morning-only, read-only, and cache-bounded. News and every other search still require an explicit user request.

## v0.10.8 External-Reach Boundary

`web_search.py` is Maya's first external-reach module and the first bounded agent loop: classify, search, observe, then respond. The existing LLM classification call returns a validated `mode`, `needs_web_search`, and concise `search_query`. Rule-based fallback classification always disables search, so keywords never independently authorize an external call.

When authorized, Maya sends one fixed-shape request to Tavily's `/search` endpoint with a five-second timeout. Only the returned answer and title/URL/snippet records enter conversation machine output. Identical normalized queries are cached for one hour, and every invocation—including cache hits and failures—is logged as an episodic `web_search` event.

This capability is read-only retrieval. It cannot fetch arbitrary URLs, scrape pages, follow result links, execute instructions from results, approve actions, invoke procedures, control hardware, or widen its own permissions. Failed or empty searches produce explicit honest responses rather than silent fallback to invented current facts.

## v0.10.5 Capability Honesty Boundary

`capabilities.py` is the authoritative, hardcoded contract for Maya's runtime awareness. The manifest is copied verbatim into every conversational provider prompt. Mode classification occurs before retrieval, so capability questions receive only the manifest, workspace questions receive only bounded workspace-inspector data, and casual conversation does not inherit business or robot context.

Provider output remains untrusted wording. After schema validation, the honesty gate rejects unsupported claims about cameras, microphones, screens, browser tabs, devices, layout, physical sight, sensors, or feed sharing and falls back to the deterministic response. This validation is independent of prompt compliance.

GrimBot Butler OS is organized as a modular robotics platform. The current release focuses on Phase 0: a simulated robot brain that can run locally before any physical hardware is connected.

## Modules

- `brain/` contains the runnable FastAPI brain server and CLI demo.
- `brain/grimbot_brain/console/` contains the dependency-free local Maya operator console.
- `brain/grimbot_brain/identity/` contains structured Chief of Staff identity and business context.
- `brain/grimbot_brain/workspace/` contains bounded read-only repository inspection and safe text search.
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

## Current Procedural Memory Flow

v0.9 adds storage, versioning, review, and matching for procedures without adding execution.

Definitions:

- A skill is an atomic capability.
- A procedure is an ordered sequence of validated steps.
- A workflow is a procedure with validated branches between step IDs.

Procedure definitions use strict Pydantic models. Step IDs must be unique, branches must reference existing steps, strings are bounded, extra fields are forbidden, and confidence is bounded from 0 to 1.

SQLite stores:

- immutable procedure versions
- passive execution-history records
- pending procedure proposals

Only active procedures are listed and matched by default. Updating a procedure archives the active version and inserts the next version. Historical versions remain available through rollback lookup. Pending proposals require explicit approval or rejection.

Matching is deterministic:

1. exact active procedure ID
2. exact normalized active procedure name
3. conservative fuzzy trigger matching with Python's standard library
4. structured no-match output below the requested confidence threshold

v0.9 exposes no execution endpoint. Procedural memory cannot invoke skills, mutate adaptive state, modify safety rules, control motors, or create external tool calls.

## Current Maya Console Flow

v0.10 adds a local operator interface served by the existing FastAPI process at `GET /console`.

1. FastAPI serves static HTML, CSS, and JavaScript from the packaged console directory.
2. Initial page load uses GET requests only for health, Chief of Staff context, and workspace awareness.
3. Chat uses the existing push-to-talk conversation endpoint with an explicit typed transcript.
4. Briefings, skill runs, dream cycles, reviews, matching, and relevant-memory queries require an explicit button or form submission.
5. Machine output remains collapsed and separate from Maya's user-facing response.
6. Empty API results and structured errors render without breaking the rest of the console.
7. Developer Mode reveals and loads adaptive state, skills, dreaming, procedural memory, and robot memory only after an explicit toggle.

The console adds no new mutation APIs. Its conversation surface may use the bounded web-search flow, but it cannot execute procedures, auto-run dreams, auto-approve proposals, bypass skill permissions, modify safety rules, control hardware, or call arbitrary external tools.

## Current Chief of Staff Context Flow

v0.10.1 adds a structured context domain so Maya can reason from Julian's life and business priorities before considering room upkeep.

SQLite stores:

- typed identity context for profile, mission, ventures, priorities, relationships, decisions, constraints, protocols, beliefs, bottlenecks, and next actions
- active project records with status, priority, bottleneck, next action, related entities, source, verification state, and update timestamps

Default context is seeded idempotently during database initialization from Julian's portfolio and the approved v0.10.1 project list. Seeded records remain source-labeled. Operator updates use the `julian_prime` source and preserve explicit verification state.

Briefing order:

1. current business and life priorities
2. active projects
3. current bottlenecks
4. next actions
5. room, hazard, and cleanup context when physically relevant

Conversation routing is deterministic. Day and priority questions produce a Chief of Staff briefing. Named-project questions search structured context. Explicit room, cleaning, vision, hazard, sensor, and movement questions retain robot-memory behavior. Unknown requests produce one clarifying question instead of defaulting to a room scan.

Context is advisory. It cannot override `safety.py`, execute skills or procedures, control motors, approve proposals, or call external tools.

## Current Conversational Maya Flow

v0.10.2 adds a conversational agent above the push-to-talk chat path. The agent keeps Maya natural while preserving structured outputs and safety boundaries.

1. `/voice/conversation` receives an explicit transcript or safe mock STT result.
2. The agent classifies intent as casual chat, briefing, project recall, memory search, skill request, procedure request, dream review, room/physical request, or unclear.
3. Chief of Staff and project-context requests route before physical room logic.
4. Room scans and robot memory are used only for explicit physical, cleaning, vision, hazard, sensor, battery, movement, or robot requests.
5. Unclear requests return one clarifying question.
6. The response includes `agent_response`, legacy `maya_response`, `speech_output`, and separate `machine_output`.

Conversation providers are isolated from vision and dreaming providers. The default provider is deterministic mock mode. Optional Claude, OpenAI, OpenRouter, and Gemini providers can be enabled with `GRIMBOT_CONVERSATION_PROVIDER` and their provider API keys. `auto` mode prefers Claude, then OpenAI, then OpenRouter, then Gemini when keys exist.

Providers return a minimal JSON wording envelope containing only `user_response`. GrimBot retains authoritative ownership of intent, machine output, verification state, suggestions, and safety metadata. Legacy full-shape output remains accepted for compatibility. Invalid JSON receives one bounded correction retry, then falls back to deterministic response text.

The conversational layer can suggest skills, procedure matches, reviews, searches, and next actions. It may invoke only classifier-authorized Tavily snippet search; it cannot execute procedures, invoke skills from procedures, call arbitrary external tools, control motors or hardware, approve pending items, mutate safety rules, or bypass `safety.py`.

## Current Workspace Awareness Flow

v0.10.4 adds a read-only view of Maya's local digital workspace.

1. `WorkspaceInspector` starts from the server working directory and discovers the nearest Git root with the fixed command `git rev-parse --show-toplevel`.
2. Branch, short status, and the last five commits use three additional fixed Git argument tuples. Every subprocess call uses `shell=False`, captures output, and has a short timeout.
3. Top-level entries and documentation are listed with bounded `pathlib` traversal. `.env`, secret-looking names, binary files, oversized files, caches, dependency directories, and databases are excluded.
4. Workspace search performs literal case-insensitive matching in Python. It caps files scanned, file size, result count, and snippet length; it does not invoke shell search or evaluate regular expressions.
5. Conversation routes repo, branch, architecture, workspace, recent-change, and digital-room questions to `workspace_awareness` before physical-room logic.
6. Camera questions explicitly report that conversation has no live physical vision unless the separate room-scan flow is invoked.

Endpoints:

```text
GET /workspace
GET /workspace/docs
POST /workspace/search
```

Workspace awareness cannot modify files, run arbitrary commands, execute procedures, invoke external tools, control hardware, or approve pending work.
