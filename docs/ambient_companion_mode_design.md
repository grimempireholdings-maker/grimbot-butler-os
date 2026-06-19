# Ambient Companion Mode v0.11.0

## Principle

Architecture is subconscious. Presence is foreground. Maya should feel aware of the day without narrating how context was classified, retrieved, or weighted.

## Modes

The existing paired-turn LLM classifier owns all six modes: `ambient_companion`, `morning_ramp`, `evening_winddown`, `casual_presence`, `approval_review`, and `gentle_orientation`. No second classifier or keyword-authorized tool path is introduced.

## Read-only context

Ambient orientation may use current priorities and projects, bottlenecks, pending fact and procedure reviews, recent repository commits, recent non-spatial memories, current local time, and weather. Adaptive signals influence tone only and are never presented as a labeled system. Calendar data is unavailable and must not be implied.

All sources are read-only and failure-tolerant. The mode cannot execute procedures, approve proposals, write workspace files, activate a camera or microphone, control motors, or widen permissions.

## Proactive weather precedent

The approved decision is yes to proactive weather, under one narrow gate. `morning_ramp` may issue a cached read-only weather search for the configured `GRIMBOT_WEATHER_LOCATION` even when the greeting did not explicitly ask for weather. This is Maya's first autonomous, non-question-triggered tool use.

The precedent is intentionally constrained:

- Ambient Mode must be enabled.
- The LLM classifier must select `morning_ramp`.
- A real conversation provider must be active; deterministic tests never spend credits.
- The existing one-hour search cache prevents repeated greetings from repeating the external call.
- No proactive news search exists.
- No other mode receives a new autonomous trigger.

Search remains snippet-only, attributed, and subject to the existing honesty and attribution gates.

## Console and diagnostics

Ambient Mode is on by default in daily chat. Developer Mode remains off by default and is the only console surface that reveals machine output, classification source, search fields, and architecture diagnostics.

## Verification

Deterministic tests cover context assembly, mode downgrade, morning weather gating, internal-language rejection, direct architecture permission, console defaults, and safety. Real-provider regressions are opt-in with `GRIMBOT_LIVE_TESTS=1` so ordinary CI remains deterministic and free of provider cost.
