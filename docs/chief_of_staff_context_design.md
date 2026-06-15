# Chief of Staff Context Design

## Purpose

Maya serves Julian's life and business. GrimBot's room and physical-environment capabilities are one bounded domain inside that mission.

The context layer gives Maya structured, reviewable knowledge about Julian's identity, ventures, projects, priorities, relationships, decisions, constraints, protocols, beliefs, bottlenecks, and next actions.

## Storage

SQLite remains the source of truth.

`identity_context` stores typed context entries with:

- name and content
- priority
- source
- verification state
- creation and update timestamps

`identity_projects` stores:

- name and status
- priority
- current bottleneck
- next action
- related entities
- source and verification state
- update timestamp

Names are normalized for stable lookup and deduplication. Default records use insert-on-conflict behavior so upgrades do not duplicate the seed.

## Source Separation

- `julian_prime`: strategic identity and second-mind direction
- `maya`: Chief of Staff observations and operator context
- `grimbot`: embodied-system observations
- `board`: future specialist-agent context
- `portfolio_seed`: initial context derived from Julian's portfolio and approved release brief

Verification is independent from source. Maya must not describe an unverified record as verified.

## Retrieval

The API provides full context, project, priority, and relationship views plus bounded search. Search prefers direct project-name matches over references in related entities or project descriptions.

When no relevant context exists, the response contains one clarification question and does not recommend a room scan.

## Briefing Policy

Chief of Staff briefings rank:

1. business and life priorities
2. active projects
3. bottlenecks
4. next actions
5. physical-environment context

An explicit room or zone request may surface the room's immediate safe next action, but room scanning is never the default for a general briefing.

## Conversation Policy

- Day, briefing, and priority language uses Chief of Staff context.
- Explicit project names use project-aware context.
- Room, cleaning, vision, hazard, sensor, battery, distance, and movement language uses robot memory.
- Ambiguous requests receive one clarifying question.

Machine output remains separate from Maya's user-facing response.

## Safety

Context is advisory only. It cannot:

- override `safety.py`
- execute skills or procedures
- control motors or hardware
- call external tools
- approve dream facts or procedures
- claim unverified context is verified
