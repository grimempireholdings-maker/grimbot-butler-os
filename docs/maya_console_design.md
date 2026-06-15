# Maya Console Design

## Purpose

Maya Console is a local operator surface for GrimBot Butler OS. It makes the existing brain, Maya, memory, skills, dreaming, and procedural-memory APIs usable from a browser before physical hardware exists.

It is intentionally a thin client. Business logic, validation, permissions, storage, and safety remain in the existing Python modules.

## Delivery

- `GET /console` serves the console HTML.
- `/console/assets/console.css` serves local styling.
- `/console/assets/console.js` serves local interaction logic.
- No frontend framework, build step, CDN, or external service is required.

## Panels

- Conversation uses `POST /voice/conversation` in explicit push-to-talk mock mode.
- Briefing uses `POST /maya/briefing`.
- Adaptive state uses `GET /state`.
- Skills use `GET /skills` and operator-triggered skill runs.
- Dreaming uses fact and promotion GET endpoints plus manual run and review actions.
- Procedural memory uses active and pending GET endpoints plus manual review and match actions.
- Robot memory uses rooms, hazards, mess-zone, and relevant-memory endpoints.

## Loading Policy

Initial page load and refresh controls call read-only GET endpoints only. Mutation-capable endpoints are bound to explicit operator form submissions or buttons. There are no timers, background jobs, automatic dream cycles, or automatic approval flows.

## Safety Boundaries

The console:

- does not add or call a procedure execution endpoint
- does not add motors or hardware control
- does not add autonomous actions
- does not add external tools
- does not bypass skill permission checks
- does not modify adaptive state on page load
- does not auto-approve dream facts or procedure proposals
- does not write outside existing backend-managed application storage

Machine output is displayed separately from Maya's user-facing text. Procedure matching is labeled and treated as matching only.

## Error Handling

The shared API helper accepts structured FastAPI error details and converts them to readable UI messages. Each panel handles empty arrays independently so one empty or unavailable subsystem does not crash the console.
