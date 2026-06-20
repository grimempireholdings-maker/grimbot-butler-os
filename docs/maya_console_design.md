# Maya Console Design

## Purpose

Maya Console is a local conversation and operator surface for GrimBot Butler OS. It remains a thin FastAPI-served HTML/CSS/JavaScript client: business logic, permissions, storage, and safety stay in Python. There is no framework, package manager, frontend build, CDN, or public deployment.

The console is intended for desktop and phone use over a trusted LAN or private Tailscale connection. It must not be exposed directly to the public internet.

## Three-mode architecture

Conversation is the default. Everything else is opt-in.

### Conversation

The initial live DOM contains the minimal header and health indicator, a row of live status tokens, chat history, input, send control, Ambient Mode toggle, and persona selector. No context, workspace, memory, state, skill, dreaming, or procedure-review panel is mounted.

### Briefing

Briefing is a discrete view backed by `POST /maya/briefing`. It is populated only after the operator selects Briefing or a non-ambient conversational briefing intent opens it. There is no automatic page-load briefing and no permanent empty briefing card in Conversation.

### Developer

Dense operator panels live inside an inert `<template>`. Checking Developer clones that template into `developer-root`, binds its controls, and then runs the existing read-only loaders. Unchecking Developer removes all mounted panel nodes with `replaceChildren()`. This is a lifecycle boundary, not CSS concealment.

Developer contains Chief of Staff context, active projects, bottlenecks, next actions, context search/write controls, workspace and commits, adaptive state, skills, dreaming review, procedure review and matching, robot memory, and the latest conversation machine diagnostics. Existing review actions retain their explicit human triggers and backend permission gates.

## Real-data status tokens

Status tokens are a compact orientation layer, not product copy. Every rendered value must come from a successful real endpoint response:

- priority count from `GET /context`
- pending review count from `GET /dream/promotions` and `GET /procedures/pending`
- latest commit hash from `GET /workspace`
- monthly search usage from `GET /search/usage`

Each token checks that its response and required fields are present before rendering. Failed or absent data removes the token entirely. Zero is displayed only when a real endpoint returned a valid zero.

Static capability claims are prohibited in console source. The UI must never claim that Maya is integrated with, connected to, or able to access a system unless that statement is derived from a real API field or capability manifest flag. This prevents capability hallucination at the presentation layer.

## Loading policy

Initial load runs only health and status-token GET requests. Developer GET loaders run only after Developer Mode is enabled. Briefing POST runs only after an explicit briefing trigger. Mutation-capable endpoints remain bound to explicit operator forms and review buttons.

There are no timers, background jobs, automatic dream cycles, automatic approvals, procedure execution, motor controls, permission bypasses, or new external tools.

## Mobile behavior

The Conversation surface is sized against the dynamic viewport, respects safe-area insets, keeps chat scroll independent, and anchors the composer near the bottom. At narrow widths, header labels compress, tokens scroll horizontally, the send control becomes a prominent square action, and Developer grids collapse to one column.

## Error handling

The shared API helper converts structured FastAPI errors into readable messages. Developer panels handle empty arrays independently. Status-token failures are intentionally silent: unavailable data produces no token rather than an empty or fabricated claim.
