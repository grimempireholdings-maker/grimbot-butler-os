# Procedural Memory Design

GrimBot Procedural Memory v0.9 stores reusable sequences without executing them.

## Definitions

- **Skill:** one atomic capability.
- **Procedure:** an ordered sequence of steps.
- **Workflow:** a procedure with branches.

## Scope

v0.9 includes:

- strict Pydantic procedure models
- SQLite persistence
- immutable version history
- archive and rollback lookup
- passive execution statistics
- pending proposal review
- exact and fuzzy matching

v0.9 excludes:

- procedure execution
- skill invocation from procedures
- motor or action control
- adaptive-state mutation
- safety-rule modification
- external tool calls
- automatic proposal approval

## Versioning

Each procedure name has at most one active version. Updating an active procedure archives it and inserts the next version. Historical records are never overwritten and can be retrieved by normalized name and version.

## Review

Pending proposals remain separate from active procedures. Approval and rejection are explicit human-review operations. Approval creates an active procedure; rejection does not.

## Matching

The matcher considers active procedures only:

1. exact procedure ID
2. exact normalized name
3. fuzzy comparison against names and trigger phrases

Fuzzy matching uses Python's standard library. Results below the caller's confidence threshold return `matched: false`.

## Safety Boundary

Procedural memory is descriptive in v0.9. A match is information, not authorization. No endpoint or store method executes procedure steps.
