# Maya Workspace Awareness Design

## Purpose

Workspace awareness gives Maya a bounded, read-only view of the local GrimBot project. It answers repository and architecture questions without adding file editing, arbitrary command execution, external tools, or autonomous actions.

## Inspection Contract

`WorkspaceInspector` reports:

- current working repository root and name
- active Git branch
- short Git status, capped at 100 entries
- five most recent one-line commits
- detected package version
- safe top-level files and directories
- detected documentation files
- structured warnings when inspection is unavailable

The Git subprocess allowlist is exact:

```text
git rev-parse --show-toplevel
git branch --show-current
git status --short
git log --oneline -5
```

Commands use argument arrays, `shell=False`, captured text output, `GIT_OPTIONAL_LOCKS=0`, and a three-second timeout. The environment setting prevents observational commands such as `git status` from refreshing the index. Missing Git, non-Git directories, failed commands, and timeouts return warnings rather than crashing the API.

## File Safety

Documentation preview and workspace search skip:

- `.git`, `.venv`, `venv`, `node_modules`, caches, and `__pycache__`
- `.env` and `.env.*`
- filenames containing secret, token, password, credential, API-key, or private-key markers
- SQLite and database files
- unsupported extensions
- files larger than 256 KB
- files whose initial byte sample contains a null byte

Documentation discovery considers at most 250 files and previews are capped at 400 characters. Search scans at most 250 files, processes at most 10,000 characters per line, returns at most 50 requested results, and caps each snippet at 240 characters. Recognizable OpenRouter/OpenAI-style, GitHub, Google, AWS, and Slack key patterns are redacted from Git output, previews, echoed queries, and snippets. Search is literal and case-insensitive; no regular expression or shell command is evaluated from user input.

## Conversation Routing

The `workspace_awareness` intent handles explicit digital questions including repo, branch, workspace, architecture, recent changes, and digital-room requests. Responses identify the access as read-only and distinguish it from physical camera vision.

Physical camera questions do not use workspace data as evidence of sight. Conversation reports no live camera access and points to the separate explicit room-scan flow.

## Console Behavior

Daily-use panels remain visible:

- Chat
- Maya Briefing
- Chief of Staff Context
- Workspace

Developer Mode reveals and then loads:

- Adaptive State
- Skills
- Dreaming
- Procedural Memory
- Robot Memory

The workspace panel loads `GET /workspace`, supports manual refresh, and runs bounded searches only after the operator submits the search form.

## Non-Goals

Workspace awareness does not:

- modify or create files
- run user-provided commands
- execute procedures or skills
- call external tools
- control motors or hardware
- approve pending facts or procedures
- read `.env` contents or expose API keys
