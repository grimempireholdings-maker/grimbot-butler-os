from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from pathlib import Path

from grimbot_brain import main as main_module
from grimbot_brain.memory import BrainMemory
from grimbot_brain.workspace.workspace_inspector import WorkspaceInspector


def _get(path: str) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict] = []
    request_sent = False

    async def receive() -> dict:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(main_module.app(scope, receive, send))
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    headers = {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in start["headers"]
    }
    return start["status"], headers, body


def _table_counts(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        return {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }


def test_console_route_returns_html_with_operator_sections() -> None:
    status, headers, body = _get("/console")
    html = body.decode("utf-8")

    assert status == 200
    assert headers["content-type"].startswith("text/html")
    for section in (
        "Chat with Maya",
        "Maya Briefing",
        "Julian's Operating Context",
        "Workspace",
        "Internal State",
        "Skills",
        "Dreaming",
        "Procedural Memory",
        "Memory",
    ):
        assert section in html


def test_console_static_assets_load() -> None:
    css_status, css_headers, css_body = _get("/console/assets/console.css")
    js_status, js_headers, js_body = _get("/console/assets/console.js")

    assert css_status == 200
    assert css_headers["content-type"].startswith("text/css")
    assert b".console-grid" in css_body
    assert js_status == 200
    assert "javascript" in js_headers["content-type"]
    assert b"loadAllReadOnlyPanels" in js_body


def test_console_chat_uses_voice_conversation_agent_response() -> None:
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")
    html = (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")
    paths = {route.path for route in main_module.app.routes}

    assert "/console/chat" not in paths
    assert 'api("/voice/conversation"' in script
    assert "result.agent_response?.user_response" in script
    assert script.index("result.agent_response?.user_response") < script.index("result.maya_response?.user_response")
    assert "/console/assets/console.js?v=0.10.5" in html
    assert "/console/assets/console.css?v=0.10.5" in html
    assert main_module.app.version == "0.10.5"


def test_console_route_disables_html_cache() -> None:
    status, headers, _ = _get("/console")

    assert status == 200
    assert headers["cache-control"] == "no-store"


def test_console_page_load_does_not_mutate_database(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "console.sqlite3")
    monkeypatch.setattr(main_module, "memory", memory)
    before = _table_counts(memory.db_path)

    status, _, _ = _get("/console")

    assert status == 200
    assert _table_counts(memory.db_path) == before


def test_console_context_load_does_not_mutate_database(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "console.sqlite3")
    monkeypatch.setattr(main_module, "memory", memory)
    before = _table_counts(memory.db_path)

    status, headers, body = _get("/context")

    assert status == 200
    assert headers["content-type"].startswith("application/json")
    assert b'"projects"' in body
    assert _table_counts(memory.db_path) == before


def test_console_initial_loaders_are_read_only() -> None:
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")
    initial_load = script.split("async function loadAllReadOnlyPanels()", 1)[1].split(
        "function bindEvents()", 1
    )[0]

    assert 'method: "POST"' not in initial_load
    assert "runDream" not in initial_load
    assert "reviewDream" not in initial_load
    assert "reviewProcedure" not in initial_load
    assert "runSkill" not in initial_load
    assert "rememberContext" not in initial_load
    assert "searchContext" not in initial_load
    daily_loaders = script.split("const DAILY_LOADERS = {", 1)[1].split("};", 1)[0]
    assert "context: loadContext" in daily_loaders
    assert "workspace: loadWorkspace" in daily_loaders
    for heavy_loader in ("loadState", "loadSkills", "loadDream", "loadProcedures", "loadMemory"):
        assert heavy_loader not in daily_loaders


def test_console_does_not_expose_procedure_execution() -> None:
    paths = {route.path for route in main_module.app.routes}
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")

    assert "/procedures/execute" not in paths
    assert "/procedures/run" not in paths
    assert "/procedures/execute" not in script
    assert "/procedures/run" not in script


def test_console_workspace_panel_and_developer_mode_defaults() -> None:
    html = (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")

    assert 'id="workspace-title"' in html
    assert 'data-refresh="workspace"' in html
    assert 'id="workspace-search-form"' in html
    assert 'id="developer-mode"' in html
    assert html.count("developer-panel") == 5
    assert html.count("developer-panel\" aria-labelledby") == 5
    assert html.count("hidden>") >= 5
    assert 'const DAILY_LOADERS = {' in script
    assert "workspace: loadWorkspace" in script
    assert 'document.querySelectorAll(".developer-panel")' in script
    for daily_panel in ("chat-title", "briefing-title", "context-title", "workspace-title"):
        panel_markup = html.split(f'id="{daily_panel}"', 1)[0].rsplit("<section", 1)[1]
        assert "developer-panel" not in panel_markup
        assert " hidden" not in panel_markup


def test_console_workspace_and_context_initial_gets_do_not_mutate(tmp_path, monkeypatch) -> None:
    memory = BrainMemory(tmp_path / "console.sqlite3")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=workspace_root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )
    (workspace_root / "README.md").write_text("read-only workspace", encoding="utf-8")
    monkeypatch.setattr(main_module, "memory", memory)
    monkeypatch.setattr(main_module, "workspace", WorkspaceInspector(workspace_root))
    database_before = _table_counts(memory.db_path)
    files_before = {
        path.relative_to(workspace_root).as_posix(): path.read_bytes()
        for path in workspace_root.rglob("*")
        if path.is_file()
    }

    context_status, _, _ = _get("/context")
    workspace_status, _, _ = _get("/workspace")

    files_after = {
        path.relative_to(workspace_root).as_posix(): path.read_bytes()
        for path in workspace_root.rglob("*")
        if path.is_file()
    }
    assert context_status == 200
    assert workspace_status == 200
    assert _table_counts(memory.db_path) == database_before
    assert files_after == files_before


def test_context_modules_do_not_add_external_or_execution_paths() -> None:
    identity_dir = Path(main_module.__file__).resolve().parent / "identity"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in identity_dir.glob("*.py")
    ).lower()
    paths = {route.path for route in main_module.app.routes}

    for forbidden in (
        "import requests",
        "import httpx",
        "import smtplib",
        "github",
        "calendar",
        "motor",
        "execute_procedure",
        "auto_approve",
    ):
        assert forbidden not in source
    assert not any(
        path.startswith("/context") and ("execute" in path or "approve" in path)
        for path in paths
    )
