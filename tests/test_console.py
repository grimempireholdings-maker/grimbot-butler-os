from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from grimbot_brain import main as main_module
from grimbot_brain.memory import BrainMemory
from grimbot_brain.workspace.workspace_inspector import WorkspaceInspector


class _TemplateAwareParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.template_depth = 0
        self.live_ids: set[str] = set()
        self.template_ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "template":
            self.template_depth += 1
        element_id = dict(attrs).get("id")
        if element_id:
            target = self.template_ids if self.template_depth else self.live_ids
            target.add(element_id)

    def handle_endtag(self, tag: str) -> None:
        if tag == "template":
            self.template_depth -= 1


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


def test_console_route_defaults_to_conversation_with_dense_panels_inert() -> None:
    status, headers, body = _get("/console")
    html = body.decode("utf-8")

    assert status == 200
    assert headers["content-type"].startswith("text/html")
    parser = _TemplateAwareParser()
    parser.feed(html)
    assert {"conversation-view", "chat-log", "chat-form", "status-tokens"} <= parser.live_ids
    assert {"context-title", "workspace-title", "dream-title", "procedure-title", "memory-title"} <= parser.template_ids
    assert not ({"context-title", "workspace-title", "dream-title", "procedure-title", "memory-title"} & parser.live_ids)
    assert 'id="conversation-view"' in html
    assert 'id="briefing-view"' in html and 'id="briefing-view" class="mode-view briefing-view"' in html


def test_console_static_assets_load() -> None:
    css_status, css_headers, css_body = _get("/console/assets/console.css")
    js_status, js_headers, js_body = _get("/console/assets/console.js")

    assert css_status == 200
    assert css_headers["content-type"].startswith("text/css")
    assert b".conversation-stage" in css_body
    assert js_status == 200
    assert "javascript" in js_headers["content-type"]
    assert b"mountDeveloperView" in js_body


def test_console_chat_uses_voice_conversation_agent_response() -> None:
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")
    html = (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")
    paths = {route.path for route in main_module.app.routes}

    assert "/console/chat" not in paths
    assert 'api("/voice/conversation"' in script
    assert "result.agent_response?.user_response" in script
    assert script.index("result.agent_response?.user_response") < script.index("result.maya_response?.user_response")
    assert "/console/assets/console.js?v=0.13.0" in html
    assert "/console/assets/console.css?v=0.13.0" in html
    assert main_module.app.version == "0.13.0"


def test_console_has_real_push_to_talk_and_browser_tts_with_text_fallback() -> None:
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")
    html = (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="voice-button"' in html
    assert 'data-state="idle"' in html
    assert 'aria-pressed="false"' in html
    assert 'id="voice-status"' in html
    assert "window.SpeechRecognition || window.webkitSpeechRecognition" in script
    assert 'button.dataset.state = "unsupported"' in script
    assert "Voice input isn't supported here. Text chat still works." in script
    assert "SpeechSynthesisUtterance" in script
    assert "window.speechSynthesis.speak" in script
    assert "Listening now. Tap again to stop." in script


def test_push_to_talk_states_are_visually_distinct() -> None:
    css = (main_module.CONSOLE_DIR / "console.css").read_text(encoding="utf-8")

    for state in ('data-state="listening"', 'data-state="sending"', 'data-state="unsupported"'):
        assert state in css
    assert "voice-pulse" in css


def test_console_photo_capture_is_single_shot_and_user_initiated() -> None:
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")
    html = (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="photo-input"' in html
    assert 'type="file"' in html
    assert 'accept="image/*"' in html
    assert 'capture="environment"' in html
    assert 'api("/vision/photo"' not in script
    assert 'fetch(`/vision/photo?prompt=' in script
    assert 'byId("photo-input").addEventListener("change", handlePhotoCapture)' in script
    assert "getUserMedia" not in script
    assert "MediaRecorder" not in script
    assert "No live feed is active" in script
    assert "The image bytes were not retained" in script


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
    initial_load = script.rsplit("bindPersistentEvents();", 1)[1]

    assert 'method: "POST"' not in initial_load
    assert "generateBriefing" not in initial_load
    assert "runDream" not in initial_load
    assert "reviewDream" not in initial_load
    assert "reviewProcedure" not in initial_load
    assert "runSkill" not in initial_load
    assert "rememberContext" not in initial_load
    assert "searchContext" not in initial_load
    assert "Promise.allSettled([loadHealth(), loadStatusTokens()])" in initial_load


def test_console_does_not_expose_procedure_execution() -> None:
    paths = {route.path for route in main_module.app.routes}
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")

    assert "/procedures/execute" not in paths
    assert "/procedures/run" not in paths
    assert "/procedures/execute" not in script
    assert "/procedures/run" not in script


def test_console_developer_mode_mounts_and_unmounts_dense_panels() -> None:
    html = (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")

    assert 'id="developer-mode"' in html
    assert 'id="ambient-mode" type="checkbox" checked' in html
    assert 'ambient_mode: byId("ambient-mode").checked' in script
    assert '<template id="developer-template">' in html
    assert 'content.cloneNode(true)' in script
    assert 'root.replaceChildren(byId("developer-template").content.cloneNode(true))' in script
    assert 'byId("developer-root").replaceChildren()' in script
    assert "mountDeveloperView" in script
    assert "unmountDeveloperView" in script


def test_briefing_only_generates_after_explicit_trigger() -> None:
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")
    html = (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")

    assert '<div id="briefing-output" class="briefing-output" aria-live="polite"></div>' in html
    assert "No briefing generated" not in html
    assert 'if (view === "briefing") await openBriefing(true)' in script
    assert 'api("/maya/briefing"' in script
    assert "generateBriefing" not in script.rsplit("bindPersistentEvents();", 1)[1]


def test_status_tokens_require_real_backing_fields() -> None:
    script = (main_module.CONSOLE_DIR / "console.js").read_text(encoding="utf-8")

    for endpoint in ("/context", "/workspace", "/search/usage", "/dream/promotions", "/procedures/pending"):
        assert f'api("{endpoint}")' in script
    assert 'contextResult.status === "fulfilled"' in script
    assert 'workspaceResult.status === "fulfilled"' in script
    assert 'usageResult.status === "fulfilled"' in script
    assert 'promotionsResult.status === "fulfilled"' in script
    assert 'proceduresResult.status === "fulfilled"' in script
    assert 'target.hidden = tokens.length === 0' in script
    assert "statusToken(" not in (main_module.CONSOLE_DIR / "index.html").read_text(encoding="utf-8")


def test_console_source_has_no_static_capability_claims() -> None:
    source = "\n".join(
        (main_module.CONSOLE_DIR / filename).read_text(encoding="utf-8").lower()
        for filename in ("index.html", "console.js")
    )

    for suspicious_claim in ("integrated ", "connected to ", "i have access to"):
        assert suspicious_claim not in source


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
