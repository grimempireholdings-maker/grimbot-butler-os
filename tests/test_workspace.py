from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from grimbot_brain import main as main_module
from grimbot_brain.workspace import workspace_inspector as inspector_module
from grimbot_brain.workspace.workspace_inspector import (
    GIT_COMMANDS,
    MAX_DOC_PREVIEW,
    MAX_FILE_SIZE,
    MAX_LINE_LENGTH,
    MAX_SNIPPET_LENGTH,
    WorkspaceInspector,
)
from grimbot_brain.workspace.workspace_schemas import WorkspaceSearchRequest


def _init_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=path,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )


def test_workspace_endpoint_works_in_git_repo(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(main_module, "workspace", WorkspaceInspector(repo_root))

    result = main_module.workspace_get()
    paths = {route.path for route in main_module.app.routes}

    assert result.repo_name == "grimbot-butler-os"
    assert result.branch
    assert result.version
    assert len(result.recent_commits) <= 5
    assert "/workspace" in paths
    assert "/workspace/docs" in paths
    assert "/workspace/search" in paths


def test_workspace_handles_non_git_folder_gracefully(tmp_path, monkeypatch) -> None:
    (tmp_path / "README.md").write_text("Local notes", encoding="utf-8")

    def git_unavailable(args, **kwargs):
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="not a git repository")

    monkeypatch.setattr(inspector_module.subprocess, "run", git_unavailable)

    result = WorkspaceInspector(tmp_path).overview()

    assert result.repo_root == str(tmp_path.resolve())
    assert result.repo_name == tmp_path.name
    assert result.branch is None
    assert any("not inside" in warning.lower() for warning in result.warnings)


def test_git_inspection_uses_only_allowlist_and_shell_false(tmp_path, monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], dict]] = []

    def fake_run(args, **kwargs):
        command = tuple(args)
        calls.append((command, kwargs))
        outputs = {
            GIT_COMMANDS["root"]: str(tmp_path),
            GIT_COMMANDS["branch"]: "main",
            GIT_COMMANDS["status"]: "",
            GIT_COMMANDS["log"]: "abc123 Test commit",
        }
        return subprocess.CompletedProcess(args, 0, stdout=outputs[command], stderr="")

    monkeypatch.setattr(inspector_module.subprocess, "run", fake_run)

    result = WorkspaceInspector(tmp_path).overview()

    assert result.branch == "main"
    assert {call[0] for call in calls} <= set(GIT_COMMANDS.values())
    assert all(call[1]["shell"] is False for call in calls)
    assert all(call[1]["timeout"] > 0 for call in calls)
    assert all(call[1]["env"]["GIT_OPTIONAL_LOCKS"] == "0" for call in calls)


def test_top_level_items_come_from_tracked_git_index(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("tracked", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("untracked", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"],
        cwd=tmp_path,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )

    result = WorkspaceInspector(tmp_path).overview()
    names = {item.name for item in result.top_level_items}

    assert "tracked.txt" in names
    assert "untracked.txt" not in names


def test_docs_skip_env_secret_binary_and_large_files(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    exposed = "sk-or-v1-" + ("z" * 64)
    (tmp_path / "README.md").write_text(
        f"Readable project overview {exposed}",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("API_KEY=do-not-read", encoding="utf-8")
    (docs / "guide.md").write_text("Workspace guide " * 100, encoding="utf-8")
    (docs / "secret_notes.md").write_text("hidden", encoding="utf-8")
    (docs / "binary.txt").write_bytes(b"safe word\x00binary")
    (docs / "large.md").write_text("x" * (MAX_FILE_SIZE + 1), encoding="utf-8")
    for directory in (".git", ".venv", "__pycache__", "node_modules", "secret_folder"):
        excluded = docs / directory
        excluded.mkdir()
        (excluded / "hidden.md").write_text("must not be read", encoding="utf-8")

    documents = WorkspaceInspector(tmp_path).documents(repo_root=tmp_path)
    paths = {document.relative_path for document in documents}

    assert "README.md" in paths
    assert "docs/guide.md" in paths
    assert ".env" not in paths
    assert "docs/secret_notes.md" not in paths
    assert "docs/binary.txt" not in paths
    assert "docs/large.md" not in paths
    assert all("hidden.md" not in path for path in paths)
    assert all(len(document.preview) <= MAX_DOC_PREVIEW for document in documents)
    assert all("do-not-read" not in document.preview for document in documents)
    assert all(exposed not in document.preview for document in documents)
    assert "[REDACTED]" in next(
        document.preview for document in documents if document.relative_path == "README.md"
    )


def test_workspace_search_skips_secrets_binary_databases_and_excluded_directories(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "safe.py").write_text("WORKSPACE_NEEDLE = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("WORKSPACE_NEEDLE=secret", encoding="utf-8")
    (tmp_path / "api_token.txt").write_text("WORKSPACE_NEEDLE", encoding="utf-8")
    (tmp_path / "memory.sqlite3").write_text("WORKSPACE_NEEDLE", encoding="utf-8")
    (tmp_path / "binary.txt").write_bytes(b"WORKSPACE_NEEDLE\x00binary")
    for directory in (".git", ".venv", "__pycache__", "node_modules"):
        path = tmp_path / directory
        path.mkdir(exist_ok=True)
        (path / "hidden.py").write_text("WORKSPACE_NEEDLE", encoding="utf-8")

    result = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query="WORKSPACE_NEEDLE", max_results=20)
    )

    assert [match.relative_path for match in result.results] == ["safe.py"]
    assert all("secret" not in match.snippet.casefold() for match in result.results)


def test_workspace_search_caps_results_and_snippets(tmp_path) -> None:
    _init_repo(tmp_path)
    long_line = f"needle {'x' * 1000}\n"
    (tmp_path / "many.txt").write_text(long_line * 10, encoding="utf-8")

    result = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query="needle", max_results=3)
    )

    assert len(result.results) == 3
    assert result.truncated is True
    assert all(len(match.snippet) <= MAX_SNIPPET_LENGTH for match in result.results)


def test_workspace_search_caps_file_count(tmp_path, monkeypatch) -> None:
    _init_repo(tmp_path)
    test_cap = 5
    monkeypatch.setattr(inspector_module, "MAX_FILES_SCANNED", test_cap)
    for index in range(test_cap + 1):
        (tmp_path / f"file_{index:03}.txt").write_text("no match", encoding="utf-8")

    result = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query="needle", max_results=10)
    )

    assert result.files_scanned == test_cap
    assert result.truncated is True
    assert result.results == []


def test_workspace_search_skips_oversized_files_and_caps_line_processing(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "oversized.txt").write_text(
        "OVERSIZED_NEEDLE" + ("x" * MAX_FILE_SIZE),
        encoding="utf-8",
    )
    (tmp_path / "long_line.txt").write_text(
        ("x" * MAX_LINE_LENGTH) + "LINE_NEEDLE",
        encoding="utf-8",
    )

    oversized = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query="OVERSIZED_NEEDLE", max_results=10)
    )
    long_line = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query="LINE_NEEDLE", max_results=10)
    )

    assert oversized.results == []
    assert long_line.results == []


def test_workspace_search_skips_symlink_escape(tmp_path) -> None:
    _init_repo(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_text("SYMLINK_NEEDLE", encoding="utf-8")
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is not permitted on this Windows host")

    result = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query="SYMLINK_NEEDLE", max_results=5)
    )

    assert result.results == []


@pytest.mark.parametrize(
    "exposed",
    [
        "sk-or-v1-" + ("a" * 64),
        "sk-proj-" + ("b" * 32),
        "ghp_" + ("c" * 36),
        "AIza" + ("d" * 32),
        "AKIA" + ("E" * 16),
        "xoxb-" + ("f" * 32),
    ],
)
def test_workspace_search_redacts_api_key_patterns(tmp_path, exposed) -> None:
    _init_repo(tmp_path)
    exposed = "sk-or-v1-" + ("a" * 64)
    (tmp_path / "settings.py").write_text(
        f'provider_value = "{exposed}"  # workspace needle\n',
        encoding="utf-8",
    )

    result = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query="workspace needle", max_results=5)
    )

    assert len(result.results) == 1
    assert exposed not in result.results[0].snippet
    assert "[REDACTED]" in result.results[0].snippet

    query_result = WorkspaceInspector(tmp_path).search(
        WorkspaceSearchRequest(query=exposed, max_results=5)
    )
    assert exposed not in query_result.query
    assert query_result.query == "[REDACTED]"


def test_workspace_inspection_is_byte_for_byte_read_only(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("workspace needle", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    inspector = WorkspaceInspector(tmp_path)
    inspector.overview()
    inspector.documents()
    inspector.search(WorkspaceSearchRequest(query="workspace", max_results=5))
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_workspace_module_has_no_mutating_or_external_command_paths() -> None:
    source = (Path(inspector_module.__file__).read_text(encoding="utf-8")).lower()

    for forbidden in (
        "shell=true",
        "os.system",
        "popen(",
        "requests",
        "httpx",
        "git add",
        "git commit",
        "git push",
        "git checkout",
        ".write_text(",
        ".write_bytes(",
        ".unlink(",
        ".rename(",
        "import smtplib",
        "import requests",
        "import httpx",
    ):
        assert forbidden not in source


def test_git_allowlist_is_immutable() -> None:
    with pytest.raises(TypeError):
        GIT_COMMANDS["extra"] = ("git", "config", "--list")
