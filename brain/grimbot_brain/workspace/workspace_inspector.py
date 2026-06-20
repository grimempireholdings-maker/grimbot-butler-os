from __future__ import annotations

import os
import re
import struct
import subprocess
import tomllib
from pathlib import Path
from types import MappingProxyType

from .workspace_schemas import (
    WorkspaceDocument,
    WorkspaceItem,
    WorkspaceOverview,
    WorkspaceSearchMatch,
    WorkspaceSearchRequest,
    WorkspaceSearchResult,
)

GIT_COMMANDS = MappingProxyType({
    "root": ("git", "rev-parse", "--show-toplevel"),
    "branch": ("git", "branch", "--show-current"),
    "status": ("git", "status", "--short"),
    "log": ("git", "log", "--oneline", "-5"),
})

MAX_FILE_SIZE = 256_000
MAX_FILES_SCANNED = 250
MAX_DOC_FILES_SCANNED = 250
MAX_LINE_LENGTH = 10_000
MAX_SNIPPET_LENGTH = 240
MAX_DOC_PREVIEW = 400
MAX_GIT_INDEX_SIZE = 8_000_000
GIT_TIMEOUT_SECONDS = 3

_EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
_MEMORY_SUFFIXES = {".db", ".db3", ".sqlite", ".sqlite3"}
_SAFE_TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_DOC_SUFFIXES = {".md", ".rst", ".txt"}
_SECRET_NAME_PARTS = {
    "api_key",
    "apikey",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
}


class WorkspaceInspector:
    def __init__(self, start_path: str | Path | None = None) -> None:
        self.start_path = Path(start_path or Path.cwd()).resolve()

    def overview(self) -> WorkspaceOverview:
        warnings: list[str] = []
        repo_root = self._repo_root(warnings)
        branch = self._git_value("branch", warnings)
        status_summary = self._git_lines("status", warnings, limit=100)
        recent_commits = self._git_lines("log", warnings, limit=5)
        documents = self.documents(repo_root=repo_root)
        return WorkspaceOverview(
            repo_root=str(repo_root),
            repo_name=repo_root.name or str(repo_root),
            branch=branch or None,
            status_summary=status_summary,
            recent_commits=recent_commits,
            version=self._detect_version(repo_root, warnings),
            top_level_items=self._top_level_items(repo_root, warnings),
            docs_detected=[document.relative_path for document in documents],
            warnings=_dedupe(warnings),
        )

    def documents(self, repo_root: Path | None = None) -> list[WorkspaceDocument]:
        root = repo_root or self._repo_root([])
        documents: list[WorkspaceDocument] = []
        for path in self._document_candidates(root):
            if not self._safe_text_file(path, root, doc_only=True):
                continue
            preview = self._preview(path)
            if preview is None:
                continue
            documents.append(
                WorkspaceDocument(
                    filename=path.name,
                    relative_path=path.relative_to(root).as_posix(),
                    preview=preview,
                )
            )
            if len(documents) >= 100:
                break
        return documents

    def search(self, request: WorkspaceSearchRequest) -> WorkspaceSearchResult:
        warnings: list[str] = []
        root = self._repo_root(warnings)
        query = request.query.casefold()
        matches: list[WorkspaceSearchMatch] = []
        files_scanned = 0
        truncated = False

        for path in self._safe_project_files(root):
            if files_scanned >= MAX_FILES_SCANNED:
                warnings.append(f"Search stopped after {MAX_FILES_SCANNED} files.")
                truncated = True
                break
            files_scanned += 1
            try:
                with path.open("r", encoding="utf-8", errors="strict") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        bounded_line = line[:MAX_LINE_LENGTH]
                        if query not in bounded_line.casefold():
                            continue
                        matches.append(
                            WorkspaceSearchMatch(
                                relative_path=path.relative_to(root).as_posix(),
                                line_number=line_number,
                                snippet=_snippet(bounded_line),
                            )
                        )
                        if len(matches) >= request.max_results:
                            truncated = True
                            break
            except (OSError, UnicodeError):
                continue
            if len(matches) >= request.max_results:
                break

        return WorkspaceSearchResult(
            query=_redact_secrets(request.query),
            results=matches,
            files_scanned=files_scanned,
            truncated=truncated,
            warnings=_dedupe(warnings),
        )

    def _repo_root(self, warnings: list[str]) -> Path:
        value = self._git_value("root", warnings)
        if value:
            candidate = Path(value).resolve()
            if candidate.is_dir():
                return candidate
        warnings.append("Current directory is not inside an available Git repository.")
        return self.start_path

    def _git_value(self, command_name: str, warnings: list[str]) -> str:
        lines = self._git_lines(command_name, warnings, limit=1)
        return lines[0] if lines else ""

    def _git_lines(self, command_name: str, warnings: list[str], limit: int) -> list[str]:
        args = GIT_COMMANDS[command_name]
        try:
            result = subprocess.run(
                args,
                cwd=self.start_path,
                shell=False,
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except FileNotFoundError:
            warnings.append("Git is not installed or not available on PATH.")
            return []
        except subprocess.TimeoutExpired:
            warnings.append(f"Git inspection timed out: {' '.join(args[1:])}.")
            return []
        except OSError:
            warnings.append("Git inspection was unavailable.")
            return []
        if result.returncode != 0:
            if command_name != "root":
                warnings.append(f"Git inspection unavailable: {' '.join(args[1:])}.")
            return []
        lines = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            if command_name == "status" and _is_secret_path_text(line[3:]):
                continue
            lines.append(_redact_secrets(line)[:500])
        return lines[:limit]

    def _top_level_items(self, root: Path, warnings: list[str]) -> list[WorkspaceItem]:
        tracked_names = self._tracked_top_level_names(root)
        if tracked_names is None:
            warnings.append("Tracked-file index unavailable; using a safe top-level filesystem listing.")
            return self._filesystem_top_level_items(root)
        items: list[WorkspaceItem] = []
        for name in sorted(tracked_names, key=str.casefold):
            if _is_secret_name(name):
                continue
            path = root / name
            items.append(
                WorkspaceItem(
                    name=name,
                    relative_path=name,
                    kind="directory" if path.is_dir() else "file",
                )
            )
            if len(items) >= 100:
                break
        return items

    def _filesystem_top_level_items(self, root: Path) -> list[WorkspaceItem]:
        items: list[WorkspaceItem] = []
        try:
            paths = sorted(root.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError:
            return items
        for path in paths:
            if path.name in _EXCLUDED_DIRECTORIES or _is_secret_name(path.name):
                continue
            items.append(
                WorkspaceItem(
                    name=path.name,
                    relative_path=path.relative_to(root).as_posix(),
                    kind="directory" if path.is_dir() else "file",
                )
            )
            if len(items) >= 100:
                break
        return items

    def _tracked_top_level_names(self, root: Path) -> set[str] | None:
        git_marker = root / ".git"
        if git_marker.is_dir():
            git_dir = git_marker
        elif git_marker.is_file():
            try:
                marker = git_marker.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                return None
            if not marker.lower().startswith("gitdir:"):
                return None
            git_dir = (root / marker.split(":", 1)[1].strip()).resolve()
        else:
            return None

        index_path = git_dir / "index"
        try:
            if not index_path.is_file() or index_path.stat().st_size > MAX_GIT_INDEX_SIZE:
                return None
            data = index_path.read_bytes()
        except OSError:
            return None
        if len(data) < 12 or data[:4] != b"DIRC":
            return None
        version, entry_count = struct.unpack(">II", data[4:12])
        if version not in {2, 3} or entry_count > 500_000:
            return None

        names: set[str] = set()
        offset = 12
        try:
            for _ in range(entry_count):
                entry_start = offset
                flags = struct.unpack(">H", data[offset + 60 : offset + 62])[0]
                offset += 62
                if flags & 0x4000:
                    offset += 2
                end = data.index(b"\x00", offset)
                relative_path = data[offset:end].decode("utf-8", errors="strict")
                top_name = relative_path.replace("\\", "/").split("/", 1)[0]
                if top_name:
                    names.add(top_name)
                offset = end + 1
                while (offset - entry_start) % 8:
                    offset += 1
        except (IndexError, struct.error, UnicodeError, ValueError):
            return None
        return names

    def _detect_version(self, root: Path, warnings: list[str]) -> str | None:
        pyproject = root / "pyproject.toml"
        if pyproject.is_file() and pyproject.stat().st_size <= MAX_FILE_SIZE:
            try:
                with pyproject.open("rb") as handle:
                    data = tomllib.load(handle)
                version = data.get("project", {}).get("version")
                if isinstance(version, str) and version.strip():
                    return version.strip()[:80]
            except (OSError, tomllib.TOMLDecodeError):
                warnings.append("Could not parse project version from pyproject.toml.")

        candidates = (
            root / "brain" / "grimbot_brain" / "__init__.py",
            root / "grimbot_brain" / "__init__.py",
            root / "__init__.py",
        )
        pattern = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
        for path in candidates:
            if not path.is_file() or path.stat().st_size > MAX_FILE_SIZE:
                continue
            try:
                match = pattern.search(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                continue
            if match:
                return match.group(1)[:80]
        return None

    def _safe_project_files(self, root: Path):
        for current_root, directory_names, file_names in os.walk(root):
            current = Path(current_root)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in _EXCLUDED_DIRECTORIES
                and not _is_secret_name(name)
                and not (current / name).is_symlink()
            )
            for file_name in sorted(file_names):
                path = current / file_name
                if self._safe_text_file(path, root):
                    yield path

    def _document_candidates(self, root: Path):
        try:
            top_level = sorted(root.iterdir(), key=lambda path: path.name.casefold())
        except OSError:
            top_level = []
        for path in top_level:
            if path.is_file() and path.suffix.lower() in _DOC_SUFFIXES:
                yield path

        docs_dir = root / "docs"
        if not docs_dir.is_dir() or docs_dir.is_symlink():
            return
        files_seen = 0
        for current_root, directory_names, file_names in os.walk(docs_dir):
            current = Path(current_root)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in _EXCLUDED_DIRECTORIES
                and not _is_secret_name(name)
                and not (current / name).is_symlink()
            )
            for file_name in sorted(file_names):
                files_seen += 1
                if files_seen > MAX_DOC_FILES_SCANNED:
                    return
                yield current / file_name

    def _safe_text_file(self, path: Path, root: Path, doc_only: bool = False) -> bool:
        try:
            relative_parts = path.relative_to(root).parts
            if any(part in _EXCLUDED_DIRECTORIES for part in relative_parts[:-1]):
                return False
            if any(_is_secret_name(part) for part in relative_parts):
                return False
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                return False
            suffix = path.suffix.lower()
            allowed = _DOC_SUFFIXES if doc_only else _SAFE_TEXT_SUFFIXES
            if suffix not in allowed or suffix in _MEMORY_SUFFIXES:
                return False
            size = path.stat().st_size
            if size > MAX_FILE_SIZE:
                return False
            with path.open("rb") as handle:
                sample = handle.read(8192)
            return b"\x00" not in sample
        except (OSError, ValueError):
            return False

    def _preview(self, path: Path) -> str | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        compact = " ".join(text.split())
        return _redact_secrets(compact)[:MAX_DOC_PREVIEW]


def _is_secret_name(name: str) -> bool:
    if _redact_secrets(name) != name:
        return True
    lowered = name.casefold()
    if lowered == ".env" or lowered.startswith(".env."):
        return True
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered)
    return any(part in normalized for part in _SECRET_NAME_PARTS)


def _snippet(line: str) -> str:
    compact = _redact_secrets(" ".join(line.split()))
    if not compact:
        return "(blank matching line)"
    return compact[:MAX_SNIPPET_LENGTH]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _is_secret_path_text(value: str) -> bool:
    return any(_is_secret_name(part) for part in re.split(r"[\\/]", value.strip()))


def _redact_secrets(value: str) -> str:
    patterns = (
        r"sk-or-v1-[A-Za-z0-9_-]+",
        r"sk-[A-Za-z0-9_-]{20,}",
        r"gh[pousr]_[A-Za-z0-9_]{20,}",
        r"AIza[A-Za-z0-9_-]{20,}",
        r"AKIA[A-Z0-9]{16}",
        r"xox[baprs]-[A-Za-z0-9-]{20,}",
    )
    redacted = value
    for pattern in patterns:
        redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted
