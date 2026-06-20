from __future__ import annotations

import json

import pytest

from grimbot_brain.conversation_agent import MockConversationProvider
from grimbot_brain.main import app
from grimbot_brain.memory import BrainMemory
from grimbot_brain.perception import analyze_user_photo
from grimbot_brain.photo_capture import process_photo_capture, validate_photo_upload
from grimbot_brain.schemas import PhotoAnalysisResult


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"safe-test-pixels"


def test_photo_capture_uses_gemini_result_then_deletes_bytes(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    monkeypatch.setenv("GRIMBOT_IMAGE_DIR", str(image_dir))
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    observed_paths = []

    def fake_analyzer(path, media_type, prompt):
        observed_paths.append(path)
        assert path.exists()
        assert path.read_bytes() == PNG_BYTES
        assert media_type == "image/png"
        assert prompt == "What is on the desk?"
        return PhotoAnalysisResult(
            description="A notebook and a blue mug are visible on the desk.",
            model="gemini-live-test",
            media_type=media_type,
        )

    result = process_photo_capture(
        PNG_BYTES,
        "image/png",
        "What is on the desk?",
        memory,
        analyzer=fake_analyzer,
        provider=MockConversationProvider(),
    )

    assert result.analysis.mode == "gemini"
    assert result.analysis.raw_media_stored is False
    assert result.agent_response.machine_output["vision_invoked"] is True
    assert result.agent_response.machine_output["search_triggered"] is False
    assert "notebook" in result.agent_response.user_response.lower()
    assert observed_paths and not observed_paths[0].exists()
    assert not list(image_dir.glob("user_photo_*"))


def test_photo_episode_contains_description_not_raw_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_IMAGE_DIR", str(tmp_path / "images"))
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    analysis = PhotoAnalysisResult(
        description="A green plant beside a window.",
        model="gemini-test",
        media_type="image/png",
    )
    process_photo_capture(
        PNG_BYTES,
        "image/png",
        "What do you see?",
        memory,
        analyzer=lambda *args: analysis,
        provider=MockConversationProvider(),
    )

    episode = memory.recent_episodes(1)[0]
    payload = json.loads(episode["content"])
    assert episode["kind"] == "photo_capture"
    assert payload["description"] == analysis.description
    assert payload["raw_media_stored"] is False
    assert "safe-test-pixels" not in episode["content"]
    assert "iVBOR" not in episode["content"]


def test_failed_photo_analysis_still_deletes_temporary_file(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    monkeypatch.setenv("GRIMBOT_IMAGE_DIR", str(image_dir))

    with pytest.raises(RuntimeError):
        process_photo_capture(
            PNG_BYTES,
            "image/png",
            "Look",
            BrainMemory(tmp_path / "memory.sqlite3"),
            analyzer=lambda *args: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
        )

    assert not list(image_dir.glob("user_photo_*"))


@pytest.mark.parametrize(
    ("data", "media_type"),
    [
        (b"not an image", "image/png"),
        (PNG_BYTES, "text/plain"),
        (b"", "image/png"),
    ],
)
def test_photo_upload_rejects_invalid_content(data, media_type) -> None:
    with pytest.raises(ValueError):
        validate_photo_upload(data, media_type)


def test_real_gemini_photo_path_never_returns_mock(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "photo.png"
    image.write_bytes(PNG_BYTES)
    monkeypatch.setenv("GRIMBOT_IMAGE_DIR", str(image_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_VISION_MODEL", "gemini-test-vision")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "candidates": [{"content": {"parts": [{"text": "A real Gemini visual description."}]}}]
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("grimbot_brain.perception.urllib.request.urlopen", fake_urlopen)
    result = analyze_user_photo(image, "image/png", "What is shown?")

    assert result.mode == "gemini"
    assert result.description == "A real Gemini visual description."
    assert result.model == "gemini-test-vision"
    assert captured["timeout"] == 30
    assert captured["payload"]["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/png"


def test_openrouter_fallback_still_pins_real_gemini_vision(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "photo.png"
    image.write_bytes(PNG_BYTES)
    monkeypatch.setenv("GRIMBOT_IMAGE_DIR", str(image_dir))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "A Gemini visual description."}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr("grimbot_brain.perception.urllib.request.urlopen", fake_urlopen)
    result = analyze_user_photo(image, "image/png", "What is shown?")

    assert result.mode == "gemini"
    assert result.model == "google/gemini-2.5-flash-lite"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["payload"]["model"].startswith("google/gemini-")
    assert captured["payload"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_photo_route_exists_without_procedure_or_hardware_side_effects() -> None:
    paths = {route.path for route in app.routes}

    assert "/vision/photo" in paths
    assert "/procedures/execute" not in paths
