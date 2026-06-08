import json

from grimbot_brain.memory import BrainMemory
from grimbot_brain.room_scan import _extract_json_object, _unreadable_gemini_result, run_room_scan
from grimbot_brain.schemas import RoomScanRequest, RoomScanResult


def test_mock_room_scan_returns_structured_json_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory(tmp_path / "cycles.sqlite3")

    result = run_room_scan(
        RoomScanRequest(mock_camera_frame="laundry, dishes, and a loose cord"),
        memory,
    )
    payload = json.loads(result.model_dump_json())

    assert set(payload) == {
        "room_summary",
        "visible_objects",
        "mess_zones",
        "hazards",
        "suggested_cleanup_order",
        "next_best_action",
        "mode",
        "image_path",
    }
    assert payload["mode"] == "mock"
    assert "loose cord on floor" in payload["hazards"]
    assert payload["next_best_action"] == payload["suggested_cleanup_order"][0]


def test_room_scan_stores_result_in_memory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory(tmp_path / "cycles.sqlite3")

    run_room_scan(RoomScanRequest(mock_camera_frame="clear room"), memory)

    rows = memory.recent_room_scans(limit=1)
    assert len(rows) == 1
    assert rows[0]["scan_result"]["mode"] == "mock"


def test_room_scan_rejects_arbitrary_image_paths_and_uses_mock(tmp_path, monkeypatch) -> None:
    safe_dir = tmp_path / "safe-images"
    safe_dir.mkdir()
    outside_image = tmp_path / "outside.jpg"
    outside_image.write_bytes(b"fake image")
    monkeypatch.setenv("GRIMBOT_IMAGE_DIR", str(safe_dir))
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = run_room_scan(
        RoomScanRequest(image_path=str(outside_image), mock_camera_frame="dishes"),
        BrainMemory(tmp_path / "cycles.sqlite3"),
    )

    assert result.mode == "mock"
    assert result.image_path is None


def test_room_scan_result_trims_and_bounds_list_items() -> None:
    long_item = "  " + ("x" * 200) + "  "

    result = RoomScanResult(
        room_summary="Room",
        visible_objects=[long_item, "   "],
        mess_zones=[],
        hazards=[],
        suggested_cleanup_order=[long_item],
        next_best_action="inspect",
    )

    assert len(result.visible_objects) == 1
    assert len(result.visible_objects[0]) == 120
    assert result.visible_objects[0] == "x" * 120


def test_extract_json_object_handles_fenced_json() -> None:
    payload = _extract_json_object('```json\n{"room_summary":"Room"}\n```')

    assert payload == {"room_summary": "Room"}


def test_unreadable_gemini_result_keeps_structured_shape(tmp_path) -> None:
    result = _unreadable_gemini_result(tmp_path / "frame.jpg")
    payload = json.loads(result.model_dump_json())

    assert set(payload) == {
        "room_summary",
        "visible_objects",
        "mess_zones",
        "hazards",
        "suggested_cleanup_order",
        "next_best_action",
        "mode",
        "image_path",
    }
    assert payload["mode"] == "gemini"
    assert payload["next_best_action"] == "retry room scan"
