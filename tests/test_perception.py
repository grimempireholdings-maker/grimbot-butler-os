from grimbot_brain.perception import _safe_image_path


def test_safe_image_path_rejects_non_images(tmp_path) -> None:
    text_file = tmp_path / "secret.txt"
    text_file.write_text("not an image")

    assert _safe_image_path(str(text_file)) is None


def test_safe_image_path_accepts_image_extensions(tmp_path) -> None:
    image_file = tmp_path / "frame.jpg"
    image_file.write_bytes(b"fake image bytes")

    assert _safe_image_path(str(image_file)) == image_file
