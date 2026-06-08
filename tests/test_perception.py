from grimbot_brain.vision import approved_image_path


def test_safe_image_path_rejects_non_images(tmp_path) -> None:
    text_file = tmp_path / "secret.txt"
    text_file.write_text("not an image")

    assert approved_image_path(str(text_file), image_dir=tmp_path) is None


def test_safe_image_path_accepts_image_extensions(tmp_path) -> None:
    image_file = tmp_path / "frame.jpg"
    image_file.write_bytes(b"fake image bytes")

    assert approved_image_path(str(image_file), image_dir=tmp_path) == image_file


def test_safe_image_path_rejects_images_outside_safe_directory(tmp_path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    outside_image = tmp_path / "outside.jpg"
    outside_image.write_bytes(b"fake image bytes")

    assert approved_image_path(str(outside_image), image_dir=safe_dir) is None
