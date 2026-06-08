from grimbot_brain.vision import approved_image_path, image_directory


def test_image_directory_uses_env_and_creates_directory(tmp_path, monkeypatch) -> None:
    target = tmp_path / "vision-images"
    monkeypatch.setenv("GRIMBOT_IMAGE_DIR", str(target))

    assert image_directory() == target
    assert target.exists()


def test_approved_image_path_requires_safe_directory(tmp_path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    image = safe_dir / "frame.png"
    image.write_bytes(b"fake image")

    approved = approved_image_path(str(image), image_dir=safe_dir)

    assert approved == image
