from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_IMAGE_DIR = Path("vision/images")


def image_directory() -> Path:
    directory = Path(os.getenv("GRIMBOT_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def approved_image_path(image_path: str | None, image_dir: str | Path | None = None) -> Path | None:
    if not image_path:
        return None

    safe_dir = Path(image_dir).expanduser() if image_dir else image_directory()
    safe_root = safe_dir.resolve()
    path = Path(image_path).expanduser()

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return None

    if not resolved.is_file():
        return None

    if resolved.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return None

    try:
        resolved.relative_to(safe_root)
    except ValueError:
        return None

    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        return None

    return resolved


def capture_webcam_frame(camera_index: int = 0, image_dir: str | Path | None = None) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for webcam capture. Install with .[vision].") from exc

    output_dir = Path(image_dir).expanduser() if image_dir else image_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("room_scan_%Y%m%d_%H%M%S_%f.jpg")
    output_path = output_dir / filename

    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError("Webcam is not available")

        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Could not capture webcam frame")

        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError("Could not save webcam frame")
    finally:
        camera.release()

    approved = approved_image_path(str(output_path), image_dir=output_dir)
    if not approved:
        raise RuntimeError("Captured frame failed image safety validation")

    return approved
