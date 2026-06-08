from __future__ import annotations

import argparse
import json
import os

from .memory import BrainMemory
from .room_scan import run_room_scan
from .schemas import RoomScanRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GrimBot room scan.")
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--mock-camera-frame", default=None)
    parser.add_argument("--capture-webcam", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("GRIMBOT_MOCK_PERCEPTION", "true")
    request = RoomScanRequest(
        image_path=args.image_path,
        mock_camera_frame=args.mock_camera_frame,
        capture_webcam=args.capture_webcam,
        camera_index=args.camera_index,
    )
    result = run_room_scan(request, BrainMemory())
    print(json.dumps(result.model_dump(), separators=(",", ":")))


if __name__ == "__main__":
    main()
