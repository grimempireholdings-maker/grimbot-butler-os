from __future__ import annotations

import argparse
import json
import os
import sys

from .conversation import run_voice_conversation
from .memory import BrainMemory
from .schemas import VoiceConversationRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a push-to-talk GrimBot voice conversation.")
    parser.add_argument("--push-to-talk", action="store_true")
    parser.add_argument("--mock-transcript", default="How's my day looking?")
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--room-name", default=None)
    parser.add_argument("--zone-name", default=None)
    args = parser.parse_args()

    os.environ.setdefault("GRIMBOT_VOICE_MOCK", "true")
    try:
        result = run_voice_conversation(
            VoiceConversationRequest(
                push_to_talk=args.push_to_talk,
                mock_transcript=args.mock_transcript,
                audio_path=args.audio_path,
                room_name=args.room_name,
                zone_name=args.zone_name,
            ),
            BrainMemory(),
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")))
        sys.exit(2)

    print(json.dumps(result.model_dump(), separators=(",", ":")))


if __name__ == "__main__":
    main()
