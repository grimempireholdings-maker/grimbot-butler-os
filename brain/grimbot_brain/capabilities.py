from __future__ import annotations

import json
from types import MappingProxyType

_CAPABILITY_DATA = {
    "has_camera_access": True,
    "camera_scope": "user-initiated single-photo capture only — not continuous video, not always-watching, no background capture",
    "has_continuous_video_access": False,
    "has_device_layout_awareness": False,
    "has_screen_or_tab_awareness": False,
    "has_microphone_access": True,
    "microphone_scope": "user-initiated browser push-to-talk only — not always-listening, no background recording",
    "has_browser_text_to_speech": True,
    "browser_text_to_speech_scope": "browser speech synthesis for replies to voice-originated turns only",
    "has_always_listening_access": False,
    "has_workspace_read_access": True,
    "workspace_read_scope": "local repo/filesystem, read-only",
    "can_modify_workspace_files": False,
    "has_robot_body": False,
    "has_physical_room_sensors": False,
    "can_execute_procedures": False,
    "has_external_tools": True,
    "external_tools_scope": "read-only Tavily web search only",
    "has_unrestricted_internet_access": False,
    "has_web_search": True,
    "web_search_scope": "live search snippets only — no page scraping beyond what the search API returns, no browsing, no following links",
    "has_news_access": True,
    "has_weather_access": True,
    "has_real_time_market_data": False,
    "external_data_scope": "classifier-authorized Tavily search snippets only",
    "memory_tiers_active": ("episodic", "semantic", "procedural"),
    "dreaming_active": True,
    "dreaming_scope": "manual, human-reviewed reflection only",
}

CAPABILITIES = MappingProxyType(_CAPABILITY_DATA)


def capabilities_manifest() -> dict:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in CAPABILITIES.items()
    }


def capabilities_prompt_block() -> str:
    return json.dumps(capabilities_manifest(), ensure_ascii=True, sort_keys=True, indent=2)
