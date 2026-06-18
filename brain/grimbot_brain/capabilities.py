from __future__ import annotations

import json
from types import MappingProxyType

_CAPABILITY_DATA = {
    "has_camera_access": False,
    "has_device_layout_awareness": False,
    "has_screen_or_tab_awareness": False,
    "has_microphone_access": False,
    "has_workspace_read_access": True,
    "workspace_read_scope": "local repo/filesystem, read-only",
    "can_modify_workspace_files": False,
    "has_robot_body": False,
    "has_physical_room_sensors": False,
    "can_execute_procedures": False,
    "has_external_tools": False,
    "has_internet_access": False,
    "has_news_access": False,
    "has_weather_access": False,
    "has_real_time_market_data": False,
    "external_data_scope": "none — all knowledge is from local memory, workspace, and stored context only",
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
