from __future__ import annotations

from .schemas import AssistantMode, PermissionLevel

MAYA_DIRECTIVES = [
    "Protect the Asset",
    "Buy Back Time",
    "Ensure Profitability",
    "Verify before acting",
    "Clarity over cleverness",
]


def directives_for_mode(mode: AssistantMode) -> list[str]:
    if mode == "maya_chief_of_staff":
        return MAYA_DIRECTIVES.copy()
    if mode == "neutral_robot":
        return ["Verify before acting", "Clarity over cleverness"]
    return ["Observe only", "Verify before acting"]


def resolve_permission(mode: AssistantMode, requested: PermissionLevel, verified: bool) -> PermissionLevel:
    if mode == "quiet_observer":
        return "observe"

    if mode == "neutral_robot" and requested == "execute":
        return "suggest" if verified else "ask_approval"

    if requested == "execute" and not verified:
        return "ask_approval"

    if requested == "execute":
        return "execute"

    if requested == "ask_approval":
        return "ask_approval"

    if mode == "neutral_robot":
        return "suggest" if requested == "suggest" else requested

    return requested


def verification_phrase(verified: bool) -> str:
    if verified:
        return "Verified."
    return "Not verified yet."
