from grimbot_brain.maya_core import build_maya_briefing
from grimbot_brain.memory import BrainMemory
from grimbot_brain.persona import resolve_permission
from grimbot_brain.response_composer import compose_maya_response
from grimbot_brain.robot_memory import RobotMemory
from grimbot_brain.room_scan import run_room_scan
from grimbot_brain.safety import validate_action
from grimbot_brain.schemas import (
    BrainCycleInput,
    MayaBriefingRequest,
    MayaComposeRequest,
    RobotIntent,
    RoomScanRequest,
)


def test_maya_response_tone_is_warm_direct_and_concise() -> None:
    response = compose_maya_response(
        MayaComposeRequest(
            raw_output={
                "next_best_action": "clear hazard: loose cord on floor",
            },
            response_mode="cleanup_coaching",
            verified=False,
        )
    )

    assert response.user_response.startswith("Not verified yet. First:")
    assert "clear hazard: loose cord on floor" in response.user_response
    assert "fluff" not in response.user_response.lower()


def test_machine_output_stays_separate_from_user_facing_text() -> None:
    raw_command = {"action": "stop", "speed": 0, "reason": "Obstacle too close"}

    response = compose_maya_response(
        MayaComposeRequest(raw_output=raw_command, verified=True)
    )

    assert response.machine_output == raw_command
    assert response.user_response != str(raw_command)
    assert "Obstacle too close" in response.user_response


def test_safety_override_still_wins_with_maya_response() -> None:
    cycle_input = BrainCycleInput(battery_percentage=80, distance_cm=5, user_command="move forward")
    intent = RobotIntent(
        requested_action="move_forward",
        requested_speed=0.25,
        reason="Maya context says hallway is usually clear",
    )
    command = validate_action(cycle_input, intent)
    response = compose_maya_response(
        MayaComposeRequest(raw_output=command.model_dump(), verified=True)
    )

    assert command.action == "stop"
    assert response.machine_output["action"] == "stop"
    assert response.user_response.startswith("Verified. Safety wins:")


def test_maya_briefing_structure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    brain_memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(
            room_name="Office",
            zone_name="Desk",
            mock_camera_frame="notebooks, drink containers, and a loose cable",
        ),
        brain_memory,
    )

    briefing = build_maya_briefing(
        MayaBriefingRequest(room_name="Office", verified=False),
        RobotMemory(brain_memory),
    )

    assert briefing.priority_items
    assert briefing.fyi
    assert briefing.hazards == ["loose cord on floor"]
    assert briefing.next_best_action == "clear hazard: loose cord on floor"
    assert briefing.permission == "suggest"


def test_verified_and_unverified_language_behavior() -> None:
    unverified = compose_maya_response(
        MayaComposeRequest(raw_output={"next_best_action": "scan room"}, verified=False)
    )
    verified = compose_maya_response(
        MayaComposeRequest(raw_output={"next_best_action": "scan room"}, verified=True)
    )

    assert unverified.user_response.startswith("Not verified yet.")
    assert verified.user_response.startswith("Verified.")


def test_permission_logic_asks_approval_for_unverified_execute() -> None:
    assert resolve_permission("maya_chief_of_staff", "execute", verified=False) == "ask_approval"
    assert resolve_permission("quiet_observer", "execute", verified=True) == "observe"


def test_neutral_robot_does_not_escalate_to_execute() -> None:
    assert resolve_permission("neutral_robot", "execute", verified=True) == "suggest"
    assert resolve_permission("neutral_robot", "execute", verified=False) == "ask_approval"


def test_unverified_response_strips_raw_verified_claim() -> None:
    response = compose_maya_response(
        MayaComposeRequest(
            raw_output={"room_summary": "Verified: the kitchen is clear."},
            verified=False,
        )
    )

    assert response.user_response.startswith("Not verified yet.")
    assert "Verified: the kitchen is clear" not in response.user_response


def test_long_raw_output_is_bounded_in_user_response() -> None:
    response = compose_maya_response(
        MayaComposeRequest(
            raw_output={"room_summary": "x" * 5000},
            verified=True,
        )
    )

    assert len(response.user_response) < 500
    assert response.user_response.endswith("...")


def test_quiet_observer_never_recommends_execution() -> None:
    response = compose_maya_response(
        MayaComposeRequest(
            raw_output={"next_best_action": "execute cleanup task"},
            mode="quiet_observer",
            requested_permission="execute",
            verified=True,
        )
    )

    assert response.permission == "observe"
    assert response.user_response == "Verified. Observed. No action taken."


def test_empty_maya_briefing_has_stable_structure(tmp_path) -> None:
    briefing = build_maya_briefing(
        MayaBriefingRequest(room_name="Unknown Room", verified=False),
        RobotMemory(BrainMemory(tmp_path / "memory.sqlite3")),
    )

    assert briefing.priority_items == ["scan room for current conditions"]
    assert briefing.fyi == ["No recurring room context yet."]
    assert briefing.wins == []
    assert briefing.hazards == []
    assert briefing.next_best_action == "scan room for current conditions"
