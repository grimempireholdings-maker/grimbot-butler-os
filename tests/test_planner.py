from grimbot_brain.planner import plan
from grimbot_brain.schemas import BrainCycleInput, PerceptionResult


def test_planner_uses_whole_words_for_go_command() -> None:
    cycle_input = BrainCycleInput(
        battery_percentage=80,
        distance_cm=100,
        user_command="diagnostic mode",
    )
    perception = PerceptionResult(
        mode="mock",
        scene_summary="Path appears clear",
        obstacle_detected=False,
        confidence=0.75,
    )

    intent = plan(cycle_input, perception)

    assert intent.requested_action == "idle"


def test_planner_stops_forward_intent_when_perception_reports_obstacle() -> None:
    cycle_input = BrainCycleInput(
        battery_percentage=80,
        distance_cm=40,
        user_command="go ahead",
    )
    perception = PerceptionResult(
        mode="mock",
        scene_summary="Obstacle detected",
        obstacle_detected=True,
        obstacle_distance_cm=40,
        confidence=0.75,
    )

    intent = plan(cycle_input, perception)

    assert intent.requested_action == "stop"
    assert intent.requested_speed == 0
