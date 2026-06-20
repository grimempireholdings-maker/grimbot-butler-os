from grimbot_brain.memory import BrainMemory
from grimbot_brain.robot_memory import RobotMemory
from grimbot_brain.room_scan import run_room_scan
from grimbot_brain.safety import validate_action
from grimbot_brain.schemas import BrainCycleInput, RememberRequest, RobotIntent, RoomScanRequest, SkillRunRequest
from grimbot_brain.skills import ChecklistBuilderSkill, SkillRegistry, create_default_registry


def test_skill_registration_and_listing(tmp_path) -> None:
    registry = SkillRegistry(BrainMemory(tmp_path / "memory.sqlite3"))

    registry.register(ChecklistBuilderSkill())

    skills = registry.list_skills()
    assert len(skills) == 1
    assert skills[0].name == "checklist_builder"
    assert skills[0].required_permission == "suggest"


def test_skill_lookup_by_name_and_category(tmp_path) -> None:
    registry = create_default_registry(BrainMemory(tmp_path / "memory.sqlite3"))

    skill = registry.get("room_cleanup_plan")
    skill_case = registry.get(" Room_Cleanup_Plan ")
    planning = registry.find_by_category(" Planning ")

    assert skill is not None
    assert skill_case is not None
    assert skill.info().category == "planning"
    assert "room_cleanup_plan" in [item.name for item in planning]


def test_web_search_skill_is_registered_as_observe_only_research(tmp_path) -> None:
    registry = create_default_registry(BrainMemory(tmp_path / "memory.sqlite3"))

    skill = registry.get("web_search")

    assert skill is not None
    assert skill.info().category == "research"
    assert skill.info().required_permission == "observe"


def test_permission_gating_blocks_higher_permission_skill(tmp_path) -> None:
    registry = create_default_registry(BrainMemory(tmp_path / "memory.sqlite3"))

    result = registry.run(
        "task_breakdown",
        SkillRunRequest(inputs={"task": "organize the office"}, permission="suggest"),
    )

    assert result.allowed is False
    assert result.permission == "suggest"
    assert result.machine_output["status"] == "blocked"
    assert result.machine_output["data"]["required_permission"] == "ask_approval"
    assert result.skill.required_permission == "ask_approval"


def test_skill_outputs_are_structured(tmp_path) -> None:
    registry = create_default_registry(BrainMemory(tmp_path / "memory.sqlite3"))

    result = registry.run(
        "checklist_builder",
        SkillRunRequest(inputs={"goal": "reset the desk"}, permission="suggest"),
    )

    assert result.allowed is True
    assert result.machine_output["skill"] == "checklist_builder"
    assert result.machine_output["status"] == "ok"
    assert isinstance(result.machine_output["data"]["checklist"], list)
    assert result.maya_response.machine_output == result.machine_output


def test_maya_skill_suggestion_for_cleanup_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="notebooks and loose cable"),
        memory,
    )

    result = create_default_registry(memory).run(
        "room_cleanup_plan",
        SkillRunRequest(inputs={"room_name": "Office", "zone_name": "Desk"}, permission="suggest"),
    )

    assert result.allowed is True
    assert result.skill.required_permission == "suggest"
    assert result.permission == "suggest"
    assert "Boss, I can run the room cleanup planning skill" in result.maya_response.user_response
    assert "clear hazard: loose cord on floor" in result.maya_response.user_response


def test_memory_backed_cleanup_planning_uses_hazards_and_mess(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GRIMBOT_MOCK_PERCEPTION", "true")
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    run_room_scan(
        RoomScanRequest(room_name="Office", zone_name="Desk", mock_camera_frame="notebooks, drink, cable"),
        memory,
    )

    result = create_default_registry(memory).run(
        "room_cleanup_plan",
        SkillRunRequest(inputs={"room_name": "Office"}, permission="suggest"),
    )

    assert result.machine_output["next_best_action"] == "clear hazard: loose cord on floor"
    assert any("notebooks on desk" in step for step in result.machine_output["data"]["plan"])


def test_memory_review_summarizes_robot_memory(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    RobotMemory(memory).remember(
        RememberRequest(text="The desk often has cables.", room_name="Office", zone_name="Desk")
    )

    result = create_default_registry(memory).run(
        "memory_review",
        SkillRunRequest(inputs={"room_name": "Office"}, permission="observe"),
    )

    assert result.allowed is True
    assert result.machine_output["skill"] == "memory_review"
    assert result.machine_output["data"]["semantic_facts"]


def test_safety_still_overrides_action_like_skill_request(tmp_path) -> None:
    memory = BrainMemory(tmp_path / "memory.sqlite3")
    result = create_default_registry(memory).run(
        "task_breakdown",
        SkillRunRequest(inputs={"task": "move forward"}, permission="ask_approval"),
    )
    cycle_input = BrainCycleInput(battery_percentage=80, distance_cm=5, user_command="move forward")
    intent = RobotIntent(
        requested_action="move_forward",
        requested_speed=0.25,
        reason=result.machine_output["next_best_action"],
    )

    command = validate_action(cycle_input, intent)

    assert result.allowed is True
    assert command.action == "stop"
    assert command.reason == "Obstacle too close"


def test_empty_memory_cleanup_plan_has_safe_fallback(tmp_path) -> None:
    result = create_default_registry(BrainMemory(tmp_path / "memory.sqlite3")).run(
        "room_cleanup_plan",
        SkillRunRequest(inputs={"room_name": "Unknown"}, permission="suggest"),
    )

    assert result.allowed is True
    assert result.machine_output["next_best_action"] == "scan room for current conditions"
    assert result.machine_output["data"]["plan"] == ["Scan the room before planning cleanup."]


def test_successful_skill_outputs_use_stable_envelope(tmp_path) -> None:
    registry = create_default_registry(BrainMemory(tmp_path / "memory.sqlite3"))

    result = registry.run(
        "checklist_builder",
        SkillRunRequest(inputs={"goal": "prepare room"}, permission="suggest"),
    )

    assert set(result.machine_output) == {"skill", "status", "next_best_action", "data"}
    assert result.machine_output["status"] == "ok"
