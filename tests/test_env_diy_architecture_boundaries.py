from __future__ import annotations
# ruff: noqa: E402

import importlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.env import make_env
from nesylink.core.world.loader import load_map
from nesylink.core.world.rooms import RoomManager


DUNGEON_ROOT = PROJECT_ROOT / "nesylink" / "map_data" / "dungeons"
EXPORT_SCRIPT = PROJECT_ROOT / "nesylink" / "tools" / "export_map.py"
MIGRATE_SCRIPT = PROJECT_ROOT / "nesylink" / "tools" / "migrate_map_schema.py"
SOURCE_ROOT = PROJECT_ROOT / "nesylink" / "diy_map_sources" / "examples"
FORBIDDEN_MAP_KEYS = {
    "task",
    "task_id",
    "task_type",
    "objective",
    "progress",
    "reward",
    "rewards",
    "reward_profile",
    "success_condition",
    "failure_condition",
}
FORBIDDEN_INFO_KEYS = {
    "task",
    "task_type",
    "progress",
    "success",
    "failure",
    "reward_profile",
    "success_condition",
    "failure_condition",
}
OBSOLETE_IMPORT_PATHS = (
    "nesylink.maps",
    "nesylink.entities",
    "nesylink.input",
    "nesylink.app",
    "nesylink.rendering",
    "nesylink.integrations",
    "nesylink.envs",
)
IMPORT_SCAN_ROOTS = (
    PROJECT_ROOT / "nesylink",
    PROJECT_ROOT / "rl" / "utils" / "env_factory.py",
    PROJECT_ROOT / "world_model" / "dreamerv3" / "embodied" / "envs" / "nesylink.py",
)


def _all_map_files() -> list[Path]:
    return sorted(DUNGEON_ROOT.rglob("*.json"))


def test_all_map_json_are_pure_map_schema() -> None:
    for path in _all_map_files():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        assert FORBIDDEN_MAP_KEYS.isdisjoint(payload.keys()), path


def test_load_map_supports_map_id_and_map_path() -> None:
    by_id = load_map(map_id="dungeon")
    by_path = load_map(map_path="nesylink/map_data/dungeons/prototype/dungeon.json")

    expected = PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "prototype" / "dungeon.json"
    assert by_id == expected
    assert by_path == expected


def test_reset_info_contains_only_base_fields_plus_reward() -> None:
    env = make_env(map_path=DUNGEON_ROOT / "avoid_traps" / "room_001.json", reward_id="sparse_exit", max_steps=10)
    try:
        _obs, info = env.reset(seed=0)
    finally:
        env.close()

    assert FORBIDDEN_INFO_KEYS.isdisjoint(info.keys())
    assert {"episode", "env", "agent", "inventory", "entities", "events", "game", "terminal_reason", "control", "debug", "reward"} <= set(
        info
    )
    assert info["reward"]["reward_name"] == "sparse_exit"
    assert "reward_signals" in info["reward"]
    assert "reward_weights" in info["reward"]


def test_step_populates_reward_info_and_signal_step() -> None:
    env = make_env(map_id="dungeon", reward_id="sparse_exit", max_steps=10)
    try:
        env.reset(seed=0)
        _obs, _reward, _terminated, _truncated, info = env.step(0)
    finally:
        env.close()

    assert info["reward"]["reward_name"] == "sparse_exit"
    assert info["reward"]["reward_signals"]["step"] == 1


def test_reward_module_creation_path_works() -> None:
    env = make_env(
        map_path=DUNGEON_ROOT / "key_door" / "room_001.json",
        reward_module="nesylink.rewards.collect_key",
        max_steps=20,
    )
    try:
        _obs, info = env.reset(seed=0)
    finally:
        env.close()

    assert info["reward"]["reward_name"] == "collect_key"


def test_max_steps_triggers_truncation() -> None:
    env = make_env(map_id="dungeon", reward_id="sparse_exit", max_steps=1)
    try:
        env.reset(seed=0)
        _obs, _reward, terminated, truncated, _info = env.step(0)
    finally:
        env.close()

    assert terminated is False
    assert truncated is True


def test_kill_monster_reward_terminates_when_room_is_cleared() -> None:
    env = make_env(
        map_path=DUNGEON_ROOT / "kill_monsters" / "room_001.json",
        reward_id="kill_monster",
        max_steps=20,
    )
    try:
        _obs, _info = env.reset(seed=0)
        env.engine.runtime.room.monsters.clear()
        _obs, _reward, terminated, truncated, info = env.step(0)
    finally:
        env.close()

    assert terminated is True
    assert truncated is False
    assert info["reward"]["terminated_reason"] == "all_monsters_defeated"


def test_collect_key_reward_terminates_when_door_opens() -> None:
    env = make_env(
        map_path=DUNGEON_ROOT / "key_door" / "room_001.json",
        reward_id="collect_key",
        max_steps=20,
    )
    try:
        _obs, _info = env.reset(seed=0)
        runtime = env.engine.runtime
        runtime.player.keys = 1
        runtime.player.position_px = (64.0, 0.0)
        _obs, _reward, terminated, truncated, info = env.step(1)
    finally:
        env.close()

    assert terminated is True
    assert truncated is False
    assert info["reward"]["reward_signals"]["door_opened"] == 1
    assert info["reward"]["terminated_reason"] == "door_opened"


def test_death_triggers_termination() -> None:
    env = make_env(map_path=DUNGEON_ROOT / "avoid_traps" / "room_001.json", reward_id="sparse_exit", max_steps=20)
    try:
        _obs, _info = env.reset(seed=0)
        env.engine.runtime.player.health = 0
        _obs, _reward, terminated, truncated, info = env.step(0)
    finally:
        env.close()

    assert terminated is True
    assert truncated is False
    assert info["game"]["dead"] is True


def test_builtin_rewards_can_run_short_episodes() -> None:
    reward_targets = [
        ("dungeon", "sparse_exit"),
        (DUNGEON_ROOT / "key_door" / "room_001.json", "collect_key"),
        (DUNGEON_ROOT / "key_door" / "room_001.json", "collect_gold"),
        (DUNGEON_ROOT / "kill_monsters" / "room_001.json", "kill_monster"),
        ("dungeon", "exploration"),
    ]

    for map_target, reward_id in reward_targets:
        kwargs = {"map_id": map_target} if isinstance(map_target, str) else {"map_path": map_target}
        env = make_env(**kwargs, reward_id=reward_id, max_steps=5)
        try:
            env.reset(seed=0)
            _obs, reward, _terminated, _truncated, info = env.step(0)
        finally:
            env.close()

        assert isinstance(reward, float)
        assert info["reward"]["reward_name"] == reward_id


def test_migrate_map_schema_converts_legacy_task_room(tmp_path: Path) -> None:
    legacy_map = tmp_path / "legacy_room.json"
    legacy_map.write_text(
        json.dumps(
            {
                "task_id": "legacy_key_door",
                "task_type": "key_door",
                "room_id": "room_001",
                "objective": {"type": "key_door", "target_exit": "north_exit"},
                "reward": {"finish": 10.0, "key": 0.5, "door_unlock": 0.5},
                "layout": [".........." for _ in range(8)],
                "spawns": {"default": [4, 6]},
                "default_spawn": "default",
                "objects": [],
                "exits": [],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "migrated_room.json"
    report_path = tmp_path / "migrated_room.report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(MIGRATE_SCRIPT),
            "--input",
            str(legacy_map),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    migrated = json.loads(output_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert FORBIDDEN_MAP_KEYS.isdisjoint(migrated.keys())
    assert migrated["id"] == "room_001"
    assert migrated["coord"] == [0, 0]
    assert report["legacy_task"]["task_type"] == "key_door"


def test_export_map_outputs_pure_map_json(tmp_path: Path) -> None:
    output_path = tmp_path / "room_001.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--input",
            str(SOURCE_ROOT / "trap_room.yaml"),
            "--output",
            str(output_path),
            "--room-id",
            "room_001",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert FORBIDDEN_MAP_KEYS.isdisjoint(payload.keys())
    RoomManager(output_path)


def test_import_nesylink_does_not_eagerly_import_app_or_pygame() -> None:
    script = """
import json
import sys

import nesylink

print(json.dumps({
    "has_make_env": callable(getattr(nesylink, "make_env", None)),
    "game_loaded": "nesylink.game" in sys.modules,
    "pygame_loaded": "pygame" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    payload = json.loads(result.stdout)
    assert payload["has_make_env"] is True
    assert payload["game_loaded"] is False
    assert payload["pygame_loaded"] is False


def test_canonical_modules_import_from_new_architecture() -> None:
    rooms_module = importlib.import_module("nesylink.core.world.rooms")
    loader_module = importlib.import_module("nesylink.core.world.loader")
    state_module = importlib.import_module("nesylink.core.state")
    monsters_module = importlib.import_module("nesylink.core.monsters")
    engine_module = importlib.import_module("nesylink.core.mechanics.engine")
    rendering_module = importlib.import_module("nesylink.core.rendering")
    input_module = importlib.import_module("nesylink.core.input.human")
    rewards_module = importlib.import_module("nesylink.rewards")
    wrappers_module = importlib.import_module("nesylink.wrappers")

    assert hasattr(rooms_module, "RoomManager")
    assert callable(getattr(loader_module, "load_map", None))
    assert hasattr(state_module, "PlayerState")
    assert hasattr(monsters_module, "MonsterState")
    assert hasattr(engine_module, "DungeonEngine")
    assert callable(getattr(rendering_module, "render_frame", None))
    assert hasattr(input_module, "HumanInputState")
    assert callable(getattr(rewards_module, "load_reward", None))
    assert callable(getattr(wrappers_module, "get_wrapper", None))


def test_core_world_and_mechanics_responsibilities_are_split_into_modules() -> None:
    schema_module = importlib.import_module("nesylink.core.world.schema")
    parser_module = importlib.import_module("nesylink.core.world.parser")
    validator_module = importlib.import_module("nesylink.core.world.validator")
    rooms_module = importlib.import_module("nesylink.core.world.rooms")
    movement_module = importlib.import_module("nesylink.core.mechanics.movement")
    interactions_module = importlib.import_module("nesylink.core.mechanics.interactions")
    combat_module = importlib.import_module("nesylink.core.mechanics.combat")
    progress_module = importlib.import_module("nesylink.core.mechanics.progress")

    assert hasattr(schema_module, "RoomTemplate")
    assert callable(getattr(parser_module, "build_room_template", None))
    assert callable(getattr(validator_module, "validate_exit_targets", None))
    assert hasattr(rooms_module, "RoomManager")
    assert callable(getattr(movement_module, "handle_move", None))
    assert callable(getattr(movement_module, "resolve_transition", None))
    assert callable(getattr(interactions_module, "handle_equipped_action", None))
    assert callable(getattr(interactions_module, "resolve_tile_effects", None))
    assert callable(getattr(combat_module, "update_monsters", None))
    assert callable(getattr(combat_module, "resolve_monster_contact", None))
    assert callable(getattr(progress_module, "step_made_progress", None))


def test_internal_code_does_not_import_obsolete_nesylink_namespaces() -> None:
    offenders: list[str] = []

    for root in IMPORT_SCAN_ROOTS:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            if path.name == "__pycache__":
                continue
            content = path.read_text(encoding="utf-8")
            for obsolete_path in OBSOLETE_IMPORT_PATHS:
                if obsolete_path in content:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {obsolete_path}")

    assert offenders == []
