from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_diy.env import make_env
from env_diy.maps.rooms import RoomManager


DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"
EXPORT_SCRIPT = PROJECT_ROOT / "env_diy" / "tools" / "export_map.py"
MIGRATE_SCRIPT = PROJECT_ROOT / "env_diy" / "tools" / "migrate_map_schema.py"
SOURCE_ROOT = PROJECT_ROOT / "env_diy" / "diy_map_sources" / "examples"
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
    "reward",
    "task_type",
    "progress",
    "success",
    "failure",
    "reward_profile",
    "success_condition",
    "failure_condition",
}
FORBIDDEN_EVENT_KEYS = {
    "task_success",
    "task_failure",
    "task_progress",
    "objective",
    "reward",
    "reward_terms",
    "success_condition",
    "failure_condition",
}


def _all_map_files() -> list[Path]:
    return sorted(DUNGEON_ROOT.rglob("*.json"))


def test_all_map_json_are_pure_map_schema() -> None:
    for path in _all_map_files():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if "room_files" in payload:
            assert FORBIDDEN_MAP_KEYS.isdisjoint(payload.keys()), path
            continue
        assert FORBIDDEN_MAP_KEYS.isdisjoint(payload.keys()), path
        assert "id" in payload, path
        assert "coord" in payload, path
        assert "layout" in payload, path
        assert "spawns" in payload, path
        assert "default_spawn" in payload, path


def test_reset_info_contains_only_base_fields() -> None:
    env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym")
    try:
        _obs, info = env.reset(seed=0)
    finally:
        env.close()

    assert FORBIDDEN_INFO_KEYS.isdisjoint(info.keys())
    assert {"episode", "env", "agent", "inventory", "entities", "events", "terminal_reason", "control", "debug"} <= set(
        info
    )


def test_step_events_are_task_agnostic() -> None:
    env = make_env(DUNGEON_ROOT / "key_door" / "room_001.json", api="gym")
    try:
        _obs, info = env.reset(seed=0)
        names: set[str] = set()
        detail_keys: set[str] = set()
        for _ in range(20):
            _obs, _reward, terminated, truncated, info = env.step(int(env.action_space.sample()))
            names.update(record["name"] for record in info["events"]["records"])
            for detail in info["events"]["details"]:
                detail_keys.update(detail.keys())
            if terminated or truncated:
                break
    finally:
        env.close()

    assert "task_finished" not in names
    assert "victory" not in names
    assert FORBIDDEN_EVENT_KEYS.isdisjoint(names)
    assert FORBIDDEN_EVENT_KEYS.isdisjoint(detail_keys)


def test_base_env_exit_reached_does_not_force_termination() -> None:
    env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym")
    try:
        _obs, _info = env.reset(seed=0)
        runtime = env.engine.runtime
        runtime.player.position_px = (64.0, 0.0)
        _obs, _reward, terminated, truncated, info = env.step(1)
    finally:
        env.close()

    assert "exit_reached" in info["events"]["counts"]
    assert not terminated
    assert not truncated


def test_reward_wrapper_computes_rewards_from_events_only() -> None:
    from env_diy.tasks.registry import get_task_spec
    from env_diy.tasks.reward_fns import AvoidTrapReward, KeyDoorReward
    from env_diy.tasks.wrappers import RewardWrapper

    base_env = make_env(DUNGEON_ROOT / "key_door" / "room_001.json", api="gym")
    wrapped = RewardWrapper(base_env, reward_fn=KeyDoorReward(get_task_spec("key_door_room_001")))
    try:
        wrapped.reset(seed=0)
        reward, terminated = wrapped.reward_fn(
            {},
            {"events": {"counts": {}}},
            {},
            {"events": {"counts": {"key_collected": 1, "door_opened": 1, "exit_reached": 1}}},
            5,
        )
        assert reward > 0.0
        assert terminated is True
        assert wrapped.reward_fn.last_outcome.success is True
    finally:
        wrapped.close()

    trap_env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym")
    trap_wrapped = RewardWrapper(trap_env, reward_fn=AvoidTrapReward(get_task_spec("avoid_traps_room_001")))
    try:
        trap_wrapped.reset(seed=0)
        reward, terminated = trap_wrapped.reward_fn(
            {},
            {"events": {"counts": {}}},
            {},
            {"events": {"counts": {"trap_triggered": 1}}},
            0,
        )
        assert reward < 0.0
        assert terminated is False
    finally:
        trap_wrapped.close()


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
