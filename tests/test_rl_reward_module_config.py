from __future__ import annotations

from pathlib import Path

from rl.config.training import PROJECT_ROOT, resolve_task_rooms
from rl.utils.env_factory import make_env


def test_resolve_task_rooms_returns_map_and_reward_targets() -> None:
    targets = resolve_task_rooms(("avoid_traps",))

    assert len(targets) == 1
    assert targets[0].name == "avoid_traps"
    assert targets[0].map_path == PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "avoid_traps" / "room_001.json"
    assert targets[0].reward_id == "sparse_exit"
    assert targets[0].reward_module is None


def test_resolve_task_rooms_explicit_config_path_preserves_reward_override_absence() -> None:
    custom_config = Path("env_diy/map_data/dungeons/prototype/dungeon.json")

    targets = resolve_task_rooms(("prototype",), config_path=custom_config)

    assert len(targets) == 1
    assert targets[0].name == "prototype"
    assert targets[0].map_path == PROJECT_ROOT / custom_config
    assert targets[0].reward_id is None
    assert targets[0].reward_module is None


def test_make_env_supports_reward_module_directly() -> None:
    env = make_env(
        config_path="env_diy/map_data/dungeons/avoid_traps/room_001.json",
        reward_module="env_diy.rewards.sparse_exit",
        action_repeat=1,
        max_steps=5,
    )
    try:
        _obs, info = env.reset(seed=0)
        assert info["reward"]["reward_name"] == "sparse_exit"
    finally:
        env.close()
