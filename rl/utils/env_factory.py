from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from env_diy.env import make_env as make_public_env
from env_diy.rewards import RewardWrapper
from env_diy.rewards.reward_fn import AvoidTrapReward, KeyDoorReward, KillMonsterReward
from env_diy.tasks import get_task_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUNGEON_CONFIG = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"


def make_env(
    config_path: str | Path | None = None,
    *,
    render_mode: str | None = None,
    seed: int | None = None,
    action_repeat: int = 1,
) -> gym.Env:
    """Create the DIY dungeon environment for lightweight RL smoke runs."""
    if action_repeat < 1:
        raise ValueError("action_repeat must be >= 1")

    dungeon_config = Path(config_path) if config_path is not None else DEFAULT_DUNGEON_CONFIG
    if not dungeon_config.is_absolute():
        dungeon_config = PROJECT_ROOT / dungeon_config

    try:
        env = make_public_env(
            dungeon_config,
            api="gym",
            render_mode=render_mode,
            action_repeat=action_repeat,
        )
    except Exception as exc:  # noqa: BLE001 - preserve source exception in a clearer message.
        raise RuntimeError(f"Failed to create DungeonEnv from config '{dungeon_config}': {exc}") from exc

    if seed is not None:
        env.action_space.seed(seed)
    return env


def make_task_env(
    task_id: str,
    *,
    config_path: str | Path | None = None,
    render_mode: str | None = None,
    seed: int | None = None,
    action_repeat: int = 1,
) -> gym.Env:
    task_spec = get_task_spec(task_id)
    base_env = make_env(
        config_path=config_path or task_spec.map_path,
        render_mode=render_mode,
        seed=seed,
        action_repeat=action_repeat,
    )
    reward_fn = _make_reward_fn(task_id)
    return RewardWrapper(base_env, reward_fn=reward_fn)


def _make_reward_fn(task_id: str):
    task_spec = get_task_spec(task_id)
    reward_factories = {
        "avoid_traps_room_001": AvoidTrapReward,
        "kill_monsters_room_001": KillMonsterReward,
        "key_door_room_001": KeyDoorReward,
    }
    reward_factory = reward_factories.get(task_spec.task_id)
    if reward_factory is None:
        available = ", ".join(sorted(reward_factories))
        raise ValueError(f"unsupported task reward spec '{task_spec.task_id}', available: {available}")
    return reward_factory(task_spec)
