from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from env_diy.env import make_env as make_public_env


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
