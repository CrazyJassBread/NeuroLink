from __future__ import annotations

from pathlib import Path

from env_diy.envs import DungeonEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUNGEON_CONFIG = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"


def make_env(
    config_path: str | Path | None = None,
    *,
    render_mode: str | None = None,
    seed: int | None = None,
) -> DungeonEnv:
    """Create the DIY dungeon environment for lightweight RL smoke runs."""
    dungeon_config = Path(config_path) if config_path is not None else DEFAULT_DUNGEON_CONFIG
    if not dungeon_config.is_absolute():
        dungeon_config = PROJECT_ROOT / dungeon_config

    try:
        env = DungeonEnv(dungeon_config, render_mode=render_mode)
    except Exception as exc:  # noqa: BLE001 - preserve source exception in a clearer message.
        raise RuntimeError(f"Failed to create DungeonEnv from config '{dungeon_config}': {exc}") from exc

    if seed is not None:
        env.action_space.seed(seed)
    return env
