from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from nesylink.wrappers.gym_env import DEFAULT_DUNGEON_CONFIG, make_gym_env, with_default_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_env(
    config_path: str | Path | None = None,
    *,
    reward_id: str | None = None,
    reward_module: str | None = None,
    render_mode: str | None = None,
    seed: int | None = None,
    action_repeat: int = 1,
    max_steps: int | None = None,
) -> gym.Env:
    """Create the NesyLink environment for lightweight RL smoke runs."""
    if action_repeat < 1:
        raise ValueError("action_repeat must be >= 1")

    dungeon_config = Path(config_path) if config_path is not None else DEFAULT_DUNGEON_CONFIG

    try:
        env = make_gym_env(
            config_path=dungeon_config,
            reward_id=reward_id,
            reward_module=reward_module,
            render_mode=render_mode,
            action_repeat=action_repeat,
            max_steps=max_steps,
        )
    except Exception as exc:  # noqa: BLE001 - preserve source exception in a clearer message.
        raise RuntimeError(f"Failed to create DungeonEnv from config '{dungeon_config}': {exc}") from exc

    if seed is not None:
        env = with_default_seed(env, seed)
    return env
