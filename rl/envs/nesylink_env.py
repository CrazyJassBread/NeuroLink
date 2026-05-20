from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from rl.config.schema import EnvironmentConfig
from rl.utils.env_factory import DEFAULT_DUNGEON_CONFIG, make_env

from .base import build_render_mode


def build_nesylink_env(
    config: EnvironmentConfig,
    *,
    seed: int,
    render: bool | None = None,
) -> gym.Env:
    render_enabled = config.render if render is None else render
    config_path = config.map_path or DEFAULT_DUNGEON_CONFIG
    return make_env(
        config_path=Path(config_path),
        reward_id=config.reward_id,
        reward_module=config.reward_module,
        render_mode=build_render_mode(render_enabled),
        seed=seed,
        action_repeat=config.action_repeat,
        max_steps=config.max_episode_steps,
    )
