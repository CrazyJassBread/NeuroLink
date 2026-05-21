from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from nesylink.core.constants import SCREEN_HEIGHT, SCREEN_WIDTH
from rl.config.schema import EnvironmentConfig
from rl.utils.env_factory import DEFAULT_DUNGEON_CONFIG, make_env

from .base import build_render_mode


class PixelObservationWrapper(gym.ObservationWrapper):
    """Expose the native NesyLink RGB frame as the observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3),
            dtype=np.uint8,
        )

    def observation(self, observation):
        del observation
        return self.env.render().astype(np.uint8)


def build_nesylink_env(
    config: EnvironmentConfig,
    *,
    seed: int,
    render: bool | None = None,
) -> gym.Env:
    render_enabled = config.render if render is None else render
    config_path = config.map_path or DEFAULT_DUNGEON_CONFIG
    observation_mode = str(config.params.get("observation_mode", "dict")).lower()
    render_mode = "rgb_array" if observation_mode == "pixels" else build_render_mode(render_enabled)

    env = make_env(
        config_path=Path(config_path),
        reward_id=config.reward_id,
        reward_module=config.reward_module,
        render_mode=render_mode,
        seed=seed,
        action_repeat=config.action_repeat,
        max_steps=config.max_episode_steps,
    )

    if observation_mode == "dict":
        return env
    if observation_mode == "pixels":
        return PixelObservationWrapper(env)
    raise ValueError(f"unsupported nesylink observation_mode: {observation_mode}")
