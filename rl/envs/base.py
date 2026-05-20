from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym

from rl.config.schema import EnvironmentConfig


EnvironmentBuilder = Callable[..., gym.Env]


def build_render_mode(enabled: bool) -> str | None:
    return "rgb_array" if enabled else None
