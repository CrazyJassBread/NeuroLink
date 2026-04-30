from __future__ import annotations

from typing import Any

import gymnasium as gym


def observation_is_valid(env: gym.Env, obs: Any) -> bool:
    return bool(env.observation_space.contains(obs))
