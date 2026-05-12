from __future__ import annotations

import warnings

warnings.warn(
    "env_diy.envs.dungeon_env is deprecated and kept only for compatibility. "
    "Import wrapper classes from env_diy.env or env_diy.wrappers instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ..wrappers.gym_env import DungeonEnv, GymDungeonEnv

__all__ = ["DungeonEnv", "GymDungeonEnv"]
