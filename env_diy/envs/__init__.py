from __future__ import annotations

import warnings
from typing import Any

__all__ = ["DungeonEnv", "GymDungeonEnv"]


def __getattr__(name: str) -> Any:
    if name in {"DungeonEnv", "GymDungeonEnv"}:
        warnings.warn(
            "env_diy.envs is deprecated and kept only as a compatibility namespace. "
            "Use env_diy.env.make_env(..., api='gym') for the canonical Gym API, "
            "or import wrapper classes from env_diy.env / env_diy.wrappers directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..wrappers import DungeonEnv, GymDungeonEnv

        return {"DungeonEnv": DungeonEnv, "GymDungeonEnv": GymDungeonEnv}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
