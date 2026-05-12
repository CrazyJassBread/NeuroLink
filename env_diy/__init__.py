from __future__ import annotations

import warnings
from typing import Any

from .app import ZeldaLikeGame
from .env import make_env

__all__ = ["DungeonEnv", "ZeldaLikeGame", "make_env"]


def __getattr__(name: str) -> Any:
    if name == "DungeonEnv":
        warnings.warn(
            "env_diy.DungeonEnv is deprecated and kept only for compatibility. "
            "Use env_diy.env.make_env(..., api='gym') for the canonical Gym API, "
            "or env_diy.env.DungeonEnv if you explicitly need legacy auto-reset behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .env import DungeonEnv

        return DungeonEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
