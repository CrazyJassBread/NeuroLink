from __future__ import annotations

from typing import Any

from .env import make_env

__all__ = ["DungeonEnv", "ZeldaLikeGame", "make_env"]


def __getattr__(name: str) -> Any:
    if name == "ZeldaLikeGame":
        from .game import ZeldaLikeGame

        return ZeldaLikeGame
    if name == "DungeonEnv":
        from .env import DungeonEnv

        return DungeonEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
