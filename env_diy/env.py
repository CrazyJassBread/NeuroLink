from __future__ import annotations

from pathlib import Path
from typing import Any

from .wrappers import DungeonEnv, GymDungeonEnv
from .wrappers import get_wrapper


def make_env(
    config_path: str | Path,
    *,
    api: str = "gym",
    **kwargs: Any,
):
    wrapper_cls = get_wrapper(api)
    return wrapper_cls(config_path, **kwargs)


__all__ = ["DungeonEnv", "GymDungeonEnv", "make_env"]
