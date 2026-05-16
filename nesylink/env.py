from __future__ import annotations

from pathlib import Path
from typing import Any

from .rewards.loader import load_reward
from .wrappers import DungeonEnv, GymDungeonEnv, get_wrapper
from .core.world.loader import load_map


def make_env(
    config_path: str | Path | None = None,
    *,
    map_id: str | None = None,
    map_path: str | Path | None = None,
    api: str = "gym",
    reward_id: str | None = None,
    reward_module: str | None = None,
    **kwargs: Any,
):
    resolved_map_path = load_map(
        map_id=map_id,
        map_path=map_path if map_path is not None else config_path,
    )
    reward_fn = load_reward(
        reward_id=reward_id,
        reward_module=reward_module,
        reward_kwargs=kwargs.pop("reward_kwargs", None),
    )
    wrapper_cls = get_wrapper(api)
    return wrapper_cls(resolved_map_path, reward_fn=reward_fn, **kwargs)


__all__ = ["DungeonEnv", "GymDungeonEnv", "make_env"]
