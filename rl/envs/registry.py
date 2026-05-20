from __future__ import annotations

from rl.config.schema import EnvironmentConfig

from .base import EnvironmentBuilder
from .minigrid_envpool import build_minigrid_env
from .nesylink_env import build_nesylink_env


_ENV_BUILDERS: dict[str, EnvironmentBuilder] = {
    "nesylink": build_nesylink_env,
    "minigrid": build_minigrid_env,
}


def get_env_builder(name: str) -> EnvironmentBuilder:
    normalized = name.lower()
    try:
        return _ENV_BUILDERS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ENV_BUILDERS))
        raise ValueError(f"unsupported environment '{name}', supported: {supported}") from exc


def register_env_builder(name: str, builder: EnvironmentBuilder) -> None:
    _ENV_BUILDERS[name.lower()] = builder
