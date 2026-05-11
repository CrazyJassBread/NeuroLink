from __future__ import annotations

from typing import Callable


WrapperFactory = Callable[..., object]

_REGISTRY: dict[str, WrapperFactory] = {}


def register_wrapper(api: str, factory: WrapperFactory) -> None:
    _REGISTRY[api] = factory


def get_wrapper(api: str) -> WrapperFactory:
    try:
        return _REGISTRY[api]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unsupported env api '{api}', available: {available}") from exc


def registered_wrappers() -> dict[str, WrapperFactory]:
    return dict(_REGISTRY)
