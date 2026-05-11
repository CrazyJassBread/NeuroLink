from __future__ import annotations

from collections.abc import Callable

from ..core.runtime import RuntimeState
from ..core.types import TaskValidationResult
from .task_spec import TaskSpec


TaskValidator = Callable[[TaskSpec | None, RuntimeState, list[str], list[dict]], TaskValidationResult]

_REGISTRY: dict[str, TaskValidator] = {}


def register_validator(task_type: str, validator: TaskValidator) -> None:
    _REGISTRY[task_type] = validator


def get_validator(task_type: str | None) -> TaskValidator | None:
    if task_type is None:
        return None
    return _REGISTRY.get(task_type)
