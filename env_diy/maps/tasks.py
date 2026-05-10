from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_TASK_TYPES = {"avoid_traps", "kill_monsters", "key_door"}
SUPPORTED_OBJECTIVE_TYPES = {
    "reach_exit",
    "reach_exit_without_trap_damage",
    "kill_monsters",
    "key_door",
}


@dataclass(frozen=True)
class TaskRewardConfig:
    step: float = -0.01
    damage: float = -1.0
    key: float = 0.4
    door_unlock: float = 0.2
    monster_kill: float = 0.3
    finish: float = 10.0


@dataclass(frozen=True)
class ObjectiveConfig:
    objective_type: str
    target_exit: str | None = None
    target_monsters: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    task_type: str
    room_id: str
    objective: ObjectiveConfig
    reward: TaskRewardConfig = field(default_factory=TaskRewardConfig)


def parse_task_config(
    payload: dict[str, Any],
    source_path: str | Path,
    validation_error_type: type[ValueError],
) -> TaskConfig:
    """Parse single-task metadata from a self-contained room JSON object."""

    def error(field_path: str, message: str) -> ValueError:
        return validation_error_type(source_path, field_path, message)

    task_id = _require_string(payload.get("task_id"), "task_id", error)
    task_type = _require_string(payload.get("task_type"), "task_type", error)
    if task_type not in SUPPORTED_TASK_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise error("task_type", f"unsupported task type '{task_type}', allowed: {allowed}")

    room_id = _require_string(payload.get("room_id", payload.get("id")), "room_id", error)

    raw_objective = payload.get("objective")
    if not isinstance(raw_objective, dict):
        raise error("objective", "must be an object")
    objective_type = _require_string(raw_objective.get("type"), "objective.type", error)
    if objective_type not in SUPPORTED_OBJECTIVE_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_OBJECTIVE_TYPES))
        raise error("objective.type", f"unsupported objective type '{objective_type}', allowed: {allowed}")

    target_exit = raw_objective.get("target_exit")
    if target_exit is not None:
        target_exit = _require_string(target_exit, "objective.target_exit", error)

    raw_targets = raw_objective.get("target_monsters", [])
    if raw_targets is None:
        raw_targets = []
    if not isinstance(raw_targets, list):
        raise error("objective.target_monsters", "must be a list")
    target_monsters = tuple(
        _require_string(value, f"objective.target_monsters[{index}]", error)
        for index, value in enumerate(raw_targets)
    )

    return TaskConfig(
        task_id=task_id,
        task_type=task_type,
        room_id=room_id,
        objective=ObjectiveConfig(
            objective_type=objective_type,
            target_exit=target_exit,
            target_monsters=target_monsters,
        ),
        reward=_parse_reward(payload.get("reward", {}), error),
    )


def _parse_reward(raw_reward: Any, error) -> TaskRewardConfig:
    if raw_reward is None:
        raw_reward = {}
    if not isinstance(raw_reward, dict):
        raise error("reward", "must be an object")
    unknown = sorted(set(raw_reward) - set(TaskRewardConfig.__dataclass_fields__))
    if unknown:
        raise error("reward", f"unsupported reward keys: {', '.join(unknown)}")
    defaults = TaskRewardConfig()
    values = {
        field_name: float(raw_reward.get(field_name, getattr(defaults, field_name)))
        for field_name in TaskRewardConfig.__dataclass_fields__
    }
    return TaskRewardConfig(**values)


def _require_string(value: Any, field_path: str, error) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(field_path, "must be a non-empty string")
    return value
