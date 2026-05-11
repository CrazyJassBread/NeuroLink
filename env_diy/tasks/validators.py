from __future__ import annotations

from ..core.runtime import RuntimeState
from ..core.types import TaskValidationResult
from .registry import get_validator, register_validator
from .task_spec import TaskSpec


def validate_task(
    task_spec: TaskSpec | None,
    runtime: RuntimeState,
    events: list[str],
    event_details: list[dict],
    *,
    legacy_done: bool,
) -> TaskValidationResult:
    validator = get_validator(task_spec.task_type if task_spec is not None else None)
    if validator is None:
        result = _generic_validator(task_spec, runtime, events)
    else:
        result = validator(task_spec, runtime, events, event_details)
    validator_done = bool(result.success or result.failure)
    return TaskValidationResult(
        success=result.success,
        failure=result.failure,
        task_progress=result.task_progress,
        terminated_reason=result.terminated_reason,
        subgoal_status=result.subgoal_status,
        legacy_done=bool(legacy_done),
        validator_done=validator_done,
        validator_matches_legacy=bool(legacy_done) == validator_done,
    )


def _generic_validator(
    task_spec: TaskSpec | None,
    runtime: RuntimeState,
    events: list[str],
) -> TaskValidationResult:
    success = runtime.task_finished or "task_finished" in events or "victory" in events
    failure = runtime.player.health <= 0 or "game_over" in events
    reason = None
    if failure:
        reason = "agent_dead"
    elif success:
        reason = "reached_goal"
    return TaskValidationResult(
        success=success,
        failure=failure,
        task_progress=1.0 if success else 0.0,
        terminated_reason=reason,
        subgoal_status={"task_type": task_spec.task_type if task_spec is not None else None},
    )


def _task_progress_from_monsters(task_spec: TaskSpec, runtime: RuntimeState) -> float:
    target_monsters = task_spec.target_monsters
    if target_monsters:
        total = len(target_monsters)
        remaining = sum(1 for monster_id in target_monsters if monster_id in runtime.room.monsters)
    else:
        total = max(1, len(runtime.room.monsters))
        remaining = len(runtime.room.monsters)
    return max(0.0, min(1.0, 1.0 - (remaining / max(1, total))))


def _avoid_traps_validator(
    task_spec: TaskSpec | None,
    runtime: RuntimeState,
    events: list[str],
    event_details: list[dict],
) -> TaskValidationResult:
    del event_details
    return _generic_validator(task_spec, runtime, events)


def _kill_monsters_validator(
    task_spec: TaskSpec | None,
    runtime: RuntimeState,
    events: list[str],
    event_details: list[dict],
) -> TaskValidationResult:
    del event_details
    generic = _generic_validator(task_spec, runtime, events)
    progress = generic.task_progress
    if task_spec is not None:
        progress = _task_progress_from_monsters(task_spec, runtime)
        if generic.success:
            progress = 1.0
    return TaskValidationResult(
        success=generic.success,
        failure=generic.failure,
        task_progress=progress,
        terminated_reason=generic.terminated_reason,
        subgoal_status={
            "monsters_remaining": len(runtime.room.monsters),
        },
    )


def _key_door_validator(
    task_spec: TaskSpec | None,
    runtime: RuntimeState,
    events: list[str],
    event_details: list[dict],
) -> TaskValidationResult:
    del event_details
    generic = _generic_validator(task_spec, runtime, events)
    key_acquired = runtime.player.keys > 0 or "got_key" in events or "door_unlocked" in events
    door_unlocked = "door_unlocked" in events
    progress = 1.0 if generic.success else 0.5 if key_acquired or door_unlocked else 0.0
    return TaskValidationResult(
        success=generic.success,
        failure=generic.failure,
        task_progress=progress,
        terminated_reason=generic.terminated_reason,
        subgoal_status={
            "has_key": runtime.player.keys > 0,
            "door_unlocked": door_unlocked,
        },
    )


def _collect_coin_validator(
    task_spec: TaskSpec | None,
    runtime: RuntimeState,
    events: list[str],
    event_details: list[dict],
) -> TaskValidationResult:
    del event_details
    generic = _generic_validator(task_spec, runtime, events)
    success = generic.success or "got_gold" in events
    reason = generic.terminated_reason
    if success and reason is None:
        reason = "collected_coin"
    return TaskValidationResult(
        success=success,
        failure=generic.failure,
        task_progress=1.0 if success else 0.0,
        terminated_reason=reason,
        subgoal_status={"gold": runtime.player.gold},
    )


register_validator("avoid_traps", _avoid_traps_validator)
register_validator("kill_monsters", _kill_monsters_validator)
register_validator("key_door", _key_door_validator)
register_validator("collect_coin", _collect_coin_validator)
