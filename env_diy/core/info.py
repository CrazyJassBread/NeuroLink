from __future__ import annotations

from typing import Any

from .events import build_event_records, event_counts_to_flags, event_records_to_counts
from .runtime import RuntimeState
from .types import TaskValidationResult


def build_info(
    runtime: RuntimeState,
    *,
    events: list[str],
    event_details: list[dict[str, Any]],
    reward_terms: dict[str, float] | None = None,
    reward_total: float = 0.0,
    reward_mode: str = "default",
    map_id: str | None = None,
    movement_pixels: int | float | None = None,
    action_repeat: int = 1,
    inner_steps: int = 1,
    engine_terminated: bool = False,
    validator_result: TaskValidationResult | None = None,
    debug_message: str | None | object = ...,
) -> dict[str, Any]:
    finish = "task_finished" in events
    victory = "victory" in events or finish
    event_records = build_event_records(events, event_details)
    event_counts = event_records_to_counts(event_records)
    event_flags = event_counts_to_flags(event_counts)
    validation = validator_result or TaskValidationResult(
        success=victory,
        failure="game_over" in events,
        task_progress=1.0 if victory else 0.0,
        terminated_reason="agent_dead" if "game_over" in events else ("reached_goal" if victory else None),
        engine_done=engine_terminated,
        validator_done=victory or "game_over" in events,
        validator_matches_engine=engine_terminated == (victory or "game_over" in events),
    )
    player_tile = runtime.snapshot().player_tile
    inventory = {
        "gold": runtime.player.gold,
        "keys": runtime.player.keys,
        "items": list(runtime.player.items),
        "tools": list(runtime.player.tools),
        "equipped": dict(runtime.player.equipped),
    }
    resolved_debug_message = runtime.last_message or None
    if debug_message is not ...:
        resolved_debug_message = debug_message

    task_info = {
        "success": bool(validation.success),
        "failure": bool(validation.failure),
        "progress": float(validation.task_progress),
        "terminated_reason": validation.terminated_reason,
        "subgoals": dict(validation.subgoal_status),
        "completed_subgoals": [],
        "failure_stage": None,
    }
    reward_info = {
        "mode": reward_mode,
        "total": float(reward_total),
        "terms": dict(reward_terms or {}),
    }
    debug_info = {
        "message": resolved_debug_message,
        "engine_done": bool(validation.engine_done),
        "validator_done": bool(validation.validator_done),
        "validator_matches_engine": bool(validation.validator_matches_engine),
    }

    info: dict[str, Any] = {
        "episode": {
            "id": runtime.episode,
            "step_count": runtime.step_count,
            "seed": runtime.seed,
        },
        "env": {
            "map_id": map_id,
            "room_id": runtime.room.room_id,
            "room_coord": runtime.room.coord,
        },
        "agent": {
            "hp": runtime.player.health,
            "position_px": runtime.player.position_px,
            "tile": player_tile,
        },
        "inventory": inventory,
        "events": {
            "records": event_records,
            "flags": event_flags,
            "counts": event_counts,
            "details": list(event_details),
        },
        "task": task_info,
        "reward": reward_info,
        "control": {
            "action_repeat": int(action_repeat),
            "inner_steps": int(inner_steps),
            "movement_pixels": movement_pixels,
        },
        "debug": debug_info,
    }
    return info
