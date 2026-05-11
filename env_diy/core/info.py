from __future__ import annotations

from typing import Any

from .events import build_event_counts, build_event_flags, build_event_records
from .runtime import RuntimeState
from .types import TaskValidationResult


def build_info(
    runtime: RuntimeState,
    *,
    events: list[str],
    event_details: list[dict[str, Any]],
    reward_terms: dict[str, float] | None = None,
    map_id: str | None = None,
    movement_pixels: int | float | None = None,
    action_repeat: int = 1,
    inner_steps: int = 1,
    legacy_terminated: bool = False,
    validator_result: TaskValidationResult | None = None,
    auto_reset: bool = False,
    task_id: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    finish = "task_finished" in events
    victory = "victory" in events or finish
    event_counts = build_event_counts(events)
    event_flags = build_event_flags(events)
    event_records = build_event_records(events, event_details)
    validation = validator_result or TaskValidationResult(
        success=victory,
        failure="game_over" in events,
        task_progress=1.0 if victory else 0.0,
        terminated_reason="agent_dead" if "game_over" in events else ("reached_goal" if victory else None),
        legacy_done=legacy_terminated,
        validator_done=victory or "game_over" in events,
        validator_matches_legacy=legacy_terminated == (victory or "game_over" in events),
    )
    inventory = {
        "gold": runtime.player.gold,
        "keys": runtime.player.keys,
        "items": list(runtime.player.items),
        "tools": list(runtime.player.tools),
        "equipped": dict(runtime.player.equipped),
    }
    info = {
        "episode_id": runtime.episode,
        "step_count": runtime.step_count,
        "room_id": runtime.room.room_id,
        "room_coord": runtime.room.coord,
        "map_id": map_id,
        "seed": runtime.seed,
        "health": runtime.player.health,
        "agent_hp": runtime.player.health,
        "gold": runtime.player.gold,
        "keys": runtime.player.keys,
        "items": list(runtime.player.items),
        "tools": list(runtime.player.tools),
        "equipped": dict(runtime.player.equipped),
        "inventory": inventory,
        "message": runtime.last_message,
        "events": events,
        "event_flags": event_flags,
        "event_counts": event_counts,
        "event_records": event_records,
        "event_details": event_details,
        "episode": runtime.episode,
        "step": runtime.step_count,
        "player_position_px": runtime.player.position_px,
        "player_tile": runtime.snapshot().player_tile,
        "agent_pos": runtime.player.position_px,
        "has_key": runtime.player.keys > 0,
        "key_count": runtime.player.keys,
        "picked_key": "got_key" in events,
        "unlocked_door": "door_unlocked" in events,
        "entered_new_room": "room_transition" in events,
        "task_success": victory,
        "finish": finish,
        "no_progress_steps": runtime.no_progress_steps,
        "task_progress": float(validation.task_progress),
        "success": bool(validation.success),
        "failure": bool(validation.failure),
        "terminated_reason": validation.terminated_reason,
        "subgoal_status": dict(validation.subgoal_status),
        "legacy_done": bool(validation.legacy_done),
        "validator_done": bool(validation.validator_done),
        "validator_matches_legacy": bool(validation.validator_matches_legacy),
        "reward_terms": dict(reward_terms or {}),
        "reward_breakdown": dict(reward_terms or {}),
        "action_repeat": int(action_repeat),
        "inner_steps": int(inner_steps),
        "movement_pixels": movement_pixels,
    }
    if task_id is not None:
        info["task_id"] = task_id
    if task_type is not None:
        info["task_type"] = task_type
    if auto_reset:
        info["auto_reset"] = True
    if victory:
        info["victory"] = True
    if "game_over" in events:
        info["game_over"] = True
    return info
