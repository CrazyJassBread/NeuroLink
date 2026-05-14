from __future__ import annotations

from typing import Any

from .events import build_event_records, event_counts_to_flags, event_records_to_counts
from .runtime import RuntimeState


def build_info(
    runtime: RuntimeState,
    *,
    events: list[str],
    event_details: list[dict[str, Any]],
    map_id: str | None = None,
    movement_pixels: int | float | None = None,
    action_repeat: int = 1,
    inner_steps: int = 1,
    engine_terminated: bool = False,
    terminal_reason: str | None = None,
    debug_message: str | None | object = ...,
) -> dict[str, Any]:
    event_records = build_event_records(events, event_details)
    event_counts = event_records_to_counts(event_records)
    event_flags = event_counts_to_flags(event_counts)
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

    entities = {
        "monsters_remaining": len(runtime.room.monsters),
        "monster_ids": sorted(runtime.room.monsters),
        "chests_remaining": sum(1 for chest in runtime.room.chests.values() if not chest.is_open),
        "traps_active": sum(1 for trap in runtime.room.traps.values() if trap.is_active),
        "buttons_pressed": sum(1 for button in runtime.room.buttons.values() if button.is_pressed),
        "exits_open": sum(1 for exit_cfg in runtime.room.exits if runtime.room.exit_state(exit_cfg).opened),
        "exits_total": len(runtime.room.exits),
    }
    debug_info = {
        "message": resolved_debug_message,
        "engine_done": bool(engine_terminated),
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
        "entities": entities,
        "events": {
            "records": event_records,
            "flags": event_flags,
            "counts": event_counts,
            "details": list(event_details),
        },
        "terminal_reason": terminal_reason,
        "control": {
            "action_repeat": int(action_repeat),
            "inner_steps": int(inner_steps),
            "movement_pixels": movement_pixels,
        },
        "debug": debug_info,
    }
    return info
