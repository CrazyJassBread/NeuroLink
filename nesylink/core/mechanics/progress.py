from __future__ import annotations

from typing import Any


def step_made_progress(
    runtime: Any,
    start_pos: tuple[float, float] | None,
    start_room_id: str | None,
    events: list[str],
) -> bool:
    if start_pos is not None and runtime.player.position_px != start_pos:
        return True
    if start_room_id is not None and runtime.room.room_id != start_room_id:
        return True
    progress_events = {
        "door_opened",
        "key_collected",
        "gold_collected",
        "item_collected",
        "agent_healed",
        "monster_killed",
        "chest_opened",
        "button_pressed",
        "room_changed",
        "exit_reached",
        "talked_npc",
        "environment_completed",
    }
    return any(event in progress_events for event in events)


def all_chests_opened(room_manager: Any) -> bool:
    total_chests = 0
    for coord in room_manager.room_templates:
        room = room_manager.get_room(coord)
        for chest in room.chests.values():
            total_chests += 1
            if not chest.is_open:
                return False
    return total_chests > 0
