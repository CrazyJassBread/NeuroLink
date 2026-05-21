from __future__ import annotations

from typing import Any

from ..constants import MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH, TILE_SIZE
from ..state import move_with_tile_collisions, tile_to_top_left_px
from ..world.rooms import ExitConfig, RoomState, direction_from_entry_name, first_valid_entry_spawn_tile


MOVE_TO_EXIT_DIRECTION = {
    "up": "north",
    "down": "south",
    "left": "west",
    "right": "east",
}


def handle_move(engine: Any, direction: str, result: Any) -> None:
    runtime = engine.runtime
    runtime.player.facing = direction
    step_dx, step_dy = {
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
    }[direction]
    previous_position = runtime.player.position_px
    current_position = previous_position
    blocked_position: tuple[float, float] | None = None
    for _ in range(engine.move_speed_px):
        proposed_position = (
            current_position[0] + step_dx,
            current_position[1] + step_dy,
        )
        next_position = move_with_tile_collisions(
            current_position,
            runtime.player.size_px,
            (step_dx, step_dy),
            runtime.room.blocking_tiles(),
        )
        if next_position == current_position:
            blocked_position = proposed_position
            break
        current_position = next_position

    runtime.player.position_px = current_position
    if runtime.player.position_px == previous_position:
        blocked_reason = "bounds"
        if blocked_position is not None and not within_map_bounds(blocked_position):
            runtime.last_message = "EDGE BLOCKED"
        else:
            runtime.last_message = "BLOCKED"
            blocked_reason = "wall"
        result.events.append("action_blocked")
        result.event_details.append(
            {
                "type": "action_blocked",
                "reason": blocked_reason,
                "direction": direction,
            }
        )
        return

    runtime.last_message = f"MOVE {direction.upper()}"
    result.events.append(f"move_{direction}")


def resolve_transition(engine: Any, direction: str, result: Any) -> None:
    runtime = engine.runtime
    player_tile = runtime.snapshot().player_tile
    exit_direction = MOVE_TO_EXIT_DIRECTION[direction]
    exit_config = runtime.room.exit_at(player_tile, exit_direction)
    if exit_config is None or not player_is_flush_with_edge(runtime, direction):
        return
    apply_exit(engine, exit_config, result)


def apply_exit(engine: Any, exit_config: ExitConfig, result: Any) -> None:
    runtime = engine.runtime
    allowed, blocked_reason = can_use_exit(runtime, exit_config)
    if not allowed:
        runtime.last_message = exit_config.blocked_message
        result.events.append("action_blocked")
        result.event_details.append(
            {
                "type": "action_blocked",
                "reason": blocked_reason,
                "exit_id": exit_config.exit_id,
                "direction": exit_config.direction,
            }
        )
        return

    from_room_id = runtime.room.room_id
    exit_state = runtime.room.exit_state(exit_config)
    if exit_config.exit_type == "locked_key" and not exit_state.unlocked:
        key_consumed = False
        if bool(exit_config.requires.get("consume_key", False)):
            runtime.player.keys -= int(exit_config.requires.get("key_count", 1))
            key_consumed = True
        exit_state.unlocked = True
        exit_state.opened = True
        result.events.append("door_opened")
        result.event_details.append(
            {
                "type": "door_opened",
                "room_id": from_room_id,
                "direction": exit_config.direction,
                "exit_id": exit_config.exit_id,
                "key_consumed": key_consumed,
            }
        )

    target_coord = engine.room_manager.coord_for_room_id(exit_config.target_room_id)
    target_room = engine.room_manager.get_room(target_coord)
    spawn_tile = entry_spawn_tile(target_room, exit_config.target_entry)
    runtime.room_coord = target_coord
    runtime.room = target_room
    runtime.player.position_px = tile_to_top_left_px(spawn_tile)
    runtime.last_message = exit_config.success_message
    result.events.append("exit_reached")
    result.events.append("room_changed")
    result.event_details.append(
        {
            "type": "exit_reached",
            "from_room": from_room_id,
            "to_room": runtime.room.room_id,
            "exit_id": exit_config.exit_id,
            "direction": exit_config.direction,
            "target_entry": exit_config.target_entry,
            "spawn_px": [runtime.player.position_px[0], runtime.player.position_px[1]],
        }
    )
    if exit_config.complete_task:
        result.events.append("environment_completed")


def can_use_exit(runtime: Any, exit_config: ExitConfig) -> tuple[bool, str]:
    if exit_config.exit_type == "normal":
        return True, ""
    if exit_config.exit_type == "locked_key":
        if runtime.room.exit_state(exit_config).unlocked:
            return True, ""
        required_keys = int(exit_config.requires.get("key_count", 1))
        if runtime.player.keys < required_keys:
            return False, "locked"
        return True, ""

    button_id = exit_config.requires.get("button_pressed")
    if button_id is not None:
        button = runtime.room.buttons.get(button_id)
        if button is None or not button.is_pressed:
            return False, "missing_requirement"
    item_name = exit_config.requires.get("item")
    if item_name is not None and item_name not in runtime.player.items:
        return False, "missing_requirement"
    if exit_config.requires.get("all_monsters_defeated") and len(runtime.room.monsters) > 0:
        return False, "missing_requirement"
    if "key_count" in exit_config.requires:
        required_keys = int(exit_config.requires.get("key_count", 1))
        if runtime.player.keys < required_keys:
            return False, "locked"
    return True, ""


def entry_spawn_tile(room: RoomState, target_entry: str) -> tuple[int, int]:
    entry_direction = direction_from_entry_name(target_entry)
    if entry_direction is not None:
        spawn_tile = first_valid_entry_spawn_tile(entry_direction, room.walls)
        if spawn_tile is None:
            raise RuntimeError(
                f"room '{room.room_id}' has no valid spawn tile for entry '{target_entry}'"
            )
        return spawn_tile
    return room.spawns[target_entry]


def player_is_flush_with_edge(runtime: Any, direction: str) -> bool:
    epsilon = 1e-6
    left, top = runtime.player.position_px
    if direction == "left":
        return left <= epsilon
    if direction == "right":
        return left >= MAP_PIXEL_WIDTH - runtime.player.size_px - epsilon
    if direction == "up":
        return top <= epsilon
    return top >= MAP_PIXEL_HEIGHT - runtime.player.size_px - epsilon


def within_map_bounds(position_px: tuple[float, float]) -> bool:
    return (
        0.0 <= position_px[0] <= MAP_PIXEL_WIDTH - TILE_SIZE
        and 0.0 <= position_px[1] <= MAP_PIXEL_HEIGHT - TILE_SIZE
    )
