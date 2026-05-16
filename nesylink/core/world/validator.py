from __future__ import annotations

from pathlib import Path

from .schema import MapValidationError, RoomTemplate, direction_from_entry_name, first_valid_entry_spawn_tile


def validate_exit_targets(
    room_file: Path,
    room_templates: dict[tuple[int, int], RoomTemplate],
    room_ids: dict[str, tuple[int, int]],
) -> None:
    for template in room_templates.values():
        for index, exit_config in enumerate(template.exits):
            if exit_config.target_room_id not in room_ids:
                raise MapValidationError(
                    room_file,
                    f"rooms[{template.room_id}].exits[{index}].target_room",
                    f"unknown target room '{exit_config.target_room_id}'",
                )
            target_template = room_templates[room_ids[exit_config.target_room_id]]
            entry_direction = direction_from_entry_name(exit_config.target_entry)
            if entry_direction is not None:
                entry_tile = first_valid_entry_spawn_tile(entry_direction, target_template.walls)
                if entry_tile is None:
                    raise MapValidationError(
                        room_file,
                        f"rooms[{template.room_id}].exits[{index}].target_entry",
                        (
                            f"entry '{exit_config.target_entry}' in room "
                            f"'{exit_config.target_room_id}' has no valid non-wall spawn tile"
                        ),
                    )
                continue
            if exit_config.target_entry not in target_template.spawns:
                raise MapValidationError(
                    room_file,
                    f"rooms[{template.room_id}].exits[{index}].target_entry",
                    (
                        f"unknown entry '{exit_config.target_entry}' in room "
                        f"'{exit_config.target_room_id}'"
                    ),
                )
