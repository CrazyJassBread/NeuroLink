from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..constants import GRID_HEIGHT, GRID_WIDTH
from .schema import (
    LAYOUT_TILES,
    SUPPORTED_EXIT_DIRECTIONS,
    SUPPORTED_EXIT_TYPES,
    SUPPORTED_OBJECT_KINDS,
    SUPPORTED_REQUIREMENT_KEYS,
    ExitConfig,
    GridPos,
    MapValidationError,
    ObjectConfig,
    RoomTemplate,
    exit_tiles_for_direction,
    opposite_direction,
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise MapValidationError(
            path,
            "",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc


def is_standalone_room(raw: dict[str, Any]) -> bool:
    return "layout" in raw and "spawns" in raw


def build_room_template(room_path: Path, payload: dict[str, Any]) -> RoomTemplate:
    if not isinstance(payload, dict):
        raise MapValidationError(room_path, "", "room file must contain a JSON object")

    room_id = _require_string(payload.get("id"), "id", room_path)
    coord = _require_room_coord(payload.get("coord"), "coord", room_path)
    walls = _parse_layout(payload.get("layout"), room_path)
    wall_set = frozenset(walls)

    raw_spawns = payload.get("spawns")
    if not isinstance(raw_spawns, dict) or not raw_spawns:
        raise MapValidationError(room_path, "spawns", "must be a non-empty object")

    spawns: dict[str, GridPos] = {}
    for spawn_name, spawn_value in raw_spawns.items():
        if not isinstance(spawn_name, str) or not spawn_name.strip():
            raise MapValidationError(room_path, "spawns", "spawn names must be non-empty strings")
        spawn_pos = _require_grid_coord(spawn_value, f"spawns.{spawn_name}", room_path)
        _validate_floor_position(spawn_pos, wall_set, f"spawns.{spawn_name}", room_path)
        spawns[spawn_name] = spawn_pos

    default_spawn_name = payload.get("default_spawn")
    if default_spawn_name is None and len(spawns) == 1:
        default_spawn_name = next(iter(spawns))
    if not isinstance(default_spawn_name, str) or default_spawn_name not in spawns:
        raise MapValidationError(room_path, "default_spawn", f"unknown spawn '{default_spawn_name or 'default'}'")

    raw_objects = payload.get("objects", [])
    if not isinstance(raw_objects, list):
        raise MapValidationError(room_path, "objects", "must be a list")
    objects = _build_objects(raw_objects, wall_set, room_path)
    object_kinds = {entry.object_id: entry.kind for entry in objects}

    raw_exits = payload.get("exits", [])
    if not isinstance(raw_exits, list):
        raise MapValidationError(room_path, "exits", "must be a list")
    exits = _build_exits(raw_exits, wall_set, object_kinds, room_path)

    return RoomTemplate(
        room_id=room_id,
        coord=coord,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        spawns=spawns,
        default_spawn_name=default_spawn_name,
        walls=wall_set,
        objects=tuple(objects),
        exits=tuple(exits),
    )


def _build_objects(
    raw_objects: list[dict[str, Any]],
    wall_tiles: frozenset[GridPos],
    room_path: Path,
) -> list[ObjectConfig]:
    seen_ids: set[str] = set()
    objects: list[ObjectConfig] = []

    for index, entry in enumerate(raw_objects):
        if not isinstance(entry, dict):
            raise MapValidationError(room_path, f"objects[{index}]", "must be an object")

        object_id = _require_string(entry.get("id"), f"objects[{index}].id", room_path)
        if object_id in seen_ids:
            raise MapValidationError(room_path, f"objects[{index}].id", f"duplicate object id '{object_id}'")
        seen_ids.add(object_id)

        kind = _require_string(entry.get("kind"), f"objects[{index}].kind", room_path)
        if kind not in SUPPORTED_OBJECT_KINDS:
            allowed = ", ".join(sorted(SUPPORTED_OBJECT_KINDS))
            raise MapValidationError(
                room_path,
                f"objects[{index}].kind",
                f"unsupported object kind '{kind}', allowed: {allowed}",
            )

        pos = _require_grid_coord(entry.get("pos"), f"objects[{index}].pos", room_path)
        _validate_floor_position(pos, wall_tiles, f"objects[{index}].pos", room_path)

        payload = dict(entry)
        payload.pop("id", None)
        payload.pop("kind", None)
        payload.pop("pos", None)
        objects.append(ObjectConfig(object_id=object_id, kind=kind, pos=pos, payload=payload))

    return objects


def _build_exits(
    raw_exits: list[dict[str, Any]],
    wall_tiles: frozenset[GridPos],
    object_kinds: dict[str, str],
    room_path: Path,
) -> list[ExitConfig]:
    seen_ids: set[str] = set()
    exits: list[ExitConfig] = []

    for index, entry in enumerate(raw_exits):
        if not isinstance(entry, dict):
            raise MapValidationError(room_path, f"exits[{index}]", "must be an object")

        exit_id = _require_string(entry.get("id"), f"exits[{index}].id", room_path)
        if exit_id in seen_ids:
            raise MapValidationError(room_path, f"exits[{index}].id", f"duplicate exit id '{exit_id}'")
        seen_ids.add(exit_id)

        direction = _require_string(entry.get("direction"), f"exits[{index}].direction", room_path).lower()
        if direction not in SUPPORTED_EXIT_DIRECTIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXIT_DIRECTIONS))
            raise MapValidationError(
                room_path,
                f"exits[{index}].direction",
                f"unsupported exit direction '{direction}', allowed: {allowed}",
            )

        tiles = exit_tiles_for_direction(direction)
        for tile_index, tile in enumerate(tiles):
            _validate_floor_position(tile, wall_tiles, f"exits[{index}].tiles[{tile_index}]", room_path)

        target_room_id = _require_string(
            entry.get("target_room"),
            f"exits[{index}].target_room",
            room_path,
        )
        raw_target_entry = entry.get("target_entry")
        if raw_target_entry is None:
            target_entry = opposite_direction(direction)
        else:
            target_entry = _require_string(
                raw_target_entry,
                f"exits[{index}].target_entry",
                room_path,
            )

        exit_type = str(entry.get("type", "normal")).lower()
        if exit_type not in SUPPORTED_EXIT_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_EXIT_TYPES))
            raise MapValidationError(
                room_path,
                f"exits[{index}].type",
                f"unsupported exit type '{exit_type}', allowed: {allowed}",
            )

        raw_requires = entry.get("requires", {})
        if raw_requires is None:
            raw_requires = {}
        if not isinstance(raw_requires, dict):
            raise MapValidationError(room_path, f"exits[{index}].requires", "must be an object")
        requires = _validate_exit_requires(
            raw_requires,
            exit_type,
            object_kinds,
            f"exits[{index}].requires",
            room_path,
        )

        exits.append(
            ExitConfig(
                exit_id=exit_id,
                direction=direction,
                tiles=tiles,
                target_room_id=target_room_id,
                target_entry=target_entry,
                exit_type=exit_type,
                requires=requires,
                blocked_message=str(entry.get("blocked_message", "BLOCKED")),
                success_message=str(entry.get("success_message", "MOVED")),
                complete_task=bool(entry.get("complete_task", False)),
            )
        )

    return exits


def _validate_exit_requires(
    raw_requires: dict[str, Any],
    exit_type: str,
    object_kinds: dict[str, str],
    field_path: str,
    room_path: Path,
) -> dict[str, Any]:
    unknown_keys = sorted(set(raw_requires) - SUPPORTED_REQUIREMENT_KEYS)
    if unknown_keys:
        raise MapValidationError(
            room_path,
            field_path,
            f"unsupported requirement keys: {', '.join(unknown_keys)}",
        )

    requires = dict(raw_requires)
    if exit_type == "normal":
        if requires:
            raise MapValidationError(room_path, field_path, "normal exits cannot declare requirements")
        return {}

    if exit_type == "locked_key":
        key_count = max(1, int(requires.get("key_count", 1)))
        consume_key = bool(requires.get("consume_key", False))
        return {"key_count": key_count, "consume_key": consume_key}

    has_condition = False
    if "button_pressed" in requires:
        button_id = _require_string(requires.get("button_pressed"), f"{field_path}.button_pressed", room_path)
        if object_kinds.get(button_id) != "button":
            raise MapValidationError(
                room_path,
                f"{field_path}.button_pressed",
                f"unknown button '{button_id}' in this room",
            )
        requires["button_pressed"] = button_id
        has_condition = True
    if "item" in requires:
        requires["item"] = _require_string(requires.get("item"), f"{field_path}.item", room_path)
        has_condition = True
    if "all_monsters_defeated" in requires:
        requires["all_monsters_defeated"] = bool(requires["all_monsters_defeated"])
        has_condition = True
    if not has_condition:
        raise MapValidationError(
            room_path,
            field_path,
            "conditional exits must declare at least one supported condition",
        )
    return requires


def _parse_layout(layout: Any, room_path: Path) -> list[GridPos]:
    if not isinstance(layout, list) or not layout:
        raise MapValidationError(room_path, "layout", "must be a non-empty list of strings")
    if len(layout) != GRID_HEIGHT:
        raise MapValidationError(
            room_path,
            "layout",
            f"must contain exactly {GRID_HEIGHT} rows for the dungeon area",
        )
    if not isinstance(layout[0], str) or len(layout[0]) != GRID_WIDTH:
        raise MapValidationError(
            room_path,
            "layout[0]",
            f"must contain exactly {GRID_WIDTH} columns",
        )

    walls: list[GridPos] = []
    for row_index, row in enumerate(layout):
        if not isinstance(row, str):
            raise MapValidationError(room_path, f"layout[{row_index}]", "must be a string")
        if len(row) != GRID_WIDTH:
            raise MapValidationError(room_path, f"layout[{row_index}]", "row width does not match layout[0]")
        for col_index, tile in enumerate(row):
            if tile not in LAYOUT_TILES:
                allowed = ", ".join(sorted(LAYOUT_TILES))
                raise MapValidationError(
                    room_path,
                    f"layout[{row_index}][{col_index}]",
                    f"unsupported tile '{tile}', allowed: {allowed}",
                )
            if tile == "#":
                walls.append((col_index, row_index))
    return walls


def _require_string(value: Any, field_path: str, source_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MapValidationError(source_path, field_path, "must be a non-empty string")
    return value


def _require_room_coord(value: Any, field_path: str, source_path: Path) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MapValidationError(source_path, field_path, "must be a 2-item coordinate")
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise MapValidationError(source_path, field_path, "coordinate values must be integers") from exc


def _require_grid_coord(value: Any, field_path: str, source_path: Path) -> GridPos:
    x, y = _require_room_coord(value, field_path, source_path)
    if not (0 <= x < GRID_WIDTH):
        raise MapValidationError(source_path, field_path, f"column {x} is outside 0..{GRID_WIDTH - 1}")
    if not (0 <= y < GRID_HEIGHT):
        raise MapValidationError(
            source_path,
            field_path,
            f"row {y} is outside dungeon rows 0..{GRID_HEIGHT - 1}; rows {GRID_HEIGHT}..9 are HUD",
        )
    return x, y


def _validate_floor_position(
    pos: GridPos,
    wall_tiles: frozenset[GridPos],
    field_path: str,
    source_path: Path,
) -> None:
    if pos in wall_tiles:
        raise MapValidationError(source_path, field_path, "position overlaps a wall tile")
