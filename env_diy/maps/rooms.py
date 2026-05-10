from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.constants import GRID_HEIGHT, GRID_WIDTH
from ..entities import ButtonState, ChestState, GridPos, NPCState, TrapState
from ..entities.monsters import MonsterState, build_monster_from_dict
from .tasks import TaskConfig, parse_task_config


SUPPORTED_OBJECT_KINDS = {
    "button",
    "chest",
    "monster",
    "npc",
    "trap",
}

SUPPORTED_EXIT_DIRECTIONS = {"north", "south", "west", "east"}
SUPPORTED_EXIT_TYPES = {"normal", "locked_key", "conditional"}
SUPPORTED_REQUIREMENT_KEYS = {"key_count", "consume_key", "button_pressed", "item", "all_monsters_defeated"}
LAYOUT_TILES = {"#", "."}

EXIT_DIRECTION_TILES: dict[str, tuple[GridPos, GridPos]] = {
    "north": ((4, 0), (5, 0)),
    "south": ((4, GRID_HEIGHT - 1), (5, GRID_HEIGHT - 1)),
    "west": ((0, 3), (0, 4)),
    "east": ((GRID_WIDTH - 1, 3), (GRID_WIDTH - 1, 4)),
}

OPPOSITE_EXIT_DIRECTIONS = {
    "north": "south",
    "south": "north",
    "west": "east",
    "east": "west",
}

ENTRY_SPAWN_TILE_CANDIDATES: dict[str, tuple[GridPos, GridPos]] = {
    "north": ((4, 1), (5, 1)),
    "south": ((4, GRID_HEIGHT - 2), (5, GRID_HEIGHT - 2)),
    "west": ((1, 3), (1, 4)),
    "east": ((GRID_WIDTH - 2, 3), (GRID_WIDTH - 2, 4)),
}


def exit_tiles_for_direction(direction: str) -> tuple[GridPos, GridPos]:
    return EXIT_DIRECTION_TILES[direction]


def opposite_direction(direction: str) -> str:
    return OPPOSITE_EXIT_DIRECTIONS[direction]


def direction_from_entry_name(entry_name: str) -> str | None:
    normalized = entry_name.strip().lower()
    for direction in SUPPORTED_EXIT_DIRECTIONS:
        if normalized in {direction, f"from_{direction}", f"{direction}_entry"}:
            return direction
    return None


def entry_spawn_tile_candidates(direction: str) -> tuple[GridPos, GridPos]:
    return ENTRY_SPAWN_TILE_CANDIDATES[direction]


def first_valid_entry_spawn_tile(direction: str, wall_tiles: set[GridPos] | frozenset[GridPos]) -> GridPos | None:
    for candidate in entry_spawn_tile_candidates(direction):
        if candidate not in wall_tiles:
            return candidate
    return None


class MapValidationError(ValueError):
    def __init__(self, file_path: str | Path, field_path: str, message: str):
        self.file_path = str(file_path)
        self.field_path = field_path
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        location = self.file_path
        if self.field_path:
            location = f"{location}: {self.field_path}"
        return f"{location} - {self.message}"


@dataclass(frozen=True)
class ObjectConfig:
    object_id: str
    kind: str
    pos: GridPos
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitConfig:
    exit_id: str
    direction: str
    tiles: tuple[GridPos, GridPos]
    target_room_id: str
    target_entry: str
    exit_type: str = "normal"
    requires: dict[str, Any] = field(default_factory=dict)
    blocked_message: str = "BLOCKED"
    success_message: str = "MOVED"
    complete_task: bool = False

    def contains(self, pos: GridPos) -> bool:
        return pos in self.tiles


@dataclass
class ExitRuntimeState:
    unlocked: bool = False
    opened: bool = False


@dataclass(frozen=True)
class RoomTemplate:
    room_id: str
    coord: tuple[int, int]
    width: int
    height: int
    spawns: dict[str, GridPos]
    default_spawn_name: str
    walls: frozenset[GridPos]
    objects: tuple[ObjectConfig, ...]
    exits: tuple[ExitConfig, ...]


@dataclass
class RoomState:
    room_id: str
    coord: tuple[int, int]
    width: int
    height: int
    spawns: dict[str, GridPos]
    default_spawn_name: str
    walls: set[GridPos]
    chests: dict[str, ChestState]
    npcs: dict[str, NPCState]
    traps: dict[str, TrapState]
    buttons: dict[str, ButtonState]
    monsters: dict[str, MonsterState]
    exits: list[ExitConfig]
    exit_states: dict[str, ExitRuntimeState] = field(default_factory=dict)

    def chest_at(self, pos: GridPos) -> ChestState | None:
        for chest in self.chests.values():
            if chest.pos == pos:
                return chest
        return None

    def npc_at(self, pos: GridPos) -> NPCState | None:
        for npc in self.npcs.values():
            if npc.pos == pos:
                return npc
        return None

    def trap_at(self, pos: GridPos) -> TrapState | None:
        for trap in self.traps.values():
            if trap.pos == pos and trap.is_active:
                return trap
        return None

    def button_at(self, pos: GridPos) -> ButtonState | None:
        for button in self.buttons.values():
            if button.pos == pos:
                return button
        return None

    def exit_at(self, pos: GridPos, direction: str) -> ExitConfig | None:
        for exit_config in self.exits:
            if exit_config.direction == direction and exit_config.contains(pos):
                return exit_config
        return None

    def exit_state(self, exit_config: ExitConfig) -> ExitRuntimeState:
        if exit_config.exit_id not in self.exit_states:
            self.exit_states[exit_config.exit_id] = ExitRuntimeState(
                unlocked=exit_config.exit_type != "locked_key",
                opened=exit_config.exit_type != "locked_key",
            )
        return self.exit_states[exit_config.exit_id]

    def blocking_tiles(self) -> set[GridPos]:
        tiles = set(self.walls)
        tiles.update(chest.pos for chest in self.chests.values())
        tiles.update(npc.pos for npc in self.npcs.values())
        return tiles


class RoomManager:
    def __init__(self, room_file: str | Path):
        self.room_file = Path(room_file)
        self.room_templates: dict[tuple[int, int], RoomTemplate] = {}
        self.room_ids: dict[str, tuple[int, int]] = {}
        self.rooms: dict[tuple[int, int], RoomState] = {}
        self.max_monsters = 0
        self.start_room = (0, 0)
        self.start_room_id = ""
        self.task_config: TaskConfig | None = None
        self._load_dungeon()

    def _load_dungeon(self) -> None:
        raw = self._read_json(self.room_file)
        if self._is_single_task_room(raw):
            self._load_single_task_room(raw)
            return

        schema_version = raw.get("schema_version")
        if schema_version != 1:
            raise MapValidationError(self.room_file, "schema_version", "only schema_version=1 is supported")

        room_files = raw.get("room_files", [])
        if not isinstance(room_files, list) or not room_files:
            raise MapValidationError(self.room_file, "room_files", "must be a non-empty list")

        for index, room_ref in enumerate(room_files):
            if not isinstance(room_ref, str) or not room_ref.strip():
                raise MapValidationError(
                    self.room_file,
                    f"room_files[{index}]",
                    "must be a non-empty relative path",
                )
            room_path = (self.room_file.parent / room_ref).resolve()
            payload = self._read_json(room_path)
            template = self._build_room_template(room_path, payload)
            self._register_template(template, room_path)

        start_room_value = raw.get("start_room")
        if isinstance(start_room_value, str):
            if start_room_value not in self.room_ids:
                raise MapValidationError(self.room_file, "start_room", f"unknown room id '{start_room_value}'")
            self.start_room_id = start_room_value
            self.start_room = self.room_ids[start_room_value]
        else:
            raise MapValidationError(self.room_file, "start_room", "must be a room id string")

        self._validate_exit_targets()

    @staticmethod
    def _is_single_task_room(raw: dict[str, Any]) -> bool:
        return "task_id" in raw or "task_type" in raw or "objective" in raw

    def _load_single_task_room(self, raw: dict[str, Any]) -> None:
        self.task_config = parse_task_config(raw, self.room_file, MapValidationError)
        room_payload = dict(raw)
        room_payload["id"] = self.task_config.room_id
        room_payload.setdefault("coord", [0, 0])

        template = self._build_room_template(self.room_file, room_payload)
        self._register_template(template, self.room_file)
        self.start_room_id = template.room_id
        self.start_room = template.coord
        self._validate_exit_targets()
        self._validate_task_config()

    def _register_template(self, template: RoomTemplate, room_path: Path) -> None:
        if template.coord in self.room_templates:
            raise MapValidationError(room_path, "coord", f"duplicate room coordinate {template.coord}")
        if template.room_id in self.room_ids:
            raise MapValidationError(room_path, "id", f"duplicate room id '{template.room_id}'")
        self.room_templates[template.coord] = template
        self.room_ids[template.room_id] = template.coord
        monster_count = sum(1 for entry in template.objects if entry.kind == "monster")
        self.max_monsters = max(self.max_monsters, monster_count)

    def _build_room_template(self, room_path: Path, payload: dict[str, Any]) -> RoomTemplate:
        if not isinstance(payload, dict):
            raise MapValidationError(room_path, "", "room file must contain a JSON object")

        room_id = self._require_string(payload.get("id"), "id", room_path)
        coord = self._require_room_coord(payload.get("coord"), "coord", room_path)
        walls = self._parse_layout(payload.get("layout"), room_path)
        wall_set = frozenset(walls)

        raw_spawns = payload.get("spawns")
        if not isinstance(raw_spawns, dict) or not raw_spawns:
            raise MapValidationError(room_path, "spawns", "must be a non-empty object")

        spawns: dict[str, GridPos] = {}
        for spawn_name, spawn_value in raw_spawns.items():
            if not isinstance(spawn_name, str) or not spawn_name.strip():
                raise MapValidationError(room_path, "spawns", "spawn names must be non-empty strings")
            spawn_pos = self._require_grid_coord(spawn_value, f"spawns.{spawn_name}", room_path)
            self._validate_floor_position(spawn_pos, wall_set, f"spawns.{spawn_name}", room_path)
            spawns[spawn_name] = spawn_pos

        default_spawn_name = payload.get("default_spawn")
        if default_spawn_name is None and len(spawns) == 1:
            default_spawn_name = next(iter(spawns))
        if not isinstance(default_spawn_name, str) or default_spawn_name not in spawns:
            raise MapValidationError(room_path, "default_spawn", f"unknown spawn '{default_spawn_name or 'default'}'")

        raw_objects = payload.get("objects", [])
        if not isinstance(raw_objects, list):
            raise MapValidationError(room_path, "objects", "must be a list")
        objects = self._build_objects(raw_objects, wall_set, room_path)
        object_kinds = {entry.object_id: entry.kind for entry in objects}

        raw_exits = payload.get("exits", [])
        if not isinstance(raw_exits, list):
            raise MapValidationError(room_path, "exits", "must be a list")
        exits = self._build_exits(raw_exits, wall_set, object_kinds, room_path)

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
        self,
        raw_objects: list[dict[str, Any]],
        wall_tiles: frozenset[GridPos],
        room_path: Path,
    ) -> list[ObjectConfig]:
        seen_ids: set[str] = set()
        objects: list[ObjectConfig] = []

        for index, entry in enumerate(raw_objects):
            if not isinstance(entry, dict):
                raise MapValidationError(room_path, f"objects[{index}]", "must be an object")

            object_id = self._require_string(entry.get("id"), f"objects[{index}].id", room_path)
            if object_id in seen_ids:
                raise MapValidationError(room_path, f"objects[{index}].id", f"duplicate object id '{object_id}'")
            seen_ids.add(object_id)

            kind = self._require_string(entry.get("kind"), f"objects[{index}].kind", room_path)
            if kind not in SUPPORTED_OBJECT_KINDS:
                allowed = ", ".join(sorted(SUPPORTED_OBJECT_KINDS))
                raise MapValidationError(
                    room_path,
                    f"objects[{index}].kind",
                    f"unsupported object kind '{kind}', allowed: {allowed}",
                )

            pos = self._require_grid_coord(entry.get("pos"), f"objects[{index}].pos", room_path)
            self._validate_floor_position(pos, wall_tiles, f"objects[{index}].pos", room_path)

            payload = dict(entry)
            payload.pop("id", None)
            payload.pop("kind", None)
            payload.pop("pos", None)
            objects.append(ObjectConfig(object_id=object_id, kind=kind, pos=pos, payload=payload))

        return objects

    def _build_exits(
        self,
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

            exit_id = self._require_string(entry.get("id"), f"exits[{index}].id", room_path)
            if exit_id in seen_ids:
                raise MapValidationError(room_path, f"exits[{index}].id", f"duplicate exit id '{exit_id}'")
            seen_ids.add(exit_id)

            direction = self._require_string(entry.get("direction"), f"exits[{index}].direction", room_path).lower()
            if direction not in SUPPORTED_EXIT_DIRECTIONS:
                allowed = ", ".join(sorted(SUPPORTED_EXIT_DIRECTIONS))
                raise MapValidationError(
                    room_path,
                    f"exits[{index}].direction",
                    f"unsupported exit direction '{direction}', allowed: {allowed}",
                )

            tiles = exit_tiles_for_direction(direction)
            for tile_index, tile in enumerate(tiles):
                self._validate_floor_position(tile, wall_tiles, f"exits[{index}].tiles[{tile_index}]", room_path)

            target_room_id = self._require_string(
                entry.get("target_room"),
                f"exits[{index}].target_room",
                room_path,
            )
            raw_target_entry = entry.get("target_entry")
            if raw_target_entry is None:
                target_entry = opposite_direction(direction)
            else:
                target_entry = self._require_string(
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
            requires = self._validate_exit_requires(
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
        self,
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
            button_id = self._require_string(requires.get("button_pressed"), f"{field_path}.button_pressed", room_path)
            if object_kinds.get(button_id) != "button":
                raise MapValidationError(
                    room_path,
                    f"{field_path}.button_pressed",
                    f"unknown button '{button_id}' in this room",
                )
            requires["button_pressed"] = button_id
            has_condition = True
        if "item" in requires:
            requires["item"] = self._require_string(requires.get("item"), f"{field_path}.item", room_path)
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

    def _validate_exit_targets(self) -> None:
        for template in self.room_templates.values():
            for index, exit_config in enumerate(template.exits):
                if exit_config.target_room_id not in self.room_ids:
                    raise MapValidationError(
                        self.room_file,
                        f"rooms[{template.room_id}].exits[{index}].target_room",
                        f"unknown target room '{exit_config.target_room_id}'",
                    )
                target_template = self.template_by_room_id(exit_config.target_room_id)
                entry_direction = direction_from_entry_name(exit_config.target_entry)
                if entry_direction is not None:
                    entry_tile = first_valid_entry_spawn_tile(entry_direction, target_template.walls)
                    if entry_tile is None:
                        raise MapValidationError(
                            self.room_file,
                            f"rooms[{template.room_id}].exits[{index}].target_entry",
                            (
                                f"entry '{exit_config.target_entry}' in room "
                                f"'{exit_config.target_room_id}' has no valid non-wall spawn tile"
                            ),
                        )
                    continue
                if exit_config.target_entry not in target_template.spawns:
                    raise MapValidationError(
                        self.room_file,
                        f"rooms[{template.room_id}].exits[{index}].target_entry",
                        (
                            f"unknown entry '{exit_config.target_entry}' in room "
                            f"'{exit_config.target_room_id}'"
                        ),
                    )

    def _validate_task_config(self) -> None:
        if self.task_config is None:
            return

        template = self.template_by_room_id(self.task_config.room_id)
        objective = self.task_config.objective
        if objective.target_exit is not None and all(
            exit_config.exit_id != objective.target_exit for exit_config in template.exits
        ):
            raise MapValidationError(
                self.room_file,
                "objective.target_exit",
                f"unknown exit '{objective.target_exit}' in task room '{template.room_id}'",
            )

        monster_ids = {entry.object_id for entry in template.objects if entry.kind == "monster"}
        missing_monsters = sorted(set(objective.target_monsters) - monster_ids)
        if missing_monsters:
            raise MapValidationError(
                self.room_file,
                "objective.target_monsters",
                f"unknown monster ids: {', '.join(missing_monsters)}",
            )

    def template_by_room_id(self, room_id: str) -> RoomTemplate:
        coord = self.room_ids[room_id]
        return self.room_templates[coord]

    def _parse_layout(self, layout: Any, room_path: Path) -> list[GridPos]:
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

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MapValidationError(
                path,
                "",
                f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            ) from exc

    @staticmethod
    def _require_string(value: Any, field_path: str, source_path: Path) -> str:
        if not isinstance(value, str) or not value.strip():
            raise MapValidationError(source_path, field_path, "must be a non-empty string")
        return value

    @staticmethod
    def _require_room_coord(value: Any, field_path: str, source_path: Path) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise MapValidationError(source_path, field_path, "must be a 2-item coordinate")
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError) as exc:
            raise MapValidationError(source_path, field_path, "coordinate values must be integers") from exc

    @staticmethod
    def _require_grid_coord(value: Any, field_path: str, source_path: Path) -> GridPos:
        x, y = RoomManager._require_room_coord(value, field_path, source_path)
        if not (0 <= x < GRID_WIDTH):
            raise MapValidationError(source_path, field_path, f"column {x} is outside 0..{GRID_WIDTH - 1}")
        if not (0 <= y < GRID_HEIGHT):
            raise MapValidationError(
                source_path,
                field_path,
                f"row {y} is outside dungeon rows 0..{GRID_HEIGHT - 1}; rows {GRID_HEIGHT}..9 are HUD",
            )
        return x, y

    @staticmethod
    def _validate_floor_position(
        pos: GridPos,
        wall_tiles: frozenset[GridPos],
        field_path: str,
        source_path: Path,
    ) -> None:
        if pos in wall_tiles:
            raise MapValidationError(source_path, field_path, "position overlaps a wall tile")

    def build_room(self, coord: tuple[int, int]) -> RoomState:
        template = self.room_templates[coord]

        chests: dict[str, ChestState] = {}
        npcs: dict[str, NPCState] = {}
        traps: dict[str, TrapState] = {}
        buttons: dict[str, ButtonState] = {}
        monsters: dict[str, MonsterState] = {}
        exit_states = {
            exit_config.exit_id: ExitRuntimeState(
                unlocked=exit_config.exit_type != "locked_key",
                opened=exit_config.exit_type != "locked_key",
            )
            for exit_config in template.exits
        }

        for entry in template.objects:
            if entry.kind == "chest":
                chests[entry.object_id] = ChestState(
                    chest_id=entry.object_id,
                    pos=entry.pos,
                    loot=dict(entry.payload.get("loot", {})),
                )
            elif entry.kind == "npc":
                npcs[entry.object_id] = NPCState(
                    npc_id=entry.object_id,
                    pos=entry.pos,
                    text=str(entry.payload.get("text", "...")),
                )
            elif entry.kind == "trap":
                traps[entry.object_id] = TrapState(
                    trap_id=entry.object_id,
                    pos=entry.pos,
                    damage=max(1, int(entry.payload.get("damage", 1))),
                    respawn_to=str(entry.payload.get("respawn_to", template.default_spawn_name)),
                    single_use=bool(entry.payload.get("single_use", False)),
                )
            elif entry.kind == "button":
                buttons[entry.object_id] = ButtonState(
                    button_id=entry.object_id,
                    pos=entry.pos,
                    message=str(entry.payload.get("message", "BUTTON")),
                )
            elif entry.kind == "monster":
                monster_data = {
                    "id": entry.object_id,
                    "grid": list(entry.pos),
                    **entry.payload,
                }
                monsters[entry.object_id] = build_monster_from_dict(monster_data)

        return RoomState(
            room_id=template.room_id,
            coord=template.coord,
            width=template.width,
            height=template.height,
            spawns=dict(template.spawns),
            default_spawn_name=template.default_spawn_name,
            walls=set(template.walls),
            chests=chests,
            npcs=npcs,
            traps=traps,
            buttons=buttons,
            monsters=monsters,
            exits=list(template.exits),
            exit_states=exit_states,
        )

    def get_room(self, coord: tuple[int, int]) -> RoomState:
        if coord not in self.rooms:
            self.rooms[coord] = self.build_room(coord)
        return self.rooms[coord]

    def get_spawn(self, room_id: str, spawn_name: str) -> GridPos:
        template = self.template_by_room_id(room_id)
        return template.spawns[spawn_name]

    def coord_for_room_id(self, room_id: str) -> tuple[int, int]:
        return self.room_ids[room_id]

    def reset_room_cache(self) -> None:
        self.rooms = {}
