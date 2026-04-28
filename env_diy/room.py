from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import GRID_HEIGHT, GRID_WIDTH
from .entities import ButtonState, ChestState, GridPos, NPCState, TrapState
from .monsters import MonsterState, build_monster_from_dict


SUPPORTED_OBJECT_KINDS = {
    "button",
    "chest",
    "monster",
    "npc",
    "trap",
}

SUPPORTED_DIRECTIONS = {"up", "down", "left", "right"}
LAYOUT_TILES = {"#", "."}


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
class TransitionConfig:
    transition_id: str
    pos: GridPos
    direction: str
    target_room_id: str
    target_spawn: str = "default"
    requires_key: int = 0
    locked_message: str = "LOCKED"
    success_message: str = "MOVED"


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
    transitions: tuple[TransitionConfig, ...]


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
    transitions: list[TransitionConfig]

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

    def transition_at(self, pos: GridPos, direction: str) -> TransitionConfig | None:
        for transition in self.transitions:
            if transition.pos == pos and transition.direction == direction:
                return transition
        return None

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
        self._load_dungeon()

    def _load_dungeon(self) -> None:
        raw = self._read_json(self.room_file)
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

        self._validate_transition_targets()

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

        default_spawn_name = str(payload.get("default_spawn", "default"))
        if default_spawn_name not in spawns:
            raise MapValidationError(room_path, "default_spawn", f"unknown spawn '{default_spawn_name}'")

        raw_objects = payload.get("objects", [])
        if not isinstance(raw_objects, list):
            raise MapValidationError(room_path, "objects", "must be a list")
        objects = self._build_objects(raw_objects, wall_set, room_path)

        raw_transitions = payload.get("transitions", [])
        if not isinstance(raw_transitions, list):
            raise MapValidationError(room_path, "transitions", "must be a list")
        transitions = self._build_transitions(raw_transitions, wall_set, room_path)

        return RoomTemplate(
            room_id=room_id,
            coord=coord,
            width=GRID_WIDTH,
            height=GRID_HEIGHT,
            spawns=spawns,
            default_spawn_name=default_spawn_name,
            walls=wall_set,
            objects=tuple(objects),
            transitions=tuple(transitions),
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

    def _build_transitions(
        self,
        raw_transitions: list[dict[str, Any]],
        wall_tiles: frozenset[GridPos],
        room_path: Path,
    ) -> list[TransitionConfig]:
        seen_ids: set[str] = set()
        transitions: list[TransitionConfig] = []

        for index, entry in enumerate(raw_transitions):
            if not isinstance(entry, dict):
                raise MapValidationError(room_path, f"transitions[{index}]", "must be an object")

            transition_id = self._require_string(entry.get("id"), f"transitions[{index}].id", room_path)
            if transition_id in seen_ids:
                raise MapValidationError(
                    room_path,
                    f"transitions[{index}].id",
                    f"duplicate transition id '{transition_id}'",
                )
            seen_ids.add(transition_id)

            pos = self._require_grid_coord(entry.get("pos"), f"transitions[{index}].pos", room_path)
            self._validate_floor_position(pos, wall_tiles, f"transitions[{index}].pos", room_path)

            direction = self._require_string(entry.get("direction"), f"transitions[{index}].direction", room_path)
            if direction not in SUPPORTED_DIRECTIONS:
                raise MapValidationError(
                    room_path,
                    f"transitions[{index}].direction",
                    f"unsupported direction '{direction}'",
                )
            self._validate_transition_edge(pos, direction, f"transitions[{index}].pos", room_path)

            target_room_id = self._require_string(
                entry.get("target_room"),
                f"transitions[{index}].target_room",
                room_path,
            )
            target_spawn = self._require_string(
                entry.get("target_spawn", "default"),
                f"transitions[{index}].target_spawn",
                room_path,
            )
            requires_key = max(0, int(entry.get("requires_key", 0)))
            locked_message = str(entry.get("locked_message", "LOCKED"))
            success_message = str(entry.get("success_message", "MOVED"))

            transitions.append(
                TransitionConfig(
                    transition_id=transition_id,
                    pos=pos,
                    direction=direction,
                    target_room_id=target_room_id,
                    target_spawn=target_spawn,
                    requires_key=requires_key,
                    locked_message=locked_message,
                    success_message=success_message,
                )
            )

        return transitions

    def _validate_transition_targets(self) -> None:
        for template in self.room_templates.values():
            for index, transition in enumerate(template.transitions):
                if transition.target_room_id not in self.room_ids:
                    raise MapValidationError(
                        self.room_file,
                        f"rooms[{template.room_id}].transitions[{index}].target_room",
                        f"unknown target room '{transition.target_room_id}'",
                    )
                target_template = self.template_by_room_id(transition.target_room_id)
                if transition.target_spawn not in target_template.spawns:
                    raise MapValidationError(
                        self.room_file,
                        f"rooms[{template.room_id}].transitions[{index}].target_spawn",
                        (
                            f"unknown spawn '{transition.target_spawn}' in room "
                            f"'{transition.target_room_id}'"
                        ),
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

    @staticmethod
    def _validate_transition_edge(pos: GridPos, direction: str, field_path: str, source_path: Path) -> None:
        x, y = pos
        valid = (
            (direction == "left" and x == 0)
            or (direction == "right" and x == GRID_WIDTH - 1)
            or (direction == "up" and y == 0)
            or (direction == "down" and y == GRID_HEIGHT - 1)
        )
        if not valid:
            raise MapValidationError(
                source_path,
                field_path,
                f"transition direction '{direction}' requires the tile to be on the matching room edge",
            )

    def build_room(self, coord: tuple[int, int]) -> RoomState:
        template = self.room_templates[coord]

        chests: dict[str, ChestState] = {}
        npcs: dict[str, NPCState] = {}
        traps: dict[str, TrapState] = {}
        buttons: dict[str, ButtonState] = {}
        monsters: dict[str, MonsterState] = {}

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
            transitions=list(template.transitions),
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
        self.rooms.clear()
