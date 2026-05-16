from __future__ import annotations

from pathlib import Path

from .parser import build_room_template, is_standalone_room, read_json
from .schema import (
    EXIT_DIRECTION_TILES,
    ENTRY_SPAWN_TILE_CANDIDATES,
    LAYOUT_TILES,
    OPPOSITE_EXIT_DIRECTIONS,
    SUPPORTED_EXIT_DIRECTIONS,
    SUPPORTED_EXIT_TYPES,
    SUPPORTED_OBJECT_KINDS,
    SUPPORTED_REQUIREMENT_KEYS,
    ExitConfig,
    ExitRuntimeState,
    MapValidationError,
    ObjectConfig,
    RoomState,
    RoomTemplate,
    direction_from_entry_name,
    entry_spawn_tile_candidates,
    exit_tiles_for_direction,
    first_valid_entry_spawn_tile,
    opposite_direction,
)
from .validator import validate_exit_targets
from ..monsters import MonsterState, build_monster_from_dict
from ..state import ButtonState, ChestState, NPCState, TrapState

__all__ = [
    "EXIT_DIRECTION_TILES",
    "ENTRY_SPAWN_TILE_CANDIDATES",
    "LAYOUT_TILES",
    "OPPOSITE_EXIT_DIRECTIONS",
    "SUPPORTED_EXIT_DIRECTIONS",
    "SUPPORTED_EXIT_TYPES",
    "SUPPORTED_OBJECT_KINDS",
    "SUPPORTED_REQUIREMENT_KEYS",
    "ExitConfig",
    "ExitRuntimeState",
    "MapValidationError",
    "ObjectConfig",
    "RoomManager",
    "RoomState",
    "RoomTemplate",
    "direction_from_entry_name",
    "entry_spawn_tile_candidates",
    "exit_tiles_for_direction",
    "first_valid_entry_spawn_tile",
    "opposite_direction",
]


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
        raw = read_json(self.room_file)
        if is_standalone_room(raw):
            self._load_standalone_room(raw)
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
            payload = read_json(room_path)
            template = build_room_template(room_path, payload)
            self._register_template(template, room_path)

        start_room_value = raw.get("start_room")
        if isinstance(start_room_value, str):
            if start_room_value not in self.room_ids:
                raise MapValidationError(self.room_file, "start_room", f"unknown room id '{start_room_value}'")
            self.start_room_id = start_room_value
            self.start_room = self.room_ids[start_room_value]
        else:
            raise MapValidationError(self.room_file, "start_room", "must be a room id string")

        validate_exit_targets(self.room_file, self.room_templates, self.room_ids)

    def _load_standalone_room(self, raw: dict) -> None:
        room_payload = dict(raw)
        room_payload.setdefault("coord", [0, 0])
        template = build_room_template(self.room_file, room_payload)
        self._register_template(template, self.room_file)
        self.start_room_id = template.room_id
        self.start_room = template.coord
        validate_exit_targets(self.room_file, self.room_templates, self.room_ids)

    def _register_template(self, template: RoomTemplate, room_path: Path) -> None:
        if template.coord in self.room_templates:
            raise MapValidationError(room_path, "coord", f"duplicate room coordinate {template.coord}")
        if template.room_id in self.room_ids:
            raise MapValidationError(room_path, "id", f"duplicate room id '{template.room_id}'")
        self.room_templates[template.coord] = template
        self.room_ids[template.room_id] = template.coord
        monster_count = sum(1 for entry in template.objects if entry.kind == "monster")
        self.max_monsters = max(self.max_monsters, monster_count)

    def template_by_room_id(self, room_id: str) -> RoomTemplate:
        coord = self.room_ids[room_id]
        return self.room_templates[coord]

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

    def get_spawn(self, room_id: str, spawn_name: str):
        template = self.template_by_room_id(room_id)
        return template.spawns[spawn_name]

    def coord_for_room_id(self, room_id: str) -> tuple[int, int]:
        return self.room_ids[room_id]

    def reset_room_cache(self) -> None:
        self.rooms = {}
