from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from .constants import TILE_SIZE
from .entities import Chest, NPC
from .monsters import Monster, build_monster_from_dict


@dataclass
class RoomDefinition:
    coord: tuple[int, int]
    player_spawn: tuple[int, int]
    walls: list[tuple[int, int]] = field(default_factory=list)
    chests: list[Chest] = field(default_factory=list)
    npcs: list[NPC] = field(default_factory=list)
    monsters: list[Monster] = field(default_factory=list)

    def wall_rects(self) -> list[pygame.Rect]:
        return [
            pygame.Rect(gx * TILE_SIZE, gy * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            for gx, gy in self.walls
        ]


class RoomManager:
    def __init__(self, room_file: str | Path):
        self.room_file = Path(room_file)
        self.rooms: dict[tuple[int, int], RoomDefinition] = {}
        self.start_room = (0, 0)
        self._load_rooms()

    def _load_rooms(self) -> None:
        raw = json.loads(self.room_file.read_text(encoding="utf-8"))
        start = raw.get("start_room", [0, 0])
        self.start_room = (int(start[0]), int(start[1]))

        for key, payload in raw.get("rooms", {}).items():
            cx, cy = self._parse_coord_key(key)
            self.rooms[(cx, cy)] = self._build_room((cx, cy), payload)

    @staticmethod
    def _parse_coord_key(key: str) -> tuple[int, int]:
        left, right = key.split(",", maxsplit=1)
        return int(left.strip()), int(right.strip())

    def _build_room(self, coord: tuple[int, int], payload: dict) -> RoomDefinition:
        spawn = payload.get("player_spawn", [1, 1])
        player_spawn = (int(spawn[0]), int(spawn[1]))

        walls = [(int(p[0]), int(p[1])) for p in payload.get("walls", [])]

        chests = [
            Chest(int(entry["grid"][0]), int(entry["grid"][1]))
            for entry in payload.get("chests", [])
        ]

        npcs = [
            NPC(
                int(entry["grid"][0]),
                int(entry["grid"][1]),
                str(entry.get("text", "...")),
            )
            for entry in payload.get("npcs", [])
        ]

        monsters = [build_monster_from_dict(entry) for entry in payload.get("monsters", [])]

        return RoomDefinition(
            coord=coord,
            player_spawn=player_spawn,
            walls=walls,
            chests=chests,
            npcs=npcs,
            monsters=monsters,
        )

    def get_room(self, coord: tuple[int, int]) -> RoomDefinition:
        if coord not in self.rooms:
            self.rooms[coord] = RoomDefinition(coord=coord, player_spawn=(1, 1))
        return self.rooms[coord]
