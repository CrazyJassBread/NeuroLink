from __future__ import annotations

import numpy as np

from .constants import GRID_HEIGHT, GRID_WIDTH
from .entities import PlayerState, tile_from_position_px
from .room import RoomState


TILE_EMPTY = 0
TILE_WALL = 1
TILE_PLAYER = 2
TILE_MONSTER = 3
TILE_CHEST = 4
TILE_EXIT = 5
TILE_TRAP = 6
TILE_BUTTON = 7
TILE_NPC = 8


def room_observation(room: RoomState, player: PlayerState) -> np.ndarray:
    """Build a stable 8x10 semantic grid in row-major order."""

    observation = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

    for col, row in room.walls:
        observation[row, col] = TILE_WALL

    for chest in room.chests.values():
        if not chest.is_open:
            observation[chest.pos[1], chest.pos[0]] = TILE_CHEST

    for npc in room.npcs.values():
        observation[npc.pos[1], npc.pos[0]] = TILE_NPC

    for trap in room.traps.values():
        if trap.is_active:
            observation[trap.pos[1], trap.pos[0]] = TILE_TRAP

    for button in room.buttons.values():
        observation[button.pos[1], button.pos[0]] = TILE_BUTTON

    for exit_config in room.exits:
        for tile in exit_config.tiles:
            observation[tile[1], tile[0]] = TILE_EXIT

    for monster in room.monsters.values():
        monster_tile = monster.tile_pos
        observation[monster_tile[1], monster_tile[0]] = TILE_MONSTER

    player_tile = tile_from_position_px(player.position_px, player.size_px)
    observation[player_tile[1], player_tile[0]] = TILE_PLAYER
    return observation
