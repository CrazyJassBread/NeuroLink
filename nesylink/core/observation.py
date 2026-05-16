from __future__ import annotations

import numpy as np

from .constants import GRID_HEIGHT, GRID_WIDTH
from .state import PlayerState, inventory_item_codes, tile_from_position_px
from .world.rooms import RoomState


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


def build_observation(room: RoomState, player: PlayerState, max_monster_slots: int) -> dict[str, np.ndarray]:
    grid = room_observation(room, player)
    player_tile = tile_from_position_px(player.position_px, player.size_px)
    monster_positions = np.full((max_monster_slots, 2), -1.0, dtype=np.float32)
    monster_tiles = np.full((max_monster_slots, 2), -1, dtype=np.int32)
    monster_mask = np.zeros((max_monster_slots,), dtype=np.uint8)
    monster_hp = np.zeros((max_monster_slots,), dtype=np.int32)

    for index, monster in enumerate(room.monsters.values()):
        if index >= max_monster_slots:
            break
        monster_positions[index] = np.asarray(monster.position_px, dtype=np.float32)
        monster_tiles[index] = np.asarray(monster.tile_pos, dtype=np.int32)
        monster_mask[index] = 1
        monster_hp[index] = monster.hp

    return {
        "grid": grid,
        "player_position_px": np.asarray(player.position_px, dtype=np.float32),
        "player_tile": np.asarray(player_tile, dtype=np.int32),
        "health": np.asarray([player.health], dtype=np.int32),
        "gold": np.asarray([player.gold], dtype=np.int32),
        "keys": np.asarray([player.keys], dtype=np.int32),
        "inventory_ids": np.asarray(inventory_item_codes(player.items), dtype=np.int32),
        "monsters_position_px": monster_positions,
        "monsters_tile": monster_tiles,
        "monsters_active_mask": monster_mask,
        "monsters_hp": monster_hp,
    }
