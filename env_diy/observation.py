from __future__ import annotations

import numpy as np

from .constants import GRID_HEIGHT, GRID_WIDTH, TILE_SIZE
from .entities import Player
from .room import RoomDefinition


def _pixel_to_grid(center_x: float, center_y: float) -> tuple[int, int]:
    gx = int(center_x // TILE_SIZE)
    gy = int(center_y // TILE_SIZE)
    gx = max(0, min(GRID_WIDTH - 1, gx))
    gy = max(0, min(GRID_HEIGHT - 1, gy))
    return gx, gy


def downsampled_observation(room: RoomDefinition, player: Player) -> np.ndarray:
    """Build an 8x10 semantic matrix from the 128x160 pixel space.

    Matrix axis order is [x, y], so shape is (8, 10).
    Semantic values:
      0 empty, 1 wall, 2 player, 3 monster
    """

    observation = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.uint8)

    for gx, gy in room.walls:
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            observation[gx, gy] = 1

    for monster in room.monsters:
        gx, gy = _pixel_to_grid(*monster.center)
        observation[gx, gy] = 3

    player_gx, player_gy = _pixel_to_grid(*player.center)
    observation[player_gx, player_gy] = 2

    return observation
