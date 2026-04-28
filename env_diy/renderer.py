from __future__ import annotations

import numpy as np

from .constants import (
    COLOR_BUTTON,
    COLOR_CHEST,
    COLOR_CHEST_OPEN,
    COLOR_EXIT,
    COLOR_EXIT_LOCKED,
    COLOR_FRAME_BG,
    COLOR_HEALTH_BAR_BG,
    COLOR_HEALTH_BAR_FILL,
    COLOR_HUD_ACCENT,
    COLOR_HUD_BG,
    COLOR_HUD_PANEL,
    COLOR_MAP_BG,
    COLOR_MAP_GRID,
    COLOR_MONSTER_AMBUSHER,
    COLOR_MONSTER_CHASER,
    COLOR_MONSTER_PATROLLER,
    COLOR_NPC,
    COLOR_PLAYER,
    COLOR_TRAP,
    COLOR_WALL,
    GRID_HEIGHT,
    GRID_WIDTH,
    HUD_PIXEL_Y,
    INTERNAL_HEIGHT,
    INTERNAL_WIDTH,
    TILE_SIZE,
)
from .entities import PlayerState
from .room import RoomState


MONSTER_COLORS = {
    "ambusher": COLOR_MONSTER_AMBUSHER,
    "patroller": COLOR_MONSTER_PATROLLER,
    "chaser": COLOR_MONSTER_CHASER,
}


def render_frame(room: RoomState, player: PlayerState) -> np.ndarray:
    frame = np.zeros((INTERNAL_HEIGHT, INTERNAL_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = COLOR_FRAME_BG

    _draw_map_background(frame)
    _draw_hud_background(frame, player)
    _draw_transitions(frame, room, player)
    _draw_walls(frame, room)
    _draw_objects(frame, room)
    _draw_dynamic_entity(frame, player.position_px, player.size_px, COLOR_PLAYER, padding=2)
    return frame


def _draw_map_background(frame: np.ndarray) -> None:
    frame[:HUD_PIXEL_Y, :] = COLOR_MAP_BG
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            top = row * TILE_SIZE
            left = col * TILE_SIZE
            frame[top : top + TILE_SIZE, left : left + TILE_SIZE] = _checker_color(row, col)
            frame[top, left : left + TILE_SIZE] = COLOR_MAP_GRID
            frame[top : top + TILE_SIZE, left] = COLOR_MAP_GRID


def _checker_color(row: int, col: int) -> tuple[int, int, int]:
    if (row + col) % 2 == 0:
        return COLOR_MAP_BG
    return tuple(min(255, channel + 6) for channel in COLOR_MAP_BG)


def _draw_hud_background(frame: np.ndarray, player: PlayerState) -> None:
    frame[HUD_PIXEL_Y:, :] = COLOR_HUD_BG
    frame[HUD_PIXEL_Y + 2 : INTERNAL_HEIGHT - 2, 2 : INTERNAL_WIDTH - 2] = COLOR_HUD_PANEL
    bar_top = HUD_PIXEL_Y + 5
    bar_left = 6
    bar_width = 56
    frame[bar_top : bar_top + 6, bar_left : bar_left + bar_width] = COLOR_HEALTH_BAR_BG
    fill_ratio = 0.0 if player.max_health <= 0 else player.health / player.max_health
    fill_width = int(round(bar_width * max(0.0, min(1.0, fill_ratio))))
    if fill_width > 0:
        frame[bar_top : bar_top + 6, bar_left : bar_left + fill_width] = COLOR_HEALTH_BAR_FILL
    frame[HUD_PIXEL_Y + 18 : HUD_PIXEL_Y + 20, 6 : INTERNAL_WIDTH - 6] = COLOR_HUD_ACCENT


def _draw_walls(frame: np.ndarray, room: RoomState) -> None:
    for col, row in room.walls:
        _fill_tile(frame, col, row, COLOR_WALL, padding=1)


def _draw_transitions(frame: np.ndarray, room: RoomState, player: PlayerState) -> None:
    for transition in room.transitions:
        color = COLOR_EXIT if player.keys >= transition.requires_key else COLOR_EXIT_LOCKED
        _fill_tile(frame, transition.pos[0], transition.pos[1], color, padding=4)


def _draw_objects(frame: np.ndarray, room: RoomState) -> None:
    for chest in room.chests.values():
        color = COLOR_CHEST_OPEN if chest.is_open else COLOR_CHEST
        _fill_tile(frame, chest.pos[0], chest.pos[1], color, padding=3)

    for npc in room.npcs.values():
        _fill_tile(frame, npc.pos[0], npc.pos[1], COLOR_NPC, padding=3)

    for trap in room.traps.values():
        if trap.is_active:
            _fill_tile(frame, trap.pos[0], trap.pos[1], COLOR_TRAP, padding=5)

    for button in room.buttons.values():
        _fill_tile(frame, button.pos[0], button.pos[1], COLOR_BUTTON, padding=4)

    for monster in room.monsters.values():
        color = MONSTER_COLORS.get(monster.monster_type, COLOR_MONSTER_CHASER)
        _draw_dynamic_entity(frame, monster.position_px, monster.size_px, color, padding=3)


def _draw_dynamic_entity(
    frame: np.ndarray,
    position_px: tuple[float, float],
    size_px: int,
    color: tuple[int, int, int],
    *,
    padding: int = 0,
) -> None:
    left = int(round(position_px[0])) + padding
    top = int(round(position_px[1])) + padding
    right = min(INTERNAL_WIDTH, left + size_px - padding * 2)
    bottom = min(HUD_PIXEL_Y, top + size_px - padding * 2)
    if left < right and top < bottom:
        frame[top:bottom, left:right] = color


def _fill_tile(
    frame: np.ndarray,
    col: int,
    row: int,
    color: tuple[int, int, int],
    *,
    padding: int = 0,
) -> None:
    top = row * TILE_SIZE + padding
    left = col * TILE_SIZE + padding
    bottom = (row + 1) * TILE_SIZE - padding
    right = (col + 1) * TILE_SIZE - padding
    frame[top:bottom, left:right] = color
