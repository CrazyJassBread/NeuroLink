from __future__ import annotations

import numpy as np

from ..constants import (
    COLOR_EXIT_CONDITIONAL,
    COLOR_EXIT_LOCKED,
    COLOR_EXIT_NORMAL,
    COLOR_FRAME_BG,
    COLOR_HUD_ACCENT,
    COLOR_HUD_BG,
    COLOR_HUD_PANEL,
    COLOR_MONSTER_AMBUSHER,
    COLOR_MONSTER_CHASER,
    COLOR_MONSTER_PATROLLER,
    COLOR_NPC,
    GRID_HEIGHT,
    GRID_WIDTH,
    HUD_PIXEL_Y,
    INTERNAL_HEIGHT,
    INTERNAL_WIDTH,
)
from ..state import PlayerState
from ..world.rooms import RoomState
from .sprites import (
    draw_button,
    draw_chest,
    draw_exit,
    draw_floor,
    draw_monster,
    draw_npc,
    draw_player,
    draw_status_bar,
    draw_trap,
    draw_wall,
)


MONSTER_COLORS = {
    "ambusher": COLOR_MONSTER_AMBUSHER,
    "patroller": COLOR_MONSTER_PATROLLER,
    "chaser": COLOR_MONSTER_CHASER,
}


def render_frame(room: RoomState, player: PlayerState) -> np.ndarray:
    frame = np.zeros((INTERNAL_HEIGHT, INTERNAL_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = COLOR_FRAME_BG

    _draw_map_background(frame)
    _draw_hud_background(frame)
    _draw_exits(frame, room)
    _draw_walls(frame, room)
    _draw_objects(frame, room)
    draw_player(frame, player)
    _draw_hud_text(frame, room, player)
    return frame


def _draw_map_background(frame: np.ndarray) -> None:
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            draw_floor(frame, col, row)


def _draw_hud_background(frame: np.ndarray) -> None:
    frame[HUD_PIXEL_Y:, :] = COLOR_HUD_BG
    frame[HUD_PIXEL_Y + 2 : INTERNAL_HEIGHT - 2, 1 : INTERNAL_WIDTH - 1] = COLOR_HUD_PANEL
    frame[HUD_PIXEL_Y : HUD_PIXEL_Y + 2, :] = COLOR_HUD_ACCENT


def _draw_walls(frame: np.ndarray, room: RoomState) -> None:
    for col, row in room.walls:
        draw_wall(frame, col, row)


def _draw_exits(frame: np.ndarray, room: RoomState) -> None:
    for exit_config in room.exits:
        if exit_config.exit_type == "locked_key":
            color = COLOR_EXIT_LOCKED
        elif exit_config.exit_type == "conditional":
            color = COLOR_EXIT_CONDITIONAL
        else:
            color = COLOR_EXIT_NORMAL
        opened = room.exit_state(exit_config).opened
        draw_exit(frame, exit_config.tiles, exit_config.exit_type, color, opened=opened)


def _draw_objects(frame: np.ndarray, room: RoomState) -> None:
    for chest in room.chests.values():
        draw_chest(
            frame,
            chest.pos[0],
            chest.pos[1],
            opened=chest.is_open,
            loot_kind=str(chest.loot.get("kind", "")),
        )

    for npc in room.npcs.values():
        draw_npc(frame, npc.pos[0], npc.pos[1], COLOR_NPC)

    for trap in room.traps.values():
        if trap.is_active:
            draw_trap(frame, trap.pos[0], trap.pos[1])

    for button in room.buttons.values():
        draw_button(frame, button.pos[0], button.pos[1], pressed=button.is_pressed)

    for monster in room.monsters.values():
        color = MONSTER_COLORS.get(monster.monster_type, COLOR_MONSTER_CHASER)
        draw_monster(frame, monster.position_px, monster.size_px, monster.monster_type, color)


def _draw_hud_text(frame: np.ndarray, _room: RoomState, player: PlayerState) -> None:
    draw_status_bar(frame, player, y=HUD_PIXEL_Y)
