from __future__ import annotations

import numpy as np

from ..constants import TILE_SIZE
from ..state import PlayerState


Color = tuple[int, int, int]
Rect = tuple[int, int, int, int]


OUTLINE = (8, 8, 16)
HIGHLIGHT = (255, 244, 112)
SHADOW = (42, 45, 88)
FLOOR_LIGHT = (72, 122, 248)
FLOOR_DARK = (36, 82, 206)
FLOOR_DARKER = (24, 52, 138)
WALL_LIGHT = (255, 86, 146)
WALL_MID = (219, 18, 82)
WALL_DARK = (88, 0, 36)
WALL_EDGE = (255, 44, 112)
PLAYER_TUNIC = (36, 198, 72)
PLAYER_TUNIC_LIGHT = (126, 248, 82)
PLAYER_FACE = (240, 154, 52)
PLAYER_HAIR = (86, 42, 18)
MONSTER_EYE = (255, 244, 112)
MONSTER_DARK = (126, 44, 0)
CHEST_WOOD = (152, 82, 36)
CHEST_BAND = (255, 216, 80)
CHEST_OPEN_INNER = (42, 18, 16)
LOCK_COLOR = (255, 216, 80)
KEY_COLOR = (255, 216, 80)
COIN_COLOR = (210, 28, 96)
HEART_COLOR = (204, 16, 72)
HEAL_CROSS = (255, 244, 112)
SPIKE_BASE = (36, 82, 206)
SPIKE_BASE_EDGE = (24, 52, 138)
SPIKE_METAL = (238, 238, 236)
SPIKE_SHADE = (112, 112, 126)
SPIKE_HIGHLIGHT = (255, 255, 255)
BUTTON_UP = (40, 190, 74)
BUTTON_DOWN = (28, 112, 52)
EXIT_GLOW = (255, 244, 112)
DOOR_WOOD = (96, 48, 26)
CONDITIONAL_GLYPH = (255, 216, 80)
TEXT_COLOR = OUTLINE
TEXT_DIM = SHADOW
HUD_BG = (255, 255, 255)
HUD_PANEL = (238, 238, 236)
HUD_DARK = OUTLINE
HUD_RUPEE = (214, 26, 96)
HUD_COIN = (255, 196, 40)
HUD_COIN_LIGHT = (255, 244, 112)
HUD_KEY = (255, 216, 80)
HUD_HEART = (172, 8, 64)
HUD_HEART_LIGHT = (255, 82, 132)
ITEM_WHITE = (238, 238, 236)
ITEM_GRAY = (156, 156, 180)
ITEM_GRAY_DARK = (58, 56, 86)
ITEM_NAVY = (28, 34, 84)
ITEM_BLUE = (58, 138, 224)
ITEM_BLUE_LIGHT = (98, 178, 255)


FONT_3X5: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "001"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
    ":": ("000", "010", "000", "010", "000"),
    ",": ("000", "000", "000", "010", "100"),
    ".": ("000", "000", "000", "000", "010"),
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "/": ("001", "001", "010", "100", "100"),
    " ": ("000", "000", "000", "000", "000"),
}


PLAYER_SPRITES: dict[str, tuple[str, ...]] = {
    "down": (
        "................",
        ".....OOOOOO.....",
        "....OGGGGGGO....",
        "...OGGGGGGGGO...",
        "...OGGOOOGGGO...",
        "...OOFFFFOOO....",
        "...OOFOFOOOO....",
        "....OFFFFOO.....",
        ".....OGGGGO.....",
        "....OGLGGLO.....",
        "...OOGGGGOO.....",
        "...OOGBGGOO.....",
        "...OOGGGGOO.....",
        "....OOO.OOO.....",
        "....OBB.OBB.....",
        "................",
    ),
    "up": (
        "................",
        "......OOOO......",
        ".....OGGGGO.....",
        "....OGGGGGGO....",
        "...OGGGGGGGGO...",
        "...OGGLLLGGGO...",
        "...OGGGHGGGGO...",
        "....OHHHHHO.....",
        ".....OGGGGO.....",
        "....OGLGGLO.....",
        "...OOGGGGOO.....",
        "...OOGBGGOO.....",
        "...OOGGGGOO.....",
        "....OOO.OOO.....",
        "....OBB.OBB.....",
        "................",
    ),
    "right": (
        "................",
        ".....OOOOO......",
        "....OGGGGGO.....",
        "...OGGGGGGGO....",
        "...OGGGGOOOO....",
        "....OOFFFHOO....",
        ".....OFFFOOO....",
        ".....OOFOOO.....",
        "....OOGGGO......",
        "...OOGGLGGO.....",
        "..OOGGGGGGO.....",
        "...OOGBGGOO.....",
        "....OOGGGO......",
        ".....OO.OO......",
        ".....OB.OB......",
        "................",
    ),
}
PLAYER_SPRITES["left"] = tuple(row[::-1] for row in PLAYER_SPRITES["right"])


PLAYER_PALETTE: dict[str, Color] = {
    "O": OUTLINE,
    "G": PLAYER_TUNIC,
    "L": PLAYER_TUNIC_LIGHT,
    "F": PLAYER_FACE,
    "H": PLAYER_HAIR,
    "B": SHADOW,
}


MONSTER_SPRITES: dict[str, tuple[str, ...]] = {
    "chaser": (
        "................",
        "......O..O......",
        "...O..OOOO..O...",
        "..OMOOMMMMOOMO..",
        "..OMMMMMMMMMMO..",
        ".OMMEOOMMEOOMMO.",
        ".OMMEOOMMEOOMMO.",
        "..OMMMMMMMMMMO..",
        "...OMMOOOOOMMO..",
        "....OMMMMMMO....",
        "...OOO....OOO...",
        "..OO........OO..",
        "................",
        "................",
        "................",
        "................",
    ),
    "patroller": (
        "................",
        "................",
        "..OO......OO....",
        ".OMMO....OMMO...",
        "OMMMMO..OMMMMO..",
        "OMMMMMOOMMMMMO..",
        ".OMMMEOOEMMMO...",
        "..OMMMOOMMMO....",
        "....OMMMMMO.....",
        ".....OHHHO......",
        "......OOO.......",
        "................",
        "................",
        "................",
        "................",
        "................",
    ),
    "ambusher": (
        "................",
        "................",
        ".....OOOOOO.....",
        "...OOMMMMMMOO...",
        "..OMMMMMMMMMMO..",
        "..OMMOOMMOOMMO..",
        ".OMMMEOOOEMMMO..",
        ".OMMMMMMMMMMMO..",
        "..OMMHHHHHMMO...",
        "...OMMMMMMMO....",
        "..OOO....OOO....",
        ".OO........OO...",
        "................",
        "................",
        "................",
        "................",
    ),
}


SHIELD_ICON: tuple[str, ...] = (
    "..KKK...KKK..",
    ".KKGG...GGKK.",
    ".KGGGKKKGGGK.",
    ".KGGGGGGGGGK.",
    ".KGGKKKKKGGK.",
    ".KGGKBBBKGGK.",
    ".KGGKBBBKGGK.",
    ".KGGKBBBKGGK.",
    ".KGGKBBBKGGK.",
    ".KGGKBBBKGGK.",
    ".KGGKKKKKGGK.",
    ".KKGGGGGGGKK.",
    "..KKGGGGGKK..",
    "...KKGGGKK...",
    "....KKKKK....",
)


SWORD_ICON: tuple[str, ...] = (
    ".....KKK.....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "....KYYYK....",
    "..KKGGGGGKK..",
    ".KK..GGG..KK.",
    ".....GGG.....",
    ".....GGG.....",
    "....KNNNK....",
    "....KNNNK....",
    ".....KKK.....",
)


ITEM_PALETTE: dict[str, Color] = {
    "K": OUTLINE,
    "G": ITEM_GRAY,
    "W": ITEM_WHITE,
    "B": ITEM_BLUE,
}


SWORD_PALETTE: dict[str, Color] = {
    "K": OUTLINE,
    "Y": HIGHLIGHT,
    "G": ITEM_GRAY,
    "N": ITEM_NAVY,
}


def tile_rect(col: int, row: int, padding: int = 0) -> Rect:
    left = col * TILE_SIZE + padding
    top = row * TILE_SIZE + padding
    return left, top, TILE_SIZE - padding * 2, TILE_SIZE - padding * 2


def draw_floor(frame: np.ndarray, col: int, row: int) -> None:
    rect = tile_rect(col, row)
    fill_rect(frame, rect, FLOOR_LIGHT)
    left, top, width, height = rect
    pebble_shift = (col * 5 + row * 3) % 4
    for pebble_left, pebble_top in ((1, 1), (8, 1), (4, 6), (12, 8), (2, 12), (9, 13)):
        x = left + pebble_left + pebble_shift % 2
        y = top + pebble_top
        fill_rect(frame, (x + 1, y, 4, 1), FLOOR_DARK)
        fill_rect(frame, (x, y + 1, 6, 3), FLOOR_DARK)
        fill_rect(frame, (x + 1, y + 4, 4, 1), FLOOR_DARKER)
        fill_rect(frame, (x + 4, y + 2, 1, 1), FLOOR_LIGHT)


def draw_wall(frame: np.ndarray, col: int, row: int) -> None:
    rect = tile_rect(col, row)
    fill_rect(frame, rect, WALL_MID)
    left, top, width, height = rect
    draw_rect_outline(frame, rect, OUTLINE)
    fill_rect(frame, (left + 2, top + 2, width - 4, 3), WALL_LIGHT)
    fill_rect(frame, (left + 3, top + 5, width - 6, 2), WALL_EDGE)
    fill_rect(frame, (left + 2, top + 11, width - 4, 2), WALL_DARK)
    fill_rect(frame, (left + 5, top + 7, 2, 5), WALL_DARK)
    fill_rect(frame, (left + 11, top + 7, 2, 5), WALL_DARK)
    fill_rect(frame, (left + 4, top + 3, 8, 1), HIGHLIGHT)


def draw_room_frame(frame: np.ndarray, map_bottom: int) -> None:
    width = frame.shape[1]
    fill_rect(frame, (0, 0, width, 4), WALL_LIGHT)
    fill_rect(frame, (0, map_bottom - 4, width, 4), WALL_DARK)
    fill_rect(frame, (0, 0, 4, map_bottom), WALL_LIGHT)
    fill_rect(frame, (width - 4, 0, 4, map_bottom), WALL_DARK)
    fill_rect(frame, (4, 4, width - 8, 2), OUTLINE)
    fill_rect(frame, (4, map_bottom - 6, width - 8, 2), OUTLINE)
    fill_rect(frame, (4, 4, 2, map_bottom - 8), OUTLINE)
    fill_rect(frame, (width - 6, 4, 2, map_bottom - 8), OUTLINE)
    for x in range(14, width - 14, 14):
        fill_rect(frame, (x, 1, 4, 2), WALL_DARK)
        fill_rect(frame, (x, map_bottom - 3, 4, 2), WALL_LIGHT)


def draw_player(frame: np.ndarray, player: PlayerState) -> None:
    left, top, _, _ = _dynamic_rect(player.position_px, player.size_px)
    facing = player.action_facing or player.facing
    draw_player_sprite(frame, left, top, facing)
    if player.action_ticks_remaining > 0 and player.action_item == "shield":
        draw_player_shield(frame, left, top, facing)
    if player.action_ticks_remaining > 0 and player.action_item == "sword":
        draw_player_sword(frame, left, top, facing)


def draw_player_front(frame: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    draw_player_sprite(frame, left, top, "down")


def draw_player_back(frame: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    draw_player_sprite(frame, left, top, "up")


def draw_player_side(frame: np.ndarray, left: int, top: int, width: int, height: int, *, facing: str) -> None:
    draw_player_sprite(frame, left, top, facing)


def draw_player_sprite(frame: np.ndarray, left: int, top: int, facing: str) -> None:
    sprite = PLAYER_SPRITES.get(facing, PLAYER_SPRITES["down"])
    draw_pixel_art(frame, sprite, left, top, PLAYER_PALETTE)


def draw_player_shield(frame: np.ndarray, left: int, top: int, facing: str) -> None:
    if facing == "left":
        draw_small_side_shield(frame, left, top, x_offset=1)
        return
    if facing == "right":
        draw_small_side_shield(frame, left, top, x_offset=11)
        return
    if facing == "up":
        draw_small_front_shield(frame, left + 5, top + 1)
        return
    draw_small_front_shield(frame, left + 5, top + 10)


def draw_player_sword(frame: np.ndarray, left: int, top: int, facing: str) -> None:
    if facing == "left":
        fill_rect(frame, (left - 7, top + 7, 7, 2), ITEM_GRAY)
        fill_rect(frame, (left - 7, top + 8, 7, 1), ITEM_WHITE)
        fill_rect(frame, (left - 1, top + 6, 2, 4), ITEM_GRAY_DARK)
        return
    if facing == "right":
        fill_rect(frame, (left + 16, top + 7, 7, 2), ITEM_GRAY)
        fill_rect(frame, (left + 16, top + 8, 7, 1), ITEM_WHITE)
        fill_rect(frame, (left + 15, top + 6, 2, 4), ITEM_GRAY_DARK)
        return
    if facing == "up":
        fill_rect(frame, (left + 7, top - 8, 2, 8), ITEM_GRAY)
        fill_rect(frame, (left + 8, top - 8, 1, 8), ITEM_WHITE)
        fill_rect(frame, (left + 5, top - 1, 6, 2), ITEM_GRAY_DARK)
        return
    fill_rect(frame, (left + 7, top + 16, 2, 8), ITEM_GRAY)
    fill_rect(frame, (left + 8, top + 16, 1, 8), ITEM_WHITE)
    fill_rect(frame, (left + 5, top + 15, 6, 2), ITEM_GRAY_DARK)


def draw_small_front_shield(frame: np.ndarray, left: int, top: int) -> None:
    fill_rect(frame, (left, top, 6, 1), ITEM_GRAY_DARK)
    fill_rect(frame, (left, top + 1, 6, 1), ITEM_WHITE)
    fill_rect(frame, (left + 1, top + 2, 4, 4), ITEM_BLUE)
    fill_rect(frame, (left, top + 2, 1, 4), ITEM_GRAY)
    fill_rect(frame, (left + 5, top + 2, 1, 4), ITEM_GRAY_DARK)
    fill_rect(frame, (left + 2, top + 6, 2, 1), ITEM_GRAY_DARK)


def draw_small_side_shield(frame: np.ndarray, left: int, top: int, *, x_offset: int) -> None:
    fill_rect(frame, (left + x_offset, top + 5, 4, 1), ITEM_WHITE)
    fill_rect(frame, (left + x_offset, top + 6, 4, 6), ITEM_GRAY)
    fill_rect(frame, (left + x_offset + 1, top + 7, 2, 4), ITEM_BLUE)
    fill_rect(frame, (left + x_offset + 3, top + 6, 1, 6), ITEM_GRAY_DARK)
    fill_rect(frame, (left + x_offset + 1, top + 12, 2, 1), ITEM_GRAY_DARK)


def draw_monster(
    frame: np.ndarray,
    position_px: tuple[float, float],
    size_px: int,
    monster_type: str,
    color: Color,
) -> None:
    left, top, _, _ = _dynamic_rect(position_px, size_px)
    sprite = MONSTER_SPRITES.get(monster_type, MONSTER_SPRITES["chaser"])
    palette = {
        "O": OUTLINE,
        "M": color,
        "H": MONSTER_DARK,
        "E": MONSTER_EYE,
    }
    draw_pixel_art(frame, sprite, left, top, palette)


def draw_chest(frame: np.ndarray, col: int, row: int, *, opened: bool, loot_kind: str | None = None) -> None:
    left, top, _, _ = tile_rect(col, row)
    fill_rect(frame, (left + 2, top + 5, 12, 8), CHEST_WOOD)
    draw_rect_outline(frame, (left + 2, top + 5, 12, 8), OUTLINE)
    if opened:
        fill_rect(frame, (left + 3, top + 3, 10, 4), CHEST_OPEN_INNER)
        fill_rect(frame, (left + 3, top + 2, 10, 2), CHEST_BAND)
    else:
        fill_rect(frame, (left + 2, top + 4, 12, 3), CHEST_BAND)
    fill_rect(frame, (left + 7, top + 7, 2, 3), LOCK_COLOR)
    if loot_kind:
        draw_loot_icon(frame, (left + 10, top + 2), loot_kind)


def draw_loot_icon(frame: np.ndarray, pos: tuple[int, int], kind: str) -> None:
    if kind == "key":
        draw_key(frame, pos)
    elif kind in {"gold", "coin"}:
        draw_coin(frame, pos)
    elif kind in {"heal", "potion", "heart"}:
        draw_heal(frame, pos)


def draw_key(frame: np.ndarray, pos: tuple[int, int]) -> None:
    left, top = pos
    fill_rect(frame, (left, top + 2, 3, 3), KEY_COLOR)
    fill_rect(frame, (left + 3, top + 3, 5, 1), KEY_COLOR)
    fill_rect(frame, (left + 6, top + 4, 1, 2), KEY_COLOR)
    fill_rect(frame, (left + 8, top + 4, 1, 2), KEY_COLOR)
    fill_rect(frame, (left + 1, top + 3, 1, 1), OUTLINE)


def draw_coin(frame: np.ndarray, pos: tuple[int, int]) -> None:
    left, top = pos
    fill_rect(frame, (left + 2, top, 3, 1), COIN_COLOR)
    fill_rect(frame, (left + 1, top + 1, 5, 4), COIN_COLOR)
    fill_rect(frame, (left + 2, top + 5, 3, 1), COIN_COLOR)
    fill_rect(frame, (left + 3, top + 1, 1, 4), HIGHLIGHT)


def draw_heal(frame: np.ndarray, pos: tuple[int, int]) -> None:
    left, top = pos
    fill_rect(frame, (left + 1, top + 1, 2, 2), HEART_COLOR)
    fill_rect(frame, (left + 4, top + 1, 2, 2), HEART_COLOR)
    fill_rect(frame, (left, top + 3, 7, 2), HEART_COLOR)
    fill_rect(frame, (left + 2, top + 5, 3, 1), HEART_COLOR)
    fill_rect(frame, (left + 3, top + 2, 1, 3), HEAL_CROSS)


def draw_trap(frame: np.ndarray, col: int, row: int) -> None:
    left, top, _, _ = tile_rect(col, row)
    fill_rect(frame, (left + 1, top + 12, TILE_SIZE - 2, 2), SPIKE_BASE_EDGE)
    fill_rect(frame, (left + 2, top + 12, TILE_SIZE - 4, 1), SPIKE_BASE)
    for spike_left in (2, 5, 8, 11):
        draw_triangle_up(frame, left + spike_left, top + 7, 3, 6, SPIKE_BASE_EDGE)
        draw_triangle_up(frame, left + spike_left + 1, top + 8, 1, 4, SPIKE_METAL)
        fill_rect(frame, (left + spike_left + 2, top + 10, 1, 2), SPIKE_SHADE)
        fill_rect(frame, (left + spike_left + 1, top + 8, 1, 1), SPIKE_HIGHLIGHT)


def draw_button(frame: np.ndarray, col: int, row: int, *, pressed: bool) -> None:
    left, top, _, _ = tile_rect(col, row)
    fill_rect(frame, (left + 3, top + 9, 10, 4), OUTLINE)
    if pressed:
        fill_rect(frame, (left + 4, top + 7, 8, 4), BUTTON_DOWN)
        fill_rect(frame, (left + 5, top + 7, 6, 1), (86, 146, 104))
    else:
        fill_rect(frame, (left + 4, top + 5, 8, 6), BUTTON_UP)
        fill_rect(frame, (left + 5, top + 5, 6, 1), HIGHLIGHT)
    draw_rect_outline(frame, (left + 4, top + (7 if pressed else 5), 8, 4 if pressed else 6), OUTLINE)


def draw_npc(frame: np.ndarray, col: int, row: int, color: Color) -> None:
    left, top, _, _ = tile_rect(col, row)
    fill_rect(frame, (left + 4, top + 3, 8, 10), color)
    draw_rect_outline(frame, (left + 4, top + 3, 8, 10), OUTLINE)
    fill_rect(frame, (left + 6, top + 6, 1, 1), OUTLINE)
    fill_rect(frame, (left + 9, top + 6, 1, 1), OUTLINE)
    fill_rect(frame, (left + 5, top + 11, 6, 1), HIGHLIGHT)


def draw_exit(
    frame: np.ndarray,
    tiles: tuple[tuple[int, int], tuple[int, int]],
    exit_type: str,
    color: Color,
    *,
    opened: bool = False,
) -> None:
    left = min(tile[0] for tile in tiles) * TILE_SIZE
    top = min(tile[1] for tile in tiles) * TILE_SIZE
    right = (max(tile[0] for tile in tiles) + 1) * TILE_SIZE
    bottom = (max(tile[1] for tile in tiles) + 1) * TILE_SIZE
    width = right - left
    height = bottom - top
    rect = (left + 2, top + 2, width - 4, height - 4)

    if exit_type == "normal":
        fill_rect(frame, rect, OUTLINE)
        draw_rect_outline(frame, rect, WALL_LIGHT)
        if width < height:
            fill_rect(frame, (left + 4, top + 5, max(1, width - 8), height - 10), SHADOW)
            fill_rect(frame, (left + 4, top + 5, 2, height - 10), HIGHLIGHT)
        else:
            fill_rect(frame, (left + 5, top + 4, width - 10, max(1, height - 8)), SHADOW)
            fill_rect(frame, (left + 5, top + 4, width - 10, 2), HIGHLIGHT)
    elif exit_type == "locked_key":
        if opened:
            fill_rect(frame, rect, color)
            fill_rect(frame, (left + 4, top + 4, width - 8, height - 8), EXIT_GLOW)
            draw_rect_outline(frame, rect, OUTLINE)
            lock_left = left + width // 2 + 2
            lock_top = top + height // 2 - 2
            fill_rect(frame, (lock_left, lock_top, 5, 4), LOCK_COLOR)
            fill_rect(frame, (lock_left + 1, lock_top - 4, 4, 2), OUTLINE)
            fill_rect(frame, (lock_left + 4, lock_top - 2, 1, 3), OUTLINE)
        else:
            fill_rect(frame, rect, DOOR_WOOD)
            draw_rect_outline(frame, rect, OUTLINE)
            fill_rect(frame, (left + width // 2 - 3, top + height // 2 - 1, 6, 5), LOCK_COLOR)
            fill_rect(frame, (left + width // 2 - 2, top + height // 2 - 4, 4, 4), OUTLINE)
            fill_rect(frame, (left + width // 2 - 1, top + height // 2 - 3, 2, 3), color)
    else:
        fill_rect(frame, rect, OUTLINE)
        draw_rect_outline(frame, rect, HIGHLIGHT)
        if width < height:
            fill_rect(frame, (left + width // 2 - 2, top + 5, 4, height - 10), CONDITIONAL_GLYPH)
            for y_offset in range(7, height - 7, 5):
                fill_rect(frame, (left + 4, top + y_offset, width - 8, 2), WALL_DARK)
        else:
            fill_rect(frame, (left + 5, top + height // 2 - 2, width - 10, 4), CONDITIONAL_GLYPH)
            for x_offset in range(7, width - 7, 5):
                fill_rect(frame, (left + x_offset, top + 4, 2, height - 8), WALL_DARK)


def draw_hud_text(frame: np.ndarray, line_1: str, line_2: str, *, y: int) -> None:
    draw_text(frame, line_1.upper(), 6, y + 5, TEXT_COLOR)
    draw_text(frame, line_2.upper(), 6, y + 18, TEXT_DIM)


def draw_status_bar(frame: np.ndarray, player: PlayerState, *, y: int) -> None:
    fill_rect(frame, (0, y, frame.shape[1], frame.shape[0] - y), HUD_BG)
    fill_rect(frame, (0, y, frame.shape[1], 1), HUD_PANEL)
    fill_rect(frame, (0, y + 1, frame.shape[1], 2), HUD_DARK)

    draw_text(frame, "B", 1, y + 5, HUD_DARK, scale=2)
    draw_item_bracket(frame, 17, y + 4, 30, 19)
    draw_tool_icon(frame, 25, y + 8, player.equipped_tool_label("B"))
    draw_text(frame, "L-1", 20, y + 24, HUD_DARK, scale=1)

    draw_text(frame, "A", 58, y + 5, HUD_DARK, scale=2)
    draw_item_bracket(frame, 75, y + 4, 30, 19)
    draw_tool_icon(frame, 83, y + 8, player.equipped_tool_label("A"))
    draw_text(frame, "L-1", 78, y + 24, HUD_DARK, scale=1)

    draw_hud_coin(frame, 113, y + 6)
    draw_text(frame, f"{player.gold:03d}", 112, y + 22, HUD_DARK, scale=1)
    draw_hud_key(frame, 140, y + 20)
    draw_text(frame, str(player.keys), 156, y + 24, HUD_DARK, scale=1)

    max_hearts = max(1, player.max_health)
    for index in range(max_hearts):
        heart_x = 129 + index * 6
        heart_y = y + 6 if index < 5 else y + 14
        draw_hud_heart(frame, heart_x, heart_y, filled=index < player.health)


def draw_item_bracket(frame: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    fill_rect(frame, (left, top, 2, height), HUD_DARK)
    fill_rect(frame, (left, top, 7, 2), HUD_DARK)
    fill_rect(frame, (left, top + height - 2, 7, 2), HUD_DARK)
    fill_rect(frame, (left + width - 2, top, 2, height), HUD_DARK)
    fill_rect(frame, (left + width - 7, top, 7, 2), HUD_DARK)
    fill_rect(frame, (left + width - 7, top + height - 2, 7, 2), HUD_DARK)


def draw_tool_icon(frame: np.ndarray, left: int, top: int, tool_name: str) -> None:
    if tool_name == "shield":
        draw_shield_icon(frame, left - 1, top - 1)
    else:
        draw_sword_icon(frame, left + 2, top - 3)


def draw_shield_icon(frame: np.ndarray, left: int, top: int) -> None:
    draw_pixel_art(frame, SHIELD_ICON, left, top, ITEM_PALETTE)


def draw_sword_icon(frame: np.ndarray, left: int, top: int) -> None:
    draw_pixel_art(frame, SWORD_ICON, left, top, SWORD_PALETTE)


def draw_hud_rupee(frame: np.ndarray, left: int, top: int) -> None:
    fill_rect(frame, (left + 2, top, 3, 1), HUD_DARK)
    fill_rect(frame, (left + 1, top + 1, 5, 2), HUD_RUPEE)
    fill_rect(frame, (left, top + 3, 7, 4), HUD_RUPEE)
    fill_rect(frame, (left + 1, top + 7, 5, 2), HUD_RUPEE)
    fill_rect(frame, (left + 2, top + 9, 3, 1), HUD_DARK)
    fill_rect(frame, (left + 3, top + 2, 1, 6), HUD_PANEL)


def draw_hud_coin(frame: np.ndarray, left: int, top: int) -> None:
    fill_rect(frame, (left + 2, top, 5, 1), HUD_DARK)
    fill_rect(frame, (left + 1, top + 1, 7, 2), HUD_DARK)
    fill_rect(frame, (left, top + 3, 9, 5), HUD_DARK)
    fill_rect(frame, (left + 1, top + 8, 7, 2), HUD_DARK)
    fill_rect(frame, (left + 2, top + 10, 5, 1), HUD_DARK)
    fill_rect(frame, (left + 2, top + 2, 5, 1), HUD_COIN_LIGHT)
    fill_rect(frame, (left + 1, top + 3, 7, 5), HUD_COIN)
    fill_rect(frame, (left + 2, top + 8, 5, 1), HUD_COIN)
    fill_rect(frame, (left + 4, top + 3, 1, 5), HUD_COIN_LIGHT)


def draw_hud_key(frame: np.ndarray, left: int, top: int) -> None:
    fill_rect(frame, (left + 1, top, 5, 1), HUD_DARK)
    fill_rect(frame, (left, top + 1, 7, 5), HUD_DARK)
    fill_rect(frame, (left + 2, top + 6, 3, 1), HUD_DARK)
    fill_rect(frame, (left + 5, top + 3, 8, 2), HUD_DARK)
    fill_rect(frame, (left + 10, top + 5, 2, 3), HUD_DARK)
    fill_rect(frame, (left + 13, top + 5, 2, 3), HUD_DARK)
    fill_rect(frame, (left + 2, top + 2, 3, 3), HUD_BG)
    fill_rect(frame, (left + 5, top + 3, 7, 1), HUD_KEY)
    fill_rect(frame, (left + 10, top + 5, 1, 2), HUD_KEY)


def draw_hud_heart(frame: np.ndarray, left: int, top: int, *, filled: bool) -> None:
    color = HUD_HEART if filled else HUD_PANEL
    for x_offset, y_offset in (
        (1, 0),
        (2, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (3, 1),
        (6, 1),
        (0, 2),
        (6, 2),
        (1, 3),
        (5, 3),
        (2, 4),
        (4, 4),
        (3, 5),
    ):
        fill_rect(frame, (left + x_offset, top + y_offset, 1, 1), HUD_DARK)
    for x_offset, y_offset in (
        (1, 1),
        (2, 1),
        (4, 1),
        (5, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (2, 3),
        (3, 3),
        (4, 3),
        (3, 4),
    ):
        fill_rect(frame, (left + x_offset, top + y_offset, 1, 1), color)
    if not filled:
        fill_rect(frame, (left + 2, top + 2, 3, 1), HUD_BG)
    else:
        fill_rect(frame, (left + 2, top + 2, 2, 1), HUD_HEART_LIGHT)


def draw_text(frame: np.ndarray, text: str, x: int, y: int, color: Color, *, scale: int = 1) -> None:
    cursor_x = x
    for char in text:
        glyph = FONT_3X5.get(char, FONT_3X5[" "])
        for row_index, row in enumerate(glyph):
            for col_index, pixel in enumerate(row):
                if pixel == "1":
                    fill_rect(
                        frame,
                        (cursor_x + col_index * scale, y + row_index * scale, scale, scale),
                        color,
                    )
        cursor_x += 4 * scale
        if cursor_x >= frame.shape[1] - 2:
            return


def draw_pixel_art(
    frame: np.ndarray,
    sprite: tuple[str, ...],
    left: int,
    top: int,
    palette: dict[str, Color],
) -> None:
    for y_offset, row in enumerate(sprite):
        for x_offset, key in enumerate(row):
            color = palette.get(key)
            if color is not None:
                fill_rect(frame, (left + x_offset, top + y_offset, 1, 1), color)


def fill_rect(frame: np.ndarray, rect: Rect, color: Color) -> None:
    left, top, width, height = rect
    if width <= 0 or height <= 0:
        return
    right = min(frame.shape[1], left + width)
    bottom = min(frame.shape[0], top + height)
    left = max(0, left)
    top = max(0, top)
    if left < right and top < bottom:
        frame[top:bottom, left:right] = color


def draw_rect_outline(frame: np.ndarray, rect: Rect, color: Color) -> None:
    left, top, width, height = rect
    fill_rect(frame, (left, top, width, 1), color)
    fill_rect(frame, (left, top + height - 1, width, 1), color)
    fill_rect(frame, (left, top, 1, height), color)
    fill_rect(frame, (left + width - 1, top, 1, height), color)


def draw_triangle_up(frame: np.ndarray, left: int, top: int, width: int, height: int, color: Color) -> None:
    center = left + width // 2
    for offset in range(height):
        row_width = max(1, int((offset + 1) * width / height))
        row_left = center - row_width // 2
        fill_rect(frame, (row_left, top + height - offset - 1, row_width, 1), color)


def _dynamic_rect(position_px: tuple[float, float], size_px: int) -> Rect:
    left = int(round(position_px[0]))
    top = int(round(position_px[1]))
    return left, top, size_px, size_px
