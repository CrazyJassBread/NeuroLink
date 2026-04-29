from __future__ import annotations

import numpy as np

from .constants import TILE_SIZE


Color = tuple[int, int, int]
Rect = tuple[int, int, int, int]


OUTLINE = (8, 10, 12)
HIGHLIGHT = (252, 242, 184)
SHADOW = (34, 28, 28)
FLOOR_LIGHT = (31, 43, 40)
FLOOR_DARK = (22, 31, 30)
WALL_LIGHT = (100, 122, 116)
WALL_DARK = (48, 62, 60)
PLAYER_TUNIC = (70, 176, 112)
PLAYER_FACE = (238, 198, 142)
PLAYER_HAIR = (72, 46, 32)
SLIME_EYE = (246, 246, 230)
CHEST_WOOD = (168, 94, 44)
CHEST_BAND = (232, 178, 76)
CHEST_OPEN_INNER = (48, 34, 30)
LOCK_COLOR = (238, 196, 84)
KEY_COLOR = (244, 210, 84)
COIN_COLOR = (248, 204, 68)
HEART_COLOR = (226, 68, 94)
HEAL_CROSS = (248, 248, 236)
TRAP_METAL = (186, 196, 202)
TRAP_WARNING = (232, 70, 78)
BUTTON_UP = (96, 186, 112)
BUTTON_DOWN = (58, 116, 84)
EXIT_GLOW = (176, 236, 210)
DOOR_WOOD = (126, 82, 48)
CONDITIONAL_GLYPH = (196, 206, 255)
TEXT_COLOR = (232, 236, 238)
TEXT_DIM = (162, 172, 176)


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
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "/": ("001", "001", "010", "100", "100"),
    " ": ("000", "000", "000", "000", "000"),
}


def tile_rect(col: int, row: int, padding: int = 0) -> Rect:
    left = col * TILE_SIZE + padding
    top = row * TILE_SIZE + padding
    return left, top, TILE_SIZE - padding * 2, TILE_SIZE - padding * 2


def draw_floor(frame: np.ndarray, col: int, row: int) -> None:
    rect = tile_rect(col, row)
    color = FLOOR_LIGHT if (col + row) % 2 == 0 else FLOOR_DARK
    fill_rect(frame, rect, color)
    left, top, width, height = rect
    frame[top, left : left + width] = (42, 58, 55)
    frame[top : top + height, left] = (42, 58, 55)
    if (col * 3 + row) % 4 == 0:
        fill_rect(frame, (left + 10, top + 4, 2, 1), (38, 52, 49))


def draw_wall(frame: np.ndarray, col: int, row: int) -> None:
    rect = tile_rect(col, row, 1)
    fill_rect(frame, rect, WALL_DARK)
    left, top, width, height = rect
    fill_rect(frame, (left + 1, top + 1, width - 2, 5), WALL_LIGHT)
    fill_rect(frame, (left + 2, top + 8, width - 3, 2), (78, 96, 91))
    draw_rect_outline(frame, rect, OUTLINE)


def draw_player(frame: np.ndarray, position_px: tuple[float, float], size_px: int) -> None:
    left, top, width, height = _dynamic_rect(position_px, size_px)
    draw_rect_outline(frame, (left + 3, top + 5, width - 6, height - 3), OUTLINE)
    fill_rect(frame, (left + 5, top + 6, 6, 7), PLAYER_TUNIC)
    fill_rect(frame, (left + 6, top + 2, 5, 5), PLAYER_FACE)
    fill_rect(frame, (left + 5, top + 1, 6, 3), PLAYER_HAIR)
    fill_rect(frame, (left + 6, top + 4, 1, 1), OUTLINE)
    fill_rect(frame, (left + 10, top + 7, 3, 2), HIGHLIGHT)
    fill_rect(frame, (left + 3, top + 12, 3, 2), SHADOW)
    fill_rect(frame, (left + 9, top + 12, 3, 2), SHADOW)


def draw_monster(
    frame: np.ndarray,
    position_px: tuple[float, float],
    size_px: int,
    monster_type: str,
    color: Color,
) -> None:
    left, top, width, height = _dynamic_rect(position_px, size_px)
    body = (left + 2, top + 5, width - 4, height - 6)
    fill_rect(frame, body, color)
    draw_rect_outline(frame, body, OUTLINE)
    if monster_type == "ambusher":
        fill_rect(frame, (left + 2, top + 3, 3, 3), color)
        fill_rect(frame, (left + width - 5, top + 3, 3, 3), color)
    elif monster_type == "patroller":
        fill_rect(frame, (left + 3, top + 3, width - 6, 3), color)
        fill_rect(frame, (left + 5, top + 2, width - 10, 1), HIGHLIGHT)
    else:
        fill_rect(frame, (left + 4, top + 4, width - 8, 2), color)
    fill_rect(frame, (left + 5, top + 8, 2, 2), SLIME_EYE)
    fill_rect(frame, (left + 10, top + 8, 2, 2), SLIME_EYE)
    fill_rect(frame, (left + 6, top + 9, 1, 1), OUTLINE)
    fill_rect(frame, (left + 10, top + 9, 1, 1), OUTLINE)


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
    fill_rect(frame, (left + 2, top + 12, 12, 2), OUTLINE)
    for spike_left in (3, 7, 11):
        draw_triangle_up(frame, left + spike_left, top + 4, 5, 9, TRAP_METAL)
        draw_triangle_up(frame, left + spike_left + 1, top + 7, 3, 4, TRAP_WARNING)


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


def draw_exit(frame: np.ndarray, tiles: tuple[tuple[int, int], tuple[int, int]], exit_type: str, color: Color) -> None:
    left = min(tile[0] for tile in tiles) * TILE_SIZE
    top = min(tile[1] for tile in tiles) * TILE_SIZE
    right = (max(tile[0] for tile in tiles) + 1) * TILE_SIZE
    bottom = (max(tile[1] for tile in tiles) + 1) * TILE_SIZE
    width = right - left
    height = bottom - top
    rect = (left + 2, top + 2, width - 4, height - 4)

    if exit_type == "normal":
        fill_rect(frame, rect, color)
        inset = 4
        fill_rect(frame, (left + inset, top + inset, width - inset * 2, height - inset * 2), EXIT_GLOW)
    elif exit_type == "locked_key":
        fill_rect(frame, rect, DOOR_WOOD)
        draw_rect_outline(frame, rect, OUTLINE)
        fill_rect(frame, (left + width // 2 - 3, top + height // 2 - 1, 6, 5), LOCK_COLOR)
        fill_rect(frame, (left + width // 2 - 2, top + height // 2 - 4, 4, 4), OUTLINE)
        fill_rect(frame, (left + width // 2 - 1, top + height // 2 - 3, 2, 3), color)
    else:
        fill_rect(frame, rect, color)
        draw_rect_outline(frame, rect, OUTLINE)
        fill_rect(frame, (left + width // 2 - 1, top + 4, 2, height - 8), CONDITIONAL_GLYPH)
        fill_rect(frame, (left + 5, top + height // 2 - 1, width - 10, 2), CONDITIONAL_GLYPH)


def draw_hud_text(frame: np.ndarray, line_1: str, line_2: str, *, y: int) -> None:
    draw_text(frame, line_1.upper(), 6, y + 5, TEXT_COLOR)
    draw_text(frame, line_2.upper(), 6, y + 18, TEXT_DIM)


def draw_text(frame: np.ndarray, text: str, x: int, y: int, color: Color) -> None:
    cursor_x = x
    for char in text:
        glyph = FONT_3X5.get(char, FONT_3X5[" "])
        for row_index, row in enumerate(glyph):
            for col_index, pixel in enumerate(row):
                if pixel == "1":
                    fill_rect(frame, (cursor_x + col_index, y + row_index, 1, 1), color)
        cursor_x += 4
        if cursor_x >= frame.shape[1] - 2:
            return


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
