from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .constants import (
    ENTITY_SIZE_PX,
    ITEM_NAME_TO_ID,
    MAP_PIXEL_HEIGHT,
    MAP_PIXEL_WIDTH,
    PLAYER_GOLD_DEFAULT,
    PLAYER_HP_DEFAULT,
    PLAYER_KEYS_DEFAULT,
    PLAYER_SPEED_PX_PER_TICK,
    TILE_SIZE,
)


GridPos = tuple[int, int]
PixelPos = tuple[float, float]


class EquipmentSlot(str, Enum):
    A = "A"
    B = "B"


class ToolType(str, Enum):
    INTERACT = "interact"
    SHIELD = "shield"
    SWORD = "sword"
    NONE = "none"


@dataclass
class PlayerState:
    position_px: PixelPos
    size_px: int = ENTITY_SIZE_PX
    speed_px_per_step: float = PLAYER_SPEED_PX_PER_TICK
    health: int = PLAYER_HP_DEFAULT
    max_health: int = PLAYER_HP_DEFAULT
    gold: int = PLAYER_GOLD_DEFAULT
    keys: int = PLAYER_KEYS_DEFAULT
    items: list[str] = field(default_factory=lambda: ["sword", "shield"])
    tools: list[str] = field(default_factory=lambda: [ToolType.SWORD.value, ToolType.SHIELD.value])
    equipped: dict[str, str] = field(
        default_factory=lambda: {
            EquipmentSlot.A.value: ToolType.SWORD.value,
            EquipmentSlot.B.value: ToolType.SHIELD.value,
        }
    )
    action_a_label: str = ToolType.SWORD.value.upper()
    action_b_label: str = ToolType.SHIELD.value.upper()
    facing: str = "down"
    action_pose: str | None = None
    action_facing: str | None = None
    action_item: str | None = None
    action_ticks_remaining: int = 0

    def equipped_tool(self, slot: EquipmentSlot) -> str:
        return self.equipped.get(slot.value, ToolType.NONE.value)

    def equipped_tool_label(self, slot_name: str) -> str:
        return self.equipped.get(slot_name, ToolType.NONE.value)

    def equip_tool(self, slot: EquipmentSlot, tool: ToolType | str) -> None:
        tool_name = tool.value if isinstance(tool, ToolType) else str(tool)
        if tool_name not in self.tools:
            self.tools.append(tool_name)
        self.equipped[slot.value] = tool_name
        if slot == EquipmentSlot.A:
            self.action_a_label = tool_name.upper()
        elif slot == EquipmentSlot.B:
            self.action_b_label = tool_name.upper()

    def start_action(self, *, item_name: str, pose: str, facing: str, ticks: int) -> None:
        self.action_item = item_name
        self.action_pose = pose
        self.action_facing = facing
        self.action_ticks_remaining = max(0, int(ticks))

    def clear_action(self) -> None:
        self.action_pose = None
        self.action_facing = None
        self.action_item = None
        self.action_ticks_remaining = 0

    def has_action_pose(self) -> bool:
        return self.action_item is not None and self.action_ticks_remaining > 0


@dataclass
class ChestState:
    chest_id: str
    pos: GridPos
    loot: dict[str, Any]
    is_open: bool = False


@dataclass
class NPCState:
    npc_id: str
    pos: GridPos
    text: str


@dataclass
class TrapState:
    trap_id: str
    pos: GridPos
    damage: int = 1
    respawn_to: str = "default"
    single_use: bool = False
    is_active: bool = True


@dataclass
class ButtonState:
    button_id: str
    pos: GridPos
    message: str = "BUTTON"
    is_pressed: bool = False


def inventory_item_codes(items: list[str], size: int = 2) -> list[int]:
    codes = [ITEM_NAME_TO_ID.get(item, 0) for item in items[:size]]
    if len(codes) < size:
        codes.extend([0] * (size - len(codes)))
    return codes


def manhattan_distance(left: GridPos, right: GridPos) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def is_adjacent(left: GridPos, right: GridPos) -> bool:
    return manhattan_distance(left, right) <= 1


def entity_rect(position_px: PixelPos, size_px: int = ENTITY_SIZE_PX) -> tuple[float, float, float, float]:
    left = float(position_px[0])
    top = float(position_px[1])
    return left, top, left + float(size_px), top + float(size_px)


def entity_center_px(position_px: PixelPos, size_px: int = ENTITY_SIZE_PX) -> tuple[float, float]:
    return position_px[0] + size_px * 0.5, position_px[1] + size_px * 0.5


def tile_from_position_px(position_px: PixelPos, size_px: int = ENTITY_SIZE_PX) -> GridPos:
    center_x, center_y = entity_center_px(position_px, size_px)
    tile_x = int(center_x // TILE_SIZE)
    tile_y = int(center_y // TILE_SIZE)
    return tile_x, tile_y


def tile_to_top_left_px(tile_pos: GridPos) -> PixelPos:
    return float(tile_pos[0] * TILE_SIZE), float(tile_pos[1] * TILE_SIZE)


def tile_center_px(tile_pos: GridPos) -> tuple[float, float]:
    return tile_pos[0] * TILE_SIZE + TILE_SIZE * 0.5, tile_pos[1] * TILE_SIZE + TILE_SIZE * 0.5


def aabb_overlap(
    left_pos: PixelPos,
    left_size: int,
    right_pos: PixelPos,
    right_size: int,
) -> bool:
    left_l, left_t, left_r, left_b = entity_rect(left_pos, left_size)
    right_l, right_t, right_r, right_b = entity_rect(right_pos, right_size)
    return not (
        left_r <= right_l
        or left_l >= right_r
        or left_b <= right_t
        or left_t >= right_b
    )


def move_with_tile_collisions(
    position_px: PixelPos,
    size_px: int,
    velocity_px: tuple[float, float],
    blocking_tiles: set[GridPos],
) -> PixelPos:
    next_x = _move_axis(
        position_px=position_px,
        size_px=size_px,
        delta=velocity_px[0],
        blocking_tiles=blocking_tiles,
        axis="x",
    )
    next_y = _move_axis(
        position_px=(next_x, position_px[1]),
        size_px=size_px,
        delta=velocity_px[1],
        blocking_tiles=blocking_tiles,
        axis="y",
    )
    return next_x, next_y


def _move_axis(
    position_px: PixelPos,
    size_px: int,
    delta: float,
    blocking_tiles: set[GridPos],
    axis: str,
) -> float:
    if axis == "x":
        candidate = _clamp(position_px[0] + delta, 0.0, MAP_PIXEL_WIDTH - size_px)
        rect = (candidate, position_px[1], candidate + size_px, position_px[1] + size_px)
    else:
        candidate = _clamp(position_px[1] + delta, 0.0, MAP_PIXEL_HEIGHT - size_px)
        rect = (position_px[0], candidate, position_px[0] + size_px, candidate + size_px)

    collision_tiles = overlapping_tiles(rect)
    for tile in collision_tiles:
        if tile not in blocking_tiles:
            continue
        tile_left = float(tile[0] * TILE_SIZE)
        tile_top = float(tile[1] * TILE_SIZE)
        tile_right = tile_left + TILE_SIZE
        tile_bottom = tile_top + TILE_SIZE
        if axis == "x":
            if delta > 0:
                candidate = min(candidate, tile_left - size_px)
            elif delta < 0:
                candidate = max(candidate, tile_right)
            rect = (candidate, position_px[1], candidate + size_px, position_px[1] + size_px)
        else:
            if delta > 0:
                candidate = min(candidate, tile_top - size_px)
            elif delta < 0:
                candidate = max(candidate, tile_bottom)
            rect = (position_px[0], candidate, position_px[0] + size_px, candidate + size_px)

    upper = MAP_PIXEL_WIDTH - size_px if axis == "x" else MAP_PIXEL_HEIGHT - size_px
    return _clamp(candidate, 0.0, upper)


def overlapping_tiles(rect: tuple[float, float, float, float]) -> set[GridPos]:
    left, top, right, bottom = rect
    epsilon = 1e-6
    min_tile_x = max(0, int(math.floor(left / TILE_SIZE)))
    max_tile_x = min(int(math.floor((right - epsilon) / TILE_SIZE)), (MAP_PIXEL_WIDTH // TILE_SIZE) - 1)
    min_tile_y = max(0, int(math.floor(top / TILE_SIZE)))
    max_tile_y = min(int(math.floor((bottom - epsilon) / TILE_SIZE)), (MAP_PIXEL_HEIGHT // TILE_SIZE) - 1)

    tiles: set[GridPos] = set()
    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            tiles.add((tile_x, tile_y))
    return tiles


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
