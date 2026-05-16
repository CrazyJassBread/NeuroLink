from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import (
    GRID_HEIGHT,
    GRID_WIDTH,
    MONSTER_DEFAULT_HP,
    MONSTER_SIZE_PX,
    MONSTER_SPEED_PX_PER_TICK,
)
from .state import (
    GridPos,
    PixelPos,
    move_with_tile_collisions,
    tile_from_position_px,
    tile_to_top_left_px,
)


@dataclass
class MonsterState:
    monster_id: str
    monster_type: str
    position_px: PixelPos
    size_px: int = MONSTER_SIZE_PX
    speed_px_per_step: float = MONSTER_SPEED_PX_PER_TICK
    hp: int = MONSTER_DEFAULT_HP
    max_hp: int = MONSTER_DEFAULT_HP
    damage: int = 1
    ambush_range_tiles: int = 2
    patrol_span_tiles: int = 1
    activated: bool = False
    patrol_points_px: list[PixelPos] = field(default_factory=list)
    patrol_index: int = 0
    stun_ticks_remaining: int = 0
    last_move_delta_px: PixelPos = (0.0, 0.0)

    @property
    def tile_pos(self) -> GridPos:
        return tile_from_position_px(self.position_px, self.size_px)

    def current_patrol_target(self) -> PixelPos | None:
        if not self.patrol_points_px:
            return None
        return self.patrol_points_px[self.patrol_index]


def build_monster_from_dict(data: dict) -> MonsterState:
    monster_type = str(data.get("monster_type", data.get("type", "chaser"))).lower()
    grid = data.get("grid", [0, 0])
    spawn_tile = (int(grid[0]), int(grid[1]))
    position_px = tile_to_top_left_px(spawn_tile)

    patrol_span_tiles = max(1, int(data.get("patrol_span", 16)) // 16)
    hp_val = max(1, int(data.get("hp", MONSTER_DEFAULT_HP)))
    monster = MonsterState(
        monster_id=str(data.get("id", "")),
        monster_type=monster_type,
        position_px=position_px,
        size_px=max(1, int(data.get("size_px", MONSTER_SIZE_PX))),
        speed_px_per_step=float(data.get("speed_px_per_step", MONSTER_SPEED_PX_PER_TICK)),
        hp=hp_val,
        max_hp=hp_val,
        damage=max(1, int(data.get("damage", 1))),
        ambush_range_tiles=max(1, int(data.get("ambush_range", 2))),
        patrol_span_tiles=patrol_span_tiles,
    )

    if monster_type == "patroller":
        monster.patrol_points_px = _build_patrol_points_px(spawn_tile, patrol_span_tiles)
    return monster


def update_monster(
    monster: MonsterState,
    player_position_px: PixelPos,
    wall_tiles: set[GridPos],
    blocking_tiles: set[GridPos],
) -> None:
    if monster.stun_ticks_remaining > 0:
        monster.last_move_delta_px = (0.0, 0.0)
        return

    if monster.monster_type == "ambusher":
        if _within_range(monster.tile_pos, tile_from_position_px(player_position_px), monster.ambush_range_tiles):
            monster.activated = True
        if monster.activated:
            _move_towards(monster, player_position_px, wall_tiles, blocking_tiles)
        return

    if monster.monster_type == "patroller":
        _advance_patroller(monster, wall_tiles, blocking_tiles)
        return

    _move_towards(monster, player_position_px, wall_tiles, blocking_tiles)


def _build_patrol_points_px(origin_tile: GridPos, patrol_span_tiles: int) -> list[PixelPos]:
    x0, y0 = origin_tile
    x1 = min(GRID_WIDTH - 1, x0 + patrol_span_tiles)
    y1 = min(GRID_HEIGHT - 1, y0 + patrol_span_tiles)
    return [
        tile_to_top_left_px((x0, y0)),
        tile_to_top_left_px((x1, y0)),
        tile_to_top_left_px((x1, y1)),
        tile_to_top_left_px((x0, y1)),
    ]


def _within_range(left: GridPos, right: GridPos, radius: int) -> bool:
    return abs(left[0] - right[0]) <= radius and abs(left[1] - right[1]) <= radius


def _advance_patroller(
    monster: MonsterState,
    wall_tiles: set[GridPos],
    blocking_tiles: set[GridPos],
) -> None:
    target = monster.current_patrol_target()
    if target is None:
        return

    if _distance(monster.position_px, target) <= monster.speed_px_per_step:
        monster.position_px = target
        monster.patrol_index = (monster.patrol_index + 1) % len(monster.patrol_points_px)
        target = monster.current_patrol_target()
        if target is None:
            return

    _move_towards(monster, target, wall_tiles, blocking_tiles)


def _move_towards(
    monster: MonsterState,
    target_position_px: PixelPos,
    wall_tiles: set[GridPos],
    blocking_tiles: set[GridPos],
) -> None:
    target_x, target_y = target_position_px
    dx = target_x - monster.position_px[0]
    dy = target_y - monster.position_px[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        monster.last_move_delta_px = (0.0, 0.0)
        return

    step_x = (dx / distance) * monster.speed_px_per_step
    step_y = (dy / distance) * monster.speed_px_per_step
    world_blockers = set(wall_tiles) | set(blocking_tiles)
    world_blockers.discard(monster.tile_pos)
    previous_position = monster.position_px
    monster.position_px = move_with_tile_collisions(
        monster.position_px,
        monster.size_px,
        (step_x, step_y),
        world_blockers,
    )
    monster.last_move_delta_px = (
        monster.position_px[0] - previous_position[0],
        monster.position_px[1] - previous_position[1],
    )


def _distance(left: PixelPos, right: PixelPos) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])
