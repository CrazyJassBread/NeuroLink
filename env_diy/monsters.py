from __future__ import annotations

from dataclasses import dataclass

import pygame

from .constants import (
    COLOR_AMBUSHER,
    COLOR_CHASER,
    COLOR_PATROLLER,
    INTERNAL_HEIGHT,
    INTERNAL_WIDTH,
    MONSTER_BASE_SPEED,
    MONSTER_SIZE,
    TILE_SIZE,
)
from .entities import Entity, Player


Bounds = tuple[float, float, float, float]


@dataclass
class MonsterSpawn:
    kind: str
    grid_x: int
    grid_y: int
    ambush_range: int = 2
    patrol_span: int = 32


class Monster(Entity):
    def __init__(self, x: float, y: float, speed: float, color: tuple[int, int, int]):
        super().__init__(x, y, size=MONSTER_SIZE)
        self.speed = speed
        self.color = color

    def update(self, player: Player, dt: float, bounds: Bounds, wall_rects: list[pygame.Rect]) -> None:
        raise NotImplementedError

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.color, self.rect)

    def _move(self, vx: float, vy: float, dt: float, bounds: Bounds, wall_rects: list[pygame.Rect]) -> None:
        self._move_axis(vx * dt, 0.0, bounds, wall_rects)
        self._move_axis(0.0, vy * dt, bounds, wall_rects)

    def _move_axis(
        self,
        dx: float,
        dy: float,
        bounds: Bounds,
        wall_rects: list[pygame.Rect],
    ) -> None:
        min_x, max_x, min_y, max_y = bounds
        self.x += dx
        self.y += dy

        self.x = max(min_x, min(self.x, max_x))
        self.y = max(min_y, min(self.y, max_y))

        entity_rect = self.rect
        for wall in wall_rects:
            if not entity_rect.colliderect(wall):
                continue
            if dx > 0.0:
                self.x = float(wall.left - self.size)
            elif dx < 0.0:
                self.x = float(wall.right)
            elif dy > 0.0:
                self.y = float(wall.top - self.size)
            elif dy < 0.0:
                self.y = float(wall.bottom)
            entity_rect = self.rect

    def _move_towards(
        self,
        target: tuple[float, float],
        speed: float,
        dt: float,
        bounds: Bounds,
        wall_rects: list[pygame.Rect],
    ) -> None:
        target_vec = pygame.Vector2(target)
        own_vec = pygame.Vector2(self.center)
        delta = target_vec - own_vec

        if delta.length_squared() <= 1e-6:
            return

        delta.normalize_ip()
        self._move(delta.x * speed, delta.y * speed, dt, bounds, wall_rects)

    def cell(self) -> tuple[int, int]:
        return int(self.center[0] // TILE_SIZE), int(self.center[1] // TILE_SIZE)


class Chaser(Monster):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, speed=MONSTER_BASE_SPEED, color=COLOR_CHASER)

    def update(self, player: Player, dt: float, bounds: Bounds, wall_rects: list[pygame.Rect]) -> None:
        self._move_towards(player.center, self.speed, dt, bounds, wall_rects)


class Ambusher(Monster):
    def __init__(self, x: float, y: float, ambush_range: int = 2):
        super().__init__(x, y, speed=MONSTER_BASE_SPEED, color=COLOR_AMBUSHER)
        self.ambush_range = ambush_range
        self.activated = False

    def update(self, player: Player, dt: float, bounds: Bounds, wall_rects: list[pygame.Rect]) -> None:
        my_cx, my_cy = self.cell()
        pl_cx = int(player.center[0] // TILE_SIZE)
        pl_cy = int(player.center[1] // TILE_SIZE)

        if abs(my_cx - pl_cx) <= self.ambush_range and abs(my_cy - pl_cy) <= self.ambush_range:
            self.activated = True

        if self.activated:
            self._move_towards(player.center, self.speed * 2.0, dt, bounds, wall_rects)


class Patroller(Monster):
    def __init__(self, x: float, y: float, patrol_span: int = 32):
        super().__init__(x, y, speed=MONSTER_BASE_SPEED * 0.9, color=COLOR_PATROLLER)
        self.spawn_x = x
        self.spawn_y = y
        self.patrol_span = float(max(TILE_SIZE, patrol_span))
        self.waypoint_index = 0
        self.waypoints = self._build_waypoints()

    def _build_waypoints(self) -> list[tuple[float, float]]:
        max_x = float(INTERNAL_WIDTH - MONSTER_SIZE)
        max_y = float(INTERNAL_HEIGHT - MONSTER_SIZE)
        x0 = max(0.0, min(self.spawn_x, max_x))
        y0 = max(0.0, min(self.spawn_y, max_y))
        x1 = max(0.0, min(x0 + self.patrol_span, max_x))
        y1 = max(0.0, min(y0 + self.patrol_span, max_y))
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def update(self, player: Player, dt: float, bounds: Bounds, wall_rects: list[pygame.Rect]) -> None:
        target = pygame.Vector2(self.waypoints[self.waypoint_index])
        own_pos = pygame.Vector2(self.x, self.y)
        delta = target - own_pos

        if delta.length() <= 1.0:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            return

        delta.normalize_ip()
        self._move(delta.x * self.speed, delta.y * self.speed, dt, bounds, wall_rects)


def build_monster_from_dict(data: dict) -> Monster:
    kind = str(data.get("type", "chaser")).lower()
    grid = data.get("grid", [0, 0])
    grid_x = int(grid[0])
    grid_y = int(grid[1])

    x = float(grid_x * TILE_SIZE)
    y = float(grid_y * TILE_SIZE)

    if kind == "ambusher":
        ambush_range = int(data.get("ambush_range", 2))
        return Ambusher(x, y, ambush_range=ambush_range)

    if kind == "patroller":
        patrol_span = int(data.get("patrol_span", 32))
        return Patroller(x, y, patrol_span=patrol_span)

    return Chaser(x, y)
