from __future__ import annotations

import math

import pygame

from .constants import (
    CHEST_INTERACT_RADIUS,
    COLOR_CHEST,
    COLOR_CHEST_OPEN,
    COLOR_NPC,
    COLOR_PLAYER,
    FACING_DOT_THRESHOLD,
    IFRAMES_SECONDS,
    NPC_INTERACT_RADIUS,
    PLAYER_HP_DEFAULT,
    PLAYER_SIZE,
    PLAYER_SPEED,
    TILE_SIZE,
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


class Entity:
    def __init__(self, x: float, y: float, size: int = PLAYER_SIZE):
        self.x = float(x)
        self.y = float(y)
        self.size = int(size)

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(round(self.x)), int(round(self.y)), self.size, self.size)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.size * 0.5, self.y + self.size * 0.5

    def distance_to(self, other: "Entity") -> float:
        sx, sy = self.center
        ox, oy = other.center
        return math.hypot(sx - ox, sy - oy)


class Player(Entity):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, size=PLAYER_SIZE)
        self.hp = PLAYER_HP_DEFAULT
        self.speed = PLAYER_SPEED
        self.facing = pygame.Vector2(0.0, 1.0)
        self.invincibility_timer = 0.0

    def update_movement(
        self,
        direction: pygame.Vector2,
        dt: float,
        wall_rects: list[pygame.Rect],
    ) -> None:
        if direction.length_squared() > 0.0:
            direction = direction.normalize()
            self.facing = direction

        dx = direction.x * self.speed * dt
        dy = direction.y * self.speed * dt

        self._move_axis(dx, 0.0, wall_rects)
        self._move_axis(0.0, dy, wall_rects)

    def _move_axis(self, dx: float, dy: float, wall_rects: list[pygame.Rect]) -> None:
        self.x += dx
        self.y += dy

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

    def update_timers(self, dt: float) -> None:
        self.invincibility_timer = max(0.0, self.invincibility_timer - dt)

    def take_damage(self, amount: int = 1, iframes_seconds: float = IFRAMES_SECONDS) -> bool:
        if self.invincibility_timer > 0.0:
            return False

        self.hp = max(0, self.hp - amount)
        self.invincibility_timer = iframes_seconds
        return True

    def can_render(self) -> bool:
        if self.invincibility_timer <= 0.0:
            return True
        return int(self.invincibility_timer * 12.0) % 2 == 0

    def draw(self, surface: pygame.Surface) -> None:
        if self.can_render():
            pygame.draw.rect(surface, COLOR_PLAYER, self.rect)


class Interactable(Entity):
    """Interactables are anchored to 16x16 grid centers."""

    def __init__(self, grid_x: int, grid_y: int, size: int = TILE_SIZE):
        top_left_x = float(grid_x * TILE_SIZE)
        top_left_y = float(grid_y * TILE_SIZE)
        super().__init__(top_left_x, top_left_y, size=size)
        self.grid_x = int(grid_x)
        self.grid_y = int(grid_y)

    def _facing_player(self, player: Player) -> bool:
        target = pygame.Vector2(self.center)
        origin = pygame.Vector2(player.center)
        to_target = target - origin
        length = to_target.length()

        if length <= 1e-6:
            return True

        to_target.normalize_ip()
        facing = player.facing
        if facing.length_squared() <= 1e-6:
            facing = pygame.Vector2(0.0, 1.0)
        return facing.dot(to_target) >= FACING_DOT_THRESHOLD

    def _within_neighborhood(self, player: Player) -> bool:
        pcx = int(player.center[0] // TILE_SIZE)
        pcy = int(player.center[1] // TILE_SIZE)
        return abs(pcx - self.grid_x) <= 1 and abs(pcy - self.grid_y) <= 1

    def can_interact(self, player: Player, radius: float) -> bool:
        near_enough = self.distance_to(player) <= radius
        return near_enough and (self._facing_player(player) or self._within_neighborhood(player))


class Chest(Interactable):
    def __init__(self, grid_x: int, grid_y: int):
        super().__init__(grid_x, grid_y)
        self.is_open = False

    def try_open(self, player: Player) -> bool:
        if self.is_open:
            return False
        if not self.can_interact(player, CHEST_INTERACT_RADIUS):
            return False

        self.is_open = True
        print("Item Found!")
        return True

    def draw(self, surface: pygame.Surface) -> None:
        color = COLOR_CHEST_OPEN if self.is_open else COLOR_CHEST
        pygame.draw.rect(surface, color, self.rect)


class NPC(Interactable):
    def __init__(self, grid_x: int, grid_y: int, text: str):
        super().__init__(grid_x, grid_y)
        self.text = text

    def try_talk(self, player: Player) -> str | None:
        if not self.can_interact(player, NPC_INTERACT_RADIUS):
            return None
        return self.text

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, COLOR_NPC, self.rect)
