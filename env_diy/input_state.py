from __future__ import annotations

from collections.abc import Sequence

import pygame


class InputState:
    """Tracks continuous directional input and edge-triggered action buttons."""

    def __init__(self) -> None:
        self._pressed: Sequence[bool] | None = None
        self._prev_a = False
        self._prev_b = False
        self._curr_a = False
        self._curr_b = False

    def update(self, pressed: Sequence[bool]) -> None:
        self._pressed = pressed
        self._prev_a = self._curr_a
        self._prev_b = self._curr_b

        self._curr_a = bool(pressed[pygame.K_z])
        self._curr_b = bool(pressed[pygame.K_x])

    def movement_vector(self) -> pygame.Vector2:
        if self._pressed is None:
            return pygame.Vector2(0.0, 0.0)

        dx = 0.0
        dy = 0.0

        if self._pressed[pygame.K_LEFT]:
            dx -= 1.0
        if self._pressed[pygame.K_RIGHT]:
            dx += 1.0
        if self._pressed[pygame.K_UP]:
            dy -= 1.0
        if self._pressed[pygame.K_DOWN]:
            dy += 1.0

        return pygame.Vector2(dx, dy)

    def action_a_pressed(self) -> bool:
        return self._curr_a and not self._prev_a

    def action_b_pressed(self) -> bool:
        return self._curr_b and not self._prev_b
