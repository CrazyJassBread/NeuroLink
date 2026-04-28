from __future__ import annotations

from .constants import ACTION_A, ACTION_B, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP
from .pygame_compat import pygame


KEY_TO_ACTION = {
    pygame.K_UP: ACTION_UP,
    pygame.K_DOWN: ACTION_DOWN,
    pygame.K_LEFT: ACTION_LEFT,
    pygame.K_RIGHT: ACTION_RIGHT,
    pygame.K_z: ACTION_A,
    pygame.K_x: ACTION_B,
}


def keydown_to_action(key: int) -> int | None:
    return KEY_TO_ACTION.get(key)
