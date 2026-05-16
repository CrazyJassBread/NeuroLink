from __future__ import annotations

from dataclasses import dataclass, field

import pygame

from ..constants import ACTION_A, ACTION_B, ACTION_DOWN, ACTION_LEFT, ACTION_NOOP, ACTION_RIGHT, ACTION_UP


DIRECTION_KEY_TO_ACTION = {
    pygame.K_UP: ACTION_UP,
    pygame.K_DOWN: ACTION_DOWN,
    pygame.K_LEFT: ACTION_LEFT,
    pygame.K_RIGHT: ACTION_RIGHT,
}

BUTTON_KEY_TO_ACTION = {
    pygame.K_z: ACTION_A,
    pygame.K_x: ACTION_B,
}


def keydown_to_action(key: int) -> int | None:
    return DIRECTION_KEY_TO_ACTION.get(key) or BUTTON_KEY_TO_ACTION.get(key)


@dataclass
class HumanInputState:
    held_directions: list[int] = field(default_factory=list)
    held_buttons: set[int] = field(default_factory=set)
    queued_actions: list[int] = field(default_factory=list)

    def handle_keydown(self, key: int) -> None:
        direction_action = DIRECTION_KEY_TO_ACTION.get(key)
        if direction_action is not None:
            if key in self.held_directions:
                self.held_directions.remove(key)
            self.held_directions.append(key)
            return

        button_action = BUTTON_KEY_TO_ACTION.get(key)
        if button_action is None:
            return
        if key in self.held_buttons:
            return
        self.held_buttons.add(key)
        if button_action == ACTION_B:
            return
        self.queued_actions.append(button_action)

    def handle_keyup(self, key: int) -> None:
        if key in self.held_directions:
            self.held_directions.remove(key)
        self.held_buttons.discard(key)

    def resolve_action(self) -> int:
        if any(BUTTON_KEY_TO_ACTION.get(key) == ACTION_B for key in self.held_buttons):
            return ACTION_B
        if self.queued_actions:
            return self.queued_actions.pop(0)
        if self.held_directions:
            return DIRECTION_KEY_TO_ACTION[self.held_directions[-1]]
        return ACTION_NOOP
