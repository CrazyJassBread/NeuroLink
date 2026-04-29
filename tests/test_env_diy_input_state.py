from __future__ import annotations

import unittest

from env_diy.constants import ACTION_A, ACTION_LEFT, ACTION_NOOP, ACTION_RIGHT, ACTION_UP
from env_diy.input_state import HumanInputState
from env_diy.pygame_compat import pygame


class HumanInputStateTests(unittest.TestCase):
    def test_held_direction_repeats_until_keyup(self) -> None:
        state = HumanInputState()

        state.handle_keydown(pygame.K_RIGHT)

        self.assertEqual(state.resolve_action(), ACTION_RIGHT)
        self.assertEqual(state.resolve_action(), ACTION_RIGHT)

        state.handle_keyup(pygame.K_RIGHT)
        self.assertEqual(state.resolve_action(), ACTION_NOOP)

    def test_most_recent_direction_wins_and_releases_fallback(self) -> None:
        state = HumanInputState()

        state.handle_keydown(pygame.K_LEFT)
        state.handle_keydown(pygame.K_UP)

        self.assertEqual(state.resolve_action(), ACTION_UP)

        state.handle_keyup(pygame.K_UP)
        self.assertEqual(state.resolve_action(), ACTION_LEFT)

    def test_buttons_are_edge_triggered(self) -> None:
        state = HumanInputState()

        state.handle_keydown(pygame.K_z)
        self.assertEqual(state.resolve_action(), ACTION_A)
        self.assertEqual(state.resolve_action(), ACTION_NOOP)

        state.handle_keydown(pygame.K_z)
        self.assertEqual(state.resolve_action(), ACTION_NOOP)

        state.handle_keyup(pygame.K_z)
        state.handle_keydown(pygame.K_z)
        self.assertEqual(state.resolve_action(), ACTION_A)


if __name__ == "__main__":
    unittest.main()
