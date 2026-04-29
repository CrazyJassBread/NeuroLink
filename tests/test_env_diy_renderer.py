from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.constants import (
    COLOR_EXIT_CONDITIONAL,
    COLOR_EXIT_LOCKED,
    COLOR_EXIT_NORMAL,
    COLOR_HUD_ACCENT,
    COLOR_HUD_BG,
    COLOR_PLAYER,
    HUD_PIXEL_Y,
    INTERNAL_HEIGHT,
    INTERNAL_WIDTH,
    MAP_PIXEL_HEIGHT,
)
from env_diy.env import DungeonEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DUNGEON = (
    PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"
)


class RendererTests(unittest.TestCase):
    def test_render_canvas_size_is_160x160(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()

        frame = env.render()

        self.assertEqual(frame.shape, (INTERNAL_HEIGHT, INTERNAL_WIDTH, 3))

    def test_hud_area_exists_below_map_area(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()

        frame = env.render()
        hud_sample = tuple(frame[HUD_PIXEL_Y + 1, 1])
        map_sample = tuple(frame[MAP_PIXEL_HEIGHT - 2, 1])
        accent_sample = tuple(frame[HUD_PIXEL_Y + 15, 20])

        self.assertEqual(hud_sample, COLOR_HUD_BG)
        self.assertNotEqual(hud_sample, map_sample)
        self.assertEqual(accent_sample, COLOR_HUD_ACCENT)

    def test_dynamic_entities_render_at_pixel_positions(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()
        env.player.position_px = (18.0, 16.0)

        frame = env.render()

        inside_player = tuple(frame[20, 20])
        left_of_player = tuple(frame[20, 17])

        self.assertEqual(inside_player, COLOR_PLAYER)
        self.assertNotEqual(left_of_player, COLOR_PLAYER)

    def test_exit_types_render_with_distinct_colors(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()

        frame = env.render()
        normal_exit = tuple(frame[52, 4])
        locked_exit = tuple(frame[52, 148])
        conditional_exit = tuple(frame[116, 68])

        self.assertEqual(normal_exit, COLOR_EXIT_NORMAL)
        self.assertEqual(locked_exit, COLOR_EXIT_LOCKED)
        self.assertEqual(conditional_exit, COLOR_EXIT_CONDITIONAL)

    def test_hud_lines_show_room_hp_gold_and_items(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()
        env.player.gold = 12
        env.player.items = ["key", "bow"]

        line_1, line_2 = env.hud_lines()

        self.assertIn("R:", line_1)
        self.assertIn("HP:", line_1)
        self.assertIn("G:12", line_1)
        self.assertEqual(line_2, "I:key,bow")


if __name__ == "__main__":
    unittest.main()
