from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.constants import (
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

        self.assertEqual(hud_sample, COLOR_HUD_BG)
        self.assertNotEqual(hud_sample, map_sample)

    def test_dynamic_entities_render_at_pixel_positions(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()
        env.player.position_px = (18.0, 16.0)

        frame = env.render()

        inside_player = tuple(frame[20, 20])
        left_of_player = tuple(frame[20, 17])

        self.assertEqual(inside_player, COLOR_PLAYER)
        self.assertNotEqual(left_of_player, COLOR_PLAYER)


if __name__ == "__main__":
    unittest.main()
