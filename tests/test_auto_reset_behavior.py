from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.core.constants import ACTION_NOOP, ACTION_RIGHT
from env_diy.entities import tile_to_top_left_px
from env_diy.env import DungeonEnv as LegacyDungeonEnv, make_env
from env_diy.wrappers import GymDungeonEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


class AutoResetBehaviorTests(unittest.TestCase):
    def test_canonical_gym_wrapper_defaults_to_no_autoreset(self) -> None:
        env = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym")
        try:
            self.assertIsInstance(env, GymDungeonEnv)
            self.assertFalse(env.auto_reset_on_step)
        finally:
            env.close()

    def test_canonical_wrapper_requires_explicit_reset_after_termination(self) -> None:
        env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym")
        try:
            env.reset(seed=0)
            env.player.health = 1
            env.player.position_px = tile_to_top_left_px((4, 2))
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertTrue(terminated)
            with self.assertRaisesRegex(RuntimeError, "Call reset\\(\\) before step"):
                env.step(ACTION_NOOP)
        finally:
            env.close()

    def test_legacy_wrapper_keeps_autoreset_behavior(self) -> None:
        env = LegacyDungeonEnv(DUNGEON_ROOT / "avoid_traps" / "room_001.json")
        try:
            self.assertTrue(env.auto_reset_on_step)
            env.reset(seed=0)
            env.player.health = 1
            env.player.position_px = tile_to_top_left_px((4, 2))
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertTrue(terminated)
            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)
            self.assertFalse(terminated)
            self.assertTrue(info["auto_reset"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
