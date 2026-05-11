from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.core.constants import ACTION_NOOP
from env_diy.env import make_env
from env_diy.envs import DungeonEnv as LegacyDungeonEnv
from env_diy.wrappers import GymDungeonEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DUNGEON = (
    PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"
)


class WrapperCompatibilityTests(unittest.TestCase):
    def test_canonical_wrapper_supports_reset_and_step(self) -> None:
        env = make_env(STRUCTURED_DUNGEON, api="gym")
        try:
            self.assertIsInstance(env, GymDungeonEnv)
            obs, info = env.reset(seed=0)
            step = env.step(ACTION_NOOP)
            self.assertEqual(len(step), 5)
        finally:
            env.close()

    def test_legacy_wrapper_exposes_legacy_properties(self) -> None:
        env = LegacyDungeonEnv(STRUCTURED_DUNGEON)
        try:
            env.reset(seed=0)
            _ = env.room
            _ = env.player
            _ = env.room_manager
            _ = env.last_message
            _ = env.step_count
            _ = env.episode
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
