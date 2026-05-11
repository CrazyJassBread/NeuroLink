from __future__ import annotations

import math
import unittest
from pathlib import Path

from env_diy.env import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


class RandomRolloutSmokeTests(unittest.TestCase):
    def test_major_maps_random_rollout_does_not_crash(self) -> None:
        configs = [
            DUNGEON_ROOT / "prototype" / "dungeon.json",
            DUNGEON_ROOT / "combat_training" / "dungeon.json",
            DUNGEON_ROOT / "evasion_training" / "dungeon.json",
            DUNGEON_ROOT / "chest_training" / "dungeon.json",
            DUNGEON_ROOT / "avoid_traps" / "room_001.json",
            DUNGEON_ROOT / "kill_monsters" / "room_001.json",
            DUNGEON_ROOT / "key_door" / "room_001.json",
        ]
        for config in configs:
            with self.subTest(config=config.name):
                env = make_env(config, api="gym")
                try:
                    obs, info = env.reset(seed=0)
                    env.action_space.seed(0)
                    for _ in range(100):
                        action = int(env.action_space.sample())
                        obs, reward, terminated, truncated, info = env.step(action)
                        self.assertFalse(math.isnan(float(reward)))
                        self.assertIn("events", info)
                        self.assertIn("reward_terms", info)
                        if terminated or truncated:
                            break
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
