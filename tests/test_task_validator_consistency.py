from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.core.constants import ACTION_A, ACTION_NOOP, ACTION_RIGHT, ACTION_UP
from env_diy.entities import tile_to_top_left_px
from env_diy.env import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


class TaskValidatorConsistencyTests(unittest.TestCase):
    def test_prototype_failure_path_matches_legacy(self) -> None:
        env = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym")
        try:
            env.reset(seed=0)
            env.room_coord = (0, 1)
            env.room = env.room_manager.get_room((0, 1))
            env.player.position_px = tile_to_top_left_px((1, 5))
            env.player.health = 1

            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)

            self.assertTrue(terminated)
            self.assertTrue(info["legacy_done"])
            self.assertTrue(info["validator_done"])
            self.assertTrue(info["validator_matches_legacy"])
            self.assertTrue(info["failure"])
            self.assertEqual(info["terminated_reason"], "agent_dead")
            self.assertIn("subgoal_status", info)
        finally:
            env.close()

    def test_avoid_traps_success_path_matches_legacy(self) -> None:
        env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym")
        try:
            env.reset(seed=0)
            env.player.position_px = (64.0, 0.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_UP)

            self.assertTrue(terminated)
            self.assertTrue(info["legacy_done"])
            self.assertTrue(info["validator_done"])
            self.assertTrue(info["validator_matches_legacy"])
            self.assertTrue(info["success"])
            self.assertFalse(info["failure"])
            self.assertEqual(info["task_progress"], 1.0)
            self.assertEqual(info["terminated_reason"], "reached_goal")
            self.assertIsInstance(info["subgoal_status"], dict)
        finally:
            env.close()

    def test_kill_monsters_success_path_matches_legacy(self) -> None:
        env = make_env(DUNGEON_ROOT / "kill_monsters" / "room_001.json", api="gym")
        try:
            env.reset(seed=0)
            monster_ids = list(env.room.monsters)
            first = env.room.monsters[monster_ids[0]]
            second = env.room.monsters[monster_ids[1]]
            env.room.monsters = {first.monster_id: first, second.monster_id: second}
            first.hp = 1
            first.position_px = env.player.position_px
            env.step(ACTION_A)
            second.hp = 1
            second.position_px = env.player.position_px

            obs, reward, terminated, truncated, info = env.step(ACTION_A)

            self.assertTrue(terminated)
            self.assertTrue(info["legacy_done"])
            self.assertTrue(info["validator_done"])
            self.assertTrue(info["validator_matches_legacy"])
            self.assertTrue(info["success"])
            self.assertEqual(info["task_progress"], 1.0)
            self.assertEqual(info["terminated_reason"], "reached_goal")
            self.assertIn("monsters_remaining", info["subgoal_status"])
        finally:
            env.close()

    def test_key_door_success_path_matches_legacy(self) -> None:
        env = make_env(DUNGEON_ROOT / "key_door" / "room_001.json", api="gym")
        try:
            env.reset(seed=0)
            env.player.position_px = tile_to_top_left_px((7, 2))
            env.step(ACTION_A)
            env.player.position_px = (64.0, 0.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_UP)

            self.assertTrue(terminated)
            self.assertTrue(info["legacy_done"])
            self.assertTrue(info["validator_done"])
            self.assertTrue(info["validator_matches_legacy"])
            self.assertTrue(info["success"])
            self.assertFalse(info["failure"])
            self.assertEqual(info["task_progress"], 1.0)
            self.assertEqual(info["terminated_reason"], "reached_goal")
            self.assertIn("door_unlocked", info["subgoal_status"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
