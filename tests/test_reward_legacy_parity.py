from __future__ import annotations

import math
import unittest
from pathlib import Path

from env_diy.core.constants import ACTION_A, ACTION_NOOP, ACTION_RIGHT, ACTION_UP
from env_diy.entities import tile_to_top_left_px
from env_diy.env import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


class RewardLegacyParityTests(unittest.TestCase):
    def test_movement_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertEqual(info["reward_terms"], {"movement": -0.01})
            self.assertAlmostEqual(reward, -0.01)
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        finally:
            env.close()

    def test_empty_action_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.player.position_px = tile_to_top_left_px((1, 1))
            obs, reward, terminated, truncated, info = env.step(ACTION_A)
            self.assertIn("empty_action", info["reward_terms"])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        finally:
            env.close()

    def test_blocked_exit_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "key_door" / "room_001.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.player.position_px = (64.0, 0.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_UP)
            self.assertIn("blocked_exit", info["reward_terms"])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        finally:
            env.close()

    def test_picked_key_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "key_door" / "room_001.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.room.monsters = {}
            env.player.position_px = tile_to_top_left_px((7, 2))
            obs, reward, terminated, truncated, info = env.step(ACTION_A)
            self.assertIn("got_key", info["events"])
            self.assertEqual(info["reward_terms"], {"got_key": 0.5})
            self.assertAlmostEqual(reward, 0.5)
        finally:
            env.close()

    def test_opened_door_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "key_door" / "room_001.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.player.keys = 1
            env.player.position_px = (64.0, 0.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_UP)
            self.assertIn("door_unlocked", info["events"])
            self.assertIn("door_unlock", info["reward_terms"])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        finally:
            env.close()

    def test_picked_coin_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.player.position_px = tile_to_top_left_px((4, 1))
            obs, reward, terminated, truncated, info = env.step(ACTION_A)
            self.assertIn("got_gold", info["events"])
            self.assertEqual(info["reward_terms"], {"got_gold": 0.2})
            self.assertAlmostEqual(reward, 0.2)
        finally:
            env.close()

    def test_killed_monster_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "kill_monsters" / "room_001.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            monster = next(iter(env.room.monsters.values()))
            monster.hp = 1
            monster.position_px = env.player.position_px
            obs, reward, terminated, truncated, info = env.step(ACTION_A)
            self.assertIn("monster_killed", info["events"])
            self.assertIn("monster_kill", info["reward_terms"])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        finally:
            env.close()

    def test_hit_trap_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.player.position_px = tile_to_top_left_px((4, 2))
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertIn("trap_damage", info["events"])
            self.assertIn("trap_damage", info["reward_terms"])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        finally:
            env.close()

    def test_agent_dead_reward_terms_sum_matches_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.player.health = 1
            env.player.position_px = tile_to_top_left_px((4, 2))
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertTrue(terminated)
            self.assertIn("game_over", info["events"])
            self.assertTrue(math.isclose(reward, sum(info["reward_terms"].values())))
        finally:
            env.close()

    def test_reached_goal_and_task_finished_reward_terms_sum_match_reward(self) -> None:
        env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym", reward_mode="legacy")
        try:
            env.reset(seed=0)
            env.player.position_px = (64.0, 0.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_UP)
            self.assertTrue(terminated)
            self.assertIn("task_finished", info["events"])
            self.assertIn("task_finished", info["reward_terms"])
            self.assertIn("victory", info["events"])
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
