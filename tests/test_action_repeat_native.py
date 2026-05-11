from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.core.constants import ACTION_RIGHT, ACTION_UP
from env_diy.env import make_env
from rl.utils.wrappers import ActionRepeatWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


def _sum_terms(term_dicts: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for terms in term_dicts:
        for key, value in terms.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


class NativeActionRepeatTests(unittest.TestCase):
    def test_action_repeat_one_matches_single_step_behavior(self) -> None:
        env = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym", action_repeat=1)
        try:
            env.reset(seed=0)
            before = env.player.position_px
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertEqual(info["action_repeat"], 1)
            self.assertEqual(info["inner_steps"], 1)
            self.assertEqual(env.player.position_px[0] - before[0], 1.0)
        finally:
            env.close()

    def test_action_repeat_four_matches_four_single_steps_without_termination(self) -> None:
        single = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym", action_repeat=1)
        repeated = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym", action_repeat=4)
        try:
            single.reset(seed=0)
            repeated.reset(seed=0)

            rewards = []
            term_dicts = []
            event_counts: dict[str, int] = {}
            for _ in range(4):
                obs, reward, terminated, truncated, info = single.step(ACTION_RIGHT)
                rewards.append(float(reward))
                term_dicts.append(info["reward_terms"])
                for key, value in info["event_counts"].items():
                    event_counts[key] = event_counts.get(key, 0) + int(value)
                self.assertFalse(terminated)
                self.assertFalse(truncated)

            obs_r, reward_r, terminated_r, truncated_r, info_r = repeated.step(ACTION_RIGHT)

            self.assertFalse(terminated_r)
            self.assertFalse(truncated_r)
            self.assertEqual(info_r["inner_steps"], 4)
            self.assertEqual(repeated.player.position_px, single.player.position_px)
            self.assertAlmostEqual(reward_r, sum(rewards))
            self.assertEqual(info_r["reward_terms"], _sum_terms(term_dicts))
            self.assertEqual(info_r["event_counts"], event_counts)
            self.assertEqual(info_r["event_flags"]["move_right"], True)
        finally:
            single.close()
            repeated.close()

    def test_action_repeat_stops_early_on_termination(self) -> None:
        env = make_env(DUNGEON_ROOT / "avoid_traps" / "room_001.json", api="gym", action_repeat=4)
        try:
            env.reset(seed=0)
            env.player.position_px = (64.0, 0.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_UP)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["inner_steps"], 1)
        finally:
            env.close()

    def test_native_action_repeat_conflicts_with_rl_wrapper(self) -> None:
        env = make_env(DUNGEON_ROOT / "prototype" / "dungeon.json", api="gym", action_repeat=4)
        try:
            with self.assertRaisesRegex(ValueError, "cannot combine env native action_repeat"):
                ActionRepeatWrapper(env, repeat=4)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
