from __future__ import annotations

import unittest
from pathlib import Path

from env_diy.core.constants import ACTION_RIGHT, ACTION_UP
from env_diy.entities import tile_to_top_left_px
from env_diy.env import make_env
from env_diy.maps.rooms import RoomManager
from env_diy.tasks.task_spec import TaskSpec
from env_diy.tasks.validators import validate_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


class TaskCompletionTests(unittest.TestCase):
    def test_task_validator_matches_legacy_on_existing_single_task_maps(self) -> None:
        cases = [
            TASK_ROOT / "avoid_traps" / "room_001.json",
            TASK_ROOT / "kill_monsters" / "room_001.json",
            TASK_ROOT / "key_door" / "room_001.json",
        ]
        for config_path in cases:
            with self.subTest(config_path=config_path.name):
                env = make_env(config_path, api="gym")
                try:
                    obs, info = env.reset(seed=0)
                    self.assertFalse(info["legacy_done"])
                    self.assertFalse(info["validator_done"])

                    manager = RoomManager(config_path)
                    task_spec = TaskSpec.from_task_config(manager.task_config)
                    validation = validate_task(
                        task_spec,
                        env.engine.runtime,
                        info["events"],
                        info["event_details"],
                        legacy_done=False,
                    )
                    self.assertFalse(validation.success)
                    self.assertTrue(validation.validator_matches_legacy)
                finally:
                    env.close()

    def test_avoid_traps_completion_reports_consistent_validator_fields(self) -> None:
        env = make_env(TASK_ROOT / "avoid_traps" / "room_001.json", api="gym")
        try:
            env.reset(seed=0)
            env.player.position_px = (64.0, 0.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_UP)

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["success"])
            self.assertFalse(info["failure"])
            self.assertEqual(info["terminated_reason"], "reached_goal")
            self.assertTrue(info["legacy_done"])
            self.assertTrue(info["validator_done"])
            self.assertTrue(info["validator_matches_legacy"])
        finally:
            env.close()

    def test_agent_death_reports_failure_reason(self) -> None:
        env = make_env(TASK_ROOT / "avoid_traps" / "room_001.json", api="gym")
        try:
            env.reset(seed=0)
            env.player.health = 1
            env.player.position_px = tile_to_top_left_px((4, 2))

            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

            self.assertTrue(terminated)
            self.assertTrue(info["failure"])
            self.assertEqual(info["terminated_reason"], "agent_dead")
            self.assertTrue(info["legacy_done"])
            self.assertTrue(info["validator_done"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
