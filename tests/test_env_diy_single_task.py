from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from env_diy.core.constants import ACTION_A, ACTION_RIGHT, ACTION_UP, GRID_HEIGHT, GRID_WIDTH
from env_diy.entities import tile_to_top_left_px
from env_diy.envs import DungeonEnv
from env_diy.maps import MapValidationError, RoomManager
from rl.train_single_task import resolve_single_task_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


class SingleTaskDungeonTests(unittest.TestCase):
    def test_single_task_room_configs_load_with_task_metadata(self) -> None:
        cases = {
            "avoid_traps": "avoid_traps_001",
            "kill_monsters": "kill_monsters_001",
            "key_door": "key_door_001",
        }
        for task_type, task_id in cases.items():
            with self.subTest(task_type=task_type):
                config_path = TASK_ROOT / task_type / "room_001.json"
                manager = RoomManager(config_path)
                room = manager.get_room((0, 0))

                self.assertIsNotNone(manager.task_config)
                assert manager.task_config is not None
                self.assertEqual(manager.task_config.task_id, task_id)
                self.assertEqual(manager.task_config.task_type, task_type)
                self.assertEqual(room.width, GRID_WIDTH)
                self.assertEqual(room.height, GRID_HEIGHT)
                self.assertEqual(manager.start_room_id, "room_001")

    def test_invalid_objective_type_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_room.json"
            payload = self._base_single_task_payload()
            payload["objective"] = {"type": "collect_all_orbs"}
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(MapValidationError) as ctx:
                RoomManager(path)

        self.assertIn("objective.type", str(ctx.exception))
        self.assertIn("unsupported objective type", str(ctx.exception))

    def test_missing_reward_config_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "default_reward_room.json"
            payload = self._base_single_task_payload()
            payload.pop("reward")
            path.write_text(json.dumps(payload), encoding="utf-8")

            manager = RoomManager(path)

        assert manager.task_config is not None
        self.assertEqual(manager.task_config.reward.finish, 10.0)
        self.assertEqual(manager.task_config.reward.step, -0.01)
        self.assertEqual(manager.task_config.reward.damage, -1.0)

    def test_avoid_traps_finishes_on_target_exit_once(self) -> None:
        env = DungeonEnv(TASK_ROOT / "avoid_traps" / "room_001.json")
        try:
            env.reset(seed=0)
            env.player.position_px = (64.0, 0.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_UP)

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["finish"])
            self.assertTrue(info["task_success"])
            self.assertEqual(info["task_id"], "avoid_traps_001")
            self.assertEqual(info["task_type"], "avoid_traps")
            self.assertIn("task_finished", info["events"])
            finish_detail = next(detail for detail in info["event_details"] if detail["type"] == "task_finished")
            self.assertEqual(finish_detail["task_id"], "avoid_traps_001")
            self.assertEqual(finish_detail["reward"], 10.0)
            self.assertAlmostEqual(reward, 10.08)

            obs, reward, terminated, truncated, info = env.step(ACTION_UP)
            self.assertFalse(info.get("finish", False))
            self.assertTrue(info["auto_reset"])
        finally:
            env.close()

    def test_avoid_traps_trap_damage_uses_task_penalty(self) -> None:
        env = DungeonEnv(TASK_ROOT / "avoid_traps" / "room_001.json")
        try:
            env.reset(seed=0)
            env.player.position_px = tile_to_top_left_px((4, 2))

            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

            self.assertIn("trap_damage", info["events"])
            self.assertLessEqual(reward, -1.0)
            self.assertFalse(info.get("finish", False))
        finally:
            env.close()

    def test_kill_monsters_finishes_when_all_targets_are_defeated(self) -> None:
        env = DungeonEnv(TASK_ROOT / "kill_monsters" / "room_001.json")
        try:
            env.reset(seed=0)
            monster_ids = list(env.room.monsters)
            self.assertGreaterEqual(len(monster_ids), 2)
            first = env.room.monsters[monster_ids[0]]
            second = env.room.monsters[monster_ids[1]]

            env.room.monsters = {first.monster_id: first, second.monster_id: second}
            first.hp = 1
            first.position_px = env.player.position_px
            obs, reward, terminated, truncated, info = env.step(ACTION_A)
            self.assertIn("monster_killed", info["events"])
            self.assertFalse(info.get("finish", False))
            self.assertFalse(terminated)

            second.hp = 1
            second.position_px = env.player.position_px
            obs, reward, terminated, truncated, info = env.step(ACTION_A)

            self.assertTrue(terminated)
            self.assertTrue(info["finish"])
            self.assertIn("task_finished", info["events"])
            self.assertEqual(info["task_type"], "kill_monsters")
        finally:
            env.close()

    def test_key_door_requires_key_then_unlocks_and_finishes(self) -> None:
        env = DungeonEnv(TASK_ROOT / "key_door" / "room_001.json")
        try:
            env.reset(seed=0)
            env.player.position_px = (64.0, 0.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_UP)
            self.assertIn("blocked_locked", info["events"])
            self.assertFalse(terminated)
            self.assertFalse(info.get("finish", False))

            chest = env.room.chests["chest_key"]
            env.player.position_px = tile_to_top_left_px((7, 2))
            obs, reward, terminated, truncated, info = env.step(ACTION_A)
            self.assertIn("got_key", info["events"])
            self.assertEqual(env.player.keys, 1)

            env.player.position_px = (64.0, 0.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_UP)

            self.assertTrue(terminated)
            self.assertTrue(info["finish"])
            self.assertIn("used_key", info["events"])
            self.assertIn("door_unlocked", info["events"])
            self.assertIn("task_finished", info["events"])
            self.assertEqual(info["task_type"], "key_door")
            self.assertEqual(env.player.keys, 0)
            self.assertTrue(chest.is_open)
        finally:
            env.close()

    def test_train_single_task_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "single_task.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "rl/train_single_task.py",
                    "--task",
                    "avoid_traps",
                    "--episodes",
                    "1",
                    "--max-steps",
                    "20",
                    "--seed",
                    "0",
                    "--action-repeat",
                    "4",
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("episode=0", completed.stdout)
        self.assertIn("finish_rate=", completed.stdout)
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertIn("finish", payload)
        self.assertIn("task_type", payload)

    def test_task_room_resolver_defaults_to_room_001(self) -> None:
        config_path = resolve_single_task_config(task="avoid_traps", room="room_001", config=None)

        self.assertEqual(config_path, TASK_ROOT / "avoid_traps" / "room_001.json")

    @staticmethod
    def _base_single_task_payload() -> dict[str, Any]:
        return {
            "task_id": "avoid_traps_test",
            "task_type": "avoid_traps",
            "room_id": "room_001",
            "objective": {"type": "reach_exit", "target_exit": "north_exit"},
            "reward": {"finish": 10.0},
            "layout": ["." * GRID_WIDTH for _ in range(GRID_HEIGHT)],
            "spawns": {"default": [4, 6], "from_south": [4, 6]},
            "default_spawn": "default",
            "objects": [],
            "exits": [
                {
                    "id": "north_exit",
                    "direction": "north",
                    "target_room": "room_001",
                    "target_entry": "from_south",
                    "type": "normal",
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
