from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from rl.config import DEFAULT_TRAINING_CONFIG, TASK_ROOM_CONFIGS, TrainingConfig, apply_overrides, resolve_task_rooms
from rl.train import build_config


class RLTrainingConfigTests(unittest.TestCase):
    def test_default_config_selects_ppo_prototype(self) -> None:
        config = DEFAULT_TRAINING_CONFIG

        self.assertEqual(config.method, "ppo")
        self.assertEqual(config.task_rooms, ("prototype",))
        self.assertEqual(config.device, "auto")

    def test_gpu_device_mapping(self) -> None:
        self.assertEqual(TrainingConfig(gpu=-1).device, "cpu")
        self.assertEqual(TrainingConfig(gpu=0).device, "cuda:0")
        self.assertEqual(TrainingConfig(gpu=2).device, "cuda:2")

    def test_apply_overrides_ignores_none_values(self) -> None:
        config = apply_overrides(DEFAULT_TRAINING_CONFIG, method="dqn", gpu=None, episodes=3)

        self.assertEqual(config.method, "dqn")
        self.assertEqual(config.gpu, DEFAULT_TRAINING_CONFIG.gpu)
        self.assertEqual(config.episodes, 3)

    def test_resolve_named_task_rooms(self) -> None:
        targets = resolve_task_rooms(["avoid_traps", "combat_training"])

        self.assertEqual([target.name for target in targets], ["avoid_traps", "combat_training"])
        self.assertEqual(targets[0].config_path, TASK_ROOM_CONFIGS["avoid_traps"])
        self.assertEqual(targets[1].config_path, TASK_ROOM_CONFIGS["combat_training"])

    def test_config_path_overrides_task_rooms(self) -> None:
        custom_path = Path("env_diy/map_data/dungeons/key_door/room_001.json")
        targets = resolve_task_rooms(["prototype"], config_path=custom_path)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].name, "room_001")
        self.assertTrue(targets[0].config_path.is_absolute())

    def test_build_config_from_cli_namespace(self) -> None:
        args = argparse.Namespace(
            method="ppo",
            episodes=2,
            max_steps=20,
            total_timesteps=100,
            gpu=-1,
            seed=7,
            action_repeat=4,
            task_rooms=["avoid_traps"],
            config_path=None,
            output_dir=Path("tmp_outputs"),
            render=True,
            skip_train=False,
        )

        config = build_config(args)

        self.assertEqual(config.episodes, 2)
        self.assertEqual(config.max_steps, 20)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.task_rooms, ("avoid_traps",))


if __name__ == "__main__":
    unittest.main()
