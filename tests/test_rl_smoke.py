from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from env_diy.core.constants import INTERNAL_HEIGHT, INTERNAL_WIDTH
from env_diy.envs import DungeonEnv
from rl.train_random import run_random_training
from rl.utils import make_env, observation_is_valid


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RLSmokeTests(unittest.TestCase):
    def test_train_random_imports_without_running_training(self) -> None:
        import rl.train_random as train_random

        self.assertTrue(callable(train_random.run_random_training))

    def test_env_factory_creates_resettable_env(self) -> None:
        env = make_env(seed=0)
        try:
            obs, info = env.reset(seed=0)

            self.assertIsInstance(env, DungeonEnv)
            self.assertIsInstance(info, dict)
            self.assertTrue(observation_is_valid(env, obs))
        finally:
            env.close()

    def test_sampled_actions_step_and_reset_across_early_termination(self) -> None:
        env = make_env(seed=0)
        try:
            obs, info = env.reset(seed=0)
            self.assertTrue(env.observation_space.contains(obs))

            steps_seen = 0
            resets_seen = 0
            while steps_seen < 10:
                action = env.action_space.sample()
                step_result = env.step(action)
                self.assertEqual(len(step_result), 5)
                obs, reward, terminated, truncated, info = step_result

                self.assertTrue(env.action_space.contains(action))
                self.assertTrue(env.observation_space.contains(obs))
                self.assertIsInstance(float(reward), float)
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIsInstance(info, dict)

                steps_seen += 1
                if terminated or truncated:
                    obs, info = env.reset(seed=steps_seen)
                    resets_seen += 1
                    self.assertTrue(env.observation_space.contains(obs))

            self.assertGreaterEqual(steps_seen, 10)
            self.assertGreaterEqual(resets_seen, 0)
        finally:
            env.close()

    def test_render_returns_rgb_array_when_called_explicitly(self) -> None:
        env = make_env(seed=0)
        try:
            env.reset(seed=0)
            frame = env.render()

            self.assertEqual(frame.shape, (INTERNAL_HEIGHT, INTERNAL_WIDTH, 3))
            self.assertEqual(frame.dtype, np.uint8)
        finally:
            env.close()

    def test_run_random_training_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "random.jsonl"

            results = run_random_training(episodes=2, max_steps=20, seed=0, action_repeat=4, output=output_path)

            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(results), 2)
        self.assertEqual(len(lines), 2)
        for index, line in enumerate(lines):
            payload: dict[str, Any] = json.loads(line)
            self.assertEqual(payload["episode"], index)
            self.assertIn("total_reward", payload)
            self.assertIn("length", payload)
            self.assertIn("terminated", payload)
            self.assertIn("truncated", payload)
            self.assertIn("game_over", payload)

    def test_train_random_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "random.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "rl/train_random.py",
                    "--episodes",
                    "2",
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
        self.assertIn("wrote 2 episode summaries", completed.stdout)
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
