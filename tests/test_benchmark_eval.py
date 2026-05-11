from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from env_diy.benchmark.eval import evaluate_suite


class BenchmarkEvalTests(unittest.TestCase):
    def test_random_eval_runs_single_episode(self) -> None:
        result = evaluate_suite("NesyLink-v0", policy="random", episodes=1, seed=0)
        self.assertEqual(result["suite_id"], "NesyLink-v0")
        self.assertIn("aggregate", result)
        self.assertIn("tasks", result)
        self.assertIn("mean_success_rate", result["aggregate"])
        self.assertIn("mean_return", result["aggregate"])
        self.assertIn("mean_episode_length", result["aggregate"])

    def test_reward_mode_override_is_respected(self) -> None:
        event_result = evaluate_suite("NesyLink-v0", policy="random", episodes=1, seed=0, reward_mode="event")
        sparse_result = evaluate_suite("NesyLink-v0", policy="random", episodes=1, seed=0, reward_mode="sparse")
        self.assertEqual(event_result["reward_mode"], "event")
        self.assertEqual(sparse_result["reward_mode"], "sparse")

    def test_json_output_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "benchmark.json"
            result = evaluate_suite("NesyLink-v0", policy="random", episodes=1, seed=0, json_output=output)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["suite_id"], result["suite_id"])
        self.assertIn("tasks", payload)


if __name__ == "__main__":
    unittest.main()
