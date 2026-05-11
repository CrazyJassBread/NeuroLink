from __future__ import annotations

import unittest

from env_diy.benchmark.registry import (
    get_suite,
    get_task_spec,
    list_suites,
    list_tasks,
    make_benchmark_env,
)


class BenchmarkRegistryTests(unittest.TestCase):
    def test_list_suites_contains_nesylink_v0(self) -> None:
        self.assertIn("NesyLink-v0", list_suites())

    def test_list_tasks_contains_initial_v0_tasks(self) -> None:
        task_ids = {task.task_id for task in list_tasks("NesyLink-v0")}
        self.assertEqual(task_ids, {"prototype", "avoid_traps", "kill_monsters", "key_door"})

    def test_task_spec_fields_are_complete(self) -> None:
        task = get_task_spec("NesyLink-v0", "avoid_traps")
        self.assertEqual(task.suite_id, "NesyLink-v0")
        self.assertEqual(task.task_id, "avoid_traps")
        self.assertIsNotNone(task.map_id)
        self.assertGreater(task.max_episode_steps, 0)
        self.assertEqual(task.default_reward_mode, "legacy")
        self.assertIn("legacy", task.supported_reward_modes)
        self.assertTrue(task.success_condition)
        self.assertTrue(task.failure_condition)

    def test_make_benchmark_env_can_reset_and_step(self) -> None:
        env = make_benchmark_env("NesyLink-v0", "prototype", seed=0)
        try:
            obs, info = env.reset(seed=0)
            self.assertIn("grid", obs)
            step = env.step(0)
            self.assertEqual(len(step), 5)
        finally:
            env.close()

    def test_get_suite_returns_task_specs(self) -> None:
        suite = get_suite("NesyLink-v0")
        self.assertEqual(suite.suite_id, "NesyLink-v0")
        self.assertEqual(len(suite.tasks), 4)


if __name__ == "__main__":
    unittest.main()
