from __future__ import annotations

import unittest
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.utils.wrappers import ActionRepeatWrapper


class CountingEnv(gym.Env):
    observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
    action_space = spaces.Discrete(3)

    def __init__(self, outcomes: list[tuple[float, bool, bool]] | None = None) -> None:
        self.outcomes = outcomes or []
        self.step_calls = 0
        self.actions: list[int] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.step_calls = 0
        self.actions.clear()
        return np.array([0], dtype=np.uint8), {"reset": True}

    def step(self, action: int):
        self.step_calls += 1
        self.actions.append(action)
        index = self.step_calls - 1
        if index < len(self.outcomes):
            reward, terminated, truncated = self.outcomes[index]
        else:
            reward, terminated, truncated = float(self.step_calls), False, False
        obs = np.array([self.step_calls], dtype=np.uint8)
        return obs, reward, terminated, truncated, {"step_calls": self.step_calls}


class ActionRepeatWrapperTests(unittest.TestCase):
    def test_repeats_same_action_and_accumulates_reward(self) -> None:
        env = ActionRepeatWrapper(CountingEnv(), repeat=4)
        env.reset(seed=0)

        obs, reward, terminated, truncated, info = env.step(2)

        self.assertEqual(env.unwrapped.step_calls, 4)
        self.assertEqual(env.unwrapped.actions, [2, 2, 2, 2])
        self.assertEqual(obs.tolist(), [4])
        self.assertEqual(reward, 10.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action_repeat"], 4)
        self.assertEqual(info["inner_steps"], 4)
        self.assertEqual(info["repeated_reward"], 10.0)
        self.assertEqual(info["step_calls"], 4)

    def test_stops_early_on_terminated(self) -> None:
        env = ActionRepeatWrapper(
            CountingEnv(outcomes=[(1.0, False, False), (2.0, True, False), (3.0, False, False)]),
            repeat=4,
        )
        env.reset(seed=0)

        obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(env.unwrapped.step_calls, 2)
        self.assertEqual(reward, 3.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["inner_steps"], 2)

    def test_stops_early_on_truncated(self) -> None:
        env = ActionRepeatWrapper(
            CountingEnv(outcomes=[(1.0, False, False), (2.0, False, True), (3.0, False, False)]),
            repeat=4,
        )
        env.reset(seed=0)

        obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(env.unwrapped.step_calls, 2)
        self.assertEqual(reward, 3.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["inner_steps"], 2)

    def test_repeat_one_matches_single_inner_step(self) -> None:
        env = ActionRepeatWrapper(CountingEnv(), repeat=1)
        env.reset(seed=0)

        obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(env.unwrapped.step_calls, 1)
        self.assertEqual(obs.tolist(), [1])
        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action_repeat"], 1)
        self.assertEqual(info["inner_steps"], 1)

    def test_invalid_repeat_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "repeat must be >= 1"):
            ActionRepeatWrapper(CountingEnv(), repeat=0)

        with self.assertRaisesRegex(ValueError, "repeat must be >= 1"):
            ActionRepeatWrapper(CountingEnv(), repeat=-1)


if __name__ == "__main__":
    unittest.main()
