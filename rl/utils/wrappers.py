from __future__ import annotations

from typing import Any

import gymnasium as gym


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat one agent action for multiple underlying environment ticks."""

    def __init__(self, env: gym.Env, repeat: int = 4):
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        native_repeat = int(getattr(env, "native_action_repeat", 1))
        if native_repeat > 1 and repeat > 1:
            raise ValueError("cannot combine env native action_repeat with ActionRepeatWrapper")
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action: Any):
        total_reward = 0.0
        inner_steps = 0

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            inner_steps += 1
            if terminated or truncated:
                break

        repeated_info = dict(info)
        repeated_info["action_repeat"] = self.repeat
        repeated_info["inner_steps"] = inner_steps
        repeated_info["repeated_reward"] = total_reward
        return obs, total_reward, terminated, truncated, repeated_info
