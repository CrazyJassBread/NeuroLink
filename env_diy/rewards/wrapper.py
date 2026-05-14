from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from ..tasks.task_spec import RewardFn, TaskOutcome


@dataclass
class EpisodeOutcome:
    success: bool = False
    failure: bool = False
    terminated_reason: str | None = None
    progress: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.last_outcome = EpisodeOutcome()
        self.episode_outcome = EpisodeOutcome()
        self.last_reward_terms: dict[str, float] = {}
        self._prev_obs: Any = None
        self._prev_info: dict[str, Any] | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_fn.reset()
        self.last_outcome = EpisodeOutcome()
        self.episode_outcome = EpisodeOutcome()
        self.last_reward_terms = {}
        self._prev_obs = obs
        self._prev_info = info
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        extra_reward, wrapper_terminated = self.reward_fn(
            self._prev_obs,
            self._prev_info or {},
            obs,
            info,
            int(action),
        )
        self.last_reward_terms = dict(getattr(self.reward_fn, "last_reward_terms", {}))
        reward_outcome = getattr(self.reward_fn, "last_outcome", TaskOutcome())
        self.last_outcome = EpisodeOutcome(
            success=bool(reward_outcome.success),
            failure=bool(reward_outcome.failure),
            terminated_reason=reward_outcome.terminated_reason,
            progress=reward_outcome.progress,
            metadata=dict(reward_outcome.metadata),
        )
        if terminated and info.get("terminal_reason") == "agent_dead":
            self.last_outcome = EpisodeOutcome(
                success=False,
                failure=True,
                terminated_reason="agent_dead",
                progress=reward_outcome.progress,
                metadata=dict(reward_outcome.metadata),
            )

        final_terminated = bool(terminated or wrapper_terminated)
        if final_terminated or truncated:
            self.episode_outcome = self.last_outcome

        self._prev_obs = obs
        self._prev_info = info
        return obs, float(base_reward) + float(extra_reward), final_terminated, truncated, info
