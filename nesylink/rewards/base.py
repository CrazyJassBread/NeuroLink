from __future__ import annotations

from typing import Any

from .context import build_reward_context, extract_reward_signals


class BaseReward:
    reward_name = "base"

    default_weights = {
        "step": 0.0,
        "hp_loss": 0.0,
        "gold_delta": 0.0,
        "keys_delta": 0.0,
        "monster_hit": 0.0,
        "monster_kill": 0.0,
        "door_opened": 0.0,
        "chest_opened": 0.0,
        "room_changed": 0.0,
        "exit_reached": 0.0,
        "death": 0.0,
        "invalid_action": 0.0,
    }

    reward_weights: dict[str, float] = {}

    def __init__(self, **reward_kwargs: float):
        self.prev_obs: Any = None
        self.prev_info: dict[str, Any] | None = None
        self.weights = dict(self.default_weights)
        self.weights.update(getattr(self, "reward_weights", {}))
        self.weights.update(reward_kwargs)

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        self.prev_obs = obs
        self.prev_info = info

    def __call__(self, obs: Any, info: dict[str, Any], action: int | None = None) -> tuple[float, dict[str, Any]]:
        signals = self.extract_signals(
            prev_obs=self.prev_obs,
            obs=obs,
            prev_info=self.prev_info,
            info=info,
            action=action,
        )
        reward = self.compute_reward(signals, obs, info, action)
        terminated, terminated_reason = self.check_termination(signals, obs, info, action)
        reward_info = self.build_reward_info(
            signals=signals,
            terminated=terminated,
            terminated_reason=terminated_reason,
        )
        self.prev_obs = obs
        self.prev_info = info
        return float(reward), reward_info

    def build_reward_info(
        self,
        *,
        signals: dict[str, Any] | None = None,
        terminated: bool = False,
        terminated_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "reward_name": self.reward_name,
            "reward_signals": dict(signals or {}),
            "reward_weights": dict(self.weights),
            "terminated": bool(terminated),
            "terminated_reason": terminated_reason,
        }

    def extract_signals(
        self,
        *,
        prev_obs: Any,
        obs: Any,
        prev_info: dict[str, Any] | None,
        info: dict[str, Any],
        action: int | None = None,
    ) -> dict[str, Any]:
        del prev_obs, obs
        context = build_reward_context(
            prev_info=prev_info,
            info=info,
            action=action,
        )
        return extract_reward_signals(context)

    def compute_reward(
        self,
        signals: dict[str, Any],
        obs: Any,
        info: dict[str, Any],
        action: int | None = None,
    ) -> float:
        reward = 0.0
        for key, weight in self.weights.items():
            reward += float(weight) * float(signals.get(key, 0.0))
        reward += self.extra_reward(signals, obs, info, action)
        return reward

    def extra_reward(
        self,
        signals: dict[str, Any],
        obs: Any,
        info: dict[str, Any],
        action: int | None = None,
    ) -> float:
        del signals, obs, info, action
        return 0.0

    def check_termination(
        self,
        signals: dict[str, Any],
        obs: Any,
        info: dict[str, Any],
        action: int | None = None,
    ) -> tuple[bool, str | None]:
        del signals, obs, info, action
        return False, None
