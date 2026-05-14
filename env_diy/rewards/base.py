from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_int(mapping: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return int(default)


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
        del prev_obs, obs, action
        prev_info = prev_info or {}

        prev_agent = _mapping(prev_info.get("agent"))
        agent = _mapping(info.get("agent"))
        prev_inventory = _mapping(prev_info.get("inventory"))
        inventory = _mapping(info.get("inventory"))
        prev_entities = _mapping(prev_info.get("entities"))
        entities = _mapping(info.get("entities"))
        events = _mapping(info.get("events"))
        event_counts = _mapping(events.get("counts"))
        event_flags = _mapping(events.get("flags"))
        game = _mapping(info.get("game"))

        hp_delta = _nested_int(agent, "hp") - _nested_int(prev_agent, "hp", _nested_int(agent, "hp"))
        gold_delta_raw = _nested_int(inventory, "gold") - _nested_int(prev_inventory, "gold", _nested_int(inventory, "gold"))
        keys_delta_raw = _nested_int(inventory, "keys") - _nested_int(prev_inventory, "keys", _nested_int(inventory, "keys"))
        monsters_remaining = _nested_int(entities, "monsters_remaining")
        prev_monsters_remaining = _nested_int(prev_entities, "monsters_remaining", monsters_remaining)

        monster_hit = _nested_int(event_counts, "monster_damaged")
        if monster_hit <= 0:
            monster_hit = _nested_int(event_counts, "action_attack")

        invalid_action = int(
            bool(event_flags.get("action_blocked", False))
            or bool(event_flags.get("action_no_effect", False))
            or _nested_int(event_counts, "action_blocked") > 0
            or _nested_int(event_counts, "action_no_effect") > 0
        )

        return {
            "step": 1,
            "hp_delta": hp_delta,
            "hp_loss": max(0, -hp_delta),
            "gold_delta": max(0, gold_delta_raw),
            "keys_delta": max(0, keys_delta_raw),
            "monster_hit": monster_hit,
            "monster_kill": _nested_int(event_counts, "monster_killed"),
            "door_opened": _nested_int(event_counts, "door_opened"),
            "chest_opened": _nested_int(event_counts, "chest_opened"),
            "room_changed": int(bool(game.get("room_changed", False))),
            "exit_reached": int(bool(game.get("exit_reached", False))),
            "death": int(bool(game.get("dead", False))),
            "invalid_action": invalid_action,
            "monsters_remaining": monsters_remaining,
            "prev_monsters_remaining": prev_monsters_remaining,
        }

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
