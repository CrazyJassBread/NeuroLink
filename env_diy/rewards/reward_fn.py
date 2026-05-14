from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tasks.task_spec import RewardFn, TaskOutcome, TaskSpec


def _event_count(info: dict[str, Any], name: str) -> int:
    return int(info.get("events", {}).get("counts", {}).get(name, 0))


def _event_records(info: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        record
        for record in info.get("events", {}).get("records", [])
        if record.get("name") == name
    ]


def _entity_value(info: dict[str, Any], name: str, default: int = 0) -> int:
    return int(info.get("entities", {}).get(name, default))


def _target_exit_reached(info: dict[str, Any], target_exit: str | None) -> bool:
    records = _event_records(info, "exit_reached")
    if records:
        for record in records:
            if target_exit is None or record.get("exit_id") == target_exit:
                return True
        return False
    return _event_count(info, "exit_reached") > 0


@dataclass
class EventDrivenRewardFn(RewardFn):
    task_spec: TaskSpec

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_outcome = TaskOutcome()
        self.last_reward_terms: dict[str, float] = {}

    def _weight(self, event_name: str, default: float = 0.0) -> float:
        return float(self.task_spec.reward_weights.get(event_name, default))

    def _apply_event_reward(self, info: dict[str, Any], event_name: str, default: float = 0.0) -> float:
        count = _event_count(info, event_name)
        if count <= 0:
            return 0.0
        value = self._weight(event_name, default) * count
        self.last_reward_terms[event_name] = self.last_reward_terms.get(event_name, 0.0) + value
        return value


class KeyDoorReward(EventDrivenRewardFn):
    def __call__(
        self,
        prev_obs: Any,
        prev_info: dict[str, Any],
        obs: Any,
        info: dict[str, Any],
        action: int,
    ) -> tuple[float, bool]:
        del prev_obs, prev_info, obs, action
        self.last_reward_terms = {}
        reward = 0.0
        reward += self._apply_event_reward(info, "key_collected", default=0.5)
        reward += self._apply_event_reward(info, "door_opened", default=0.5)

        if _target_exit_reached(info, self.task_spec.target_exit):
            reward += self._apply_event_reward(info, "exit_reached", default=10.0)
            self.last_outcome = TaskOutcome(success=True, terminated_reason="exit_reached", progress=1.0)
            return reward, True

        progress = 0.0
        if _event_count(info, "key_collected") > 0 or _event_count(info, "door_opened") > 0:
            progress = 0.5
        self.last_outcome = TaskOutcome(progress=progress)
        return reward, False


class KillMonsterReward(EventDrivenRewardFn):
    def __call__(
        self,
        prev_obs: Any,
        prev_info: dict[str, Any],
        obs: Any,
        info: dict[str, Any],
        action: int,
    ) -> tuple[float, bool]:
        del prev_obs, obs, action
        self.last_reward_terms = {}
        reward = self._apply_event_reward(info, "monster_killed", default=0.5)

        monsters_remaining = _entity_value(info, "monsters_remaining")
        previous_remaining = _entity_value(prev_info, "monsters_remaining", default=monsters_remaining + 1)
        if monsters_remaining == 0 and previous_remaining > 0:
            clear_reward = self._weight("room_cleared", 10.0)
            self.last_reward_terms["room_cleared"] = clear_reward
            reward += clear_reward
            self.last_outcome = TaskOutcome(
                success=True,
                terminated_reason="all_monsters_defeated",
                progress=1.0,
            )
            return reward, True

        progress = 0.0
        if previous_remaining > 0:
            progress = max(0.0, min(1.0, 1.0 - (monsters_remaining / previous_remaining)))
        self.last_outcome = TaskOutcome(progress=progress)
        return reward, False


class AvoidTrapReward(EventDrivenRewardFn):
    def reset(self) -> None:
        super().reset()
        self._trap_triggered = False

    def __call__(
        self,
        prev_obs: Any,
        prev_info: dict[str, Any],
        obs: Any,
        info: dict[str, Any],
        action: int,
    ) -> tuple[float, bool]:
        del prev_obs, prev_info, obs, action
        self.last_reward_terms = {}
        reward = self._apply_event_reward(info, "trap_triggered", default=-1.0)
        if _event_count(info, "trap_triggered") > 0:
            self._trap_triggered = True

        if _target_exit_reached(info, self.task_spec.target_exit):
            if not self._trap_triggered:
                reward += self._apply_event_reward(info, "exit_reached", default=10.0)
                self.last_outcome = TaskOutcome(success=True, terminated_reason="exit_reached", progress=1.0)
            else:
                self.last_outcome = TaskOutcome(
                    failure=True,
                    terminated_reason="exit_reached_after_trap",
                    progress=0.0,
                )
            return reward, True

        self.last_outcome = TaskOutcome(progress=0.0)
        return reward, False
