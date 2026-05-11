from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.events import event_records_to_counts, normalize_event_records
from ..core.types import RewardTerms, RuntimeSnapshot, StuckPenaltyConfig
from ..maps.tasks import TaskConfig


@dataclass(frozen=True)
class RewardConfig:
    reward_mode: str = "legacy"
    stuck_penalty: StuckPenaltyConfig = StuckPenaltyConfig(enabled=False, steps=30, reward=-0.01)


def compute_reward(
    prev_state: RuntimeSnapshot,
    state: RuntimeSnapshot,
    events: Any,
    task_spec: TaskConfig | None = None,
    config: RewardConfig | None = None,
) -> tuple[float, RewardTerms]:
    reward_config = config or RewardConfig()
    normalized = _normalize_events(events)
    reward_mode = reward_config.reward_mode
    if reward_mode == "legacy":
        return _compute_legacy_reward(prev_state, state, normalized, task_spec, reward_config.stuck_penalty)
    if reward_mode == "event":
        return _compute_event_reward(prev_state, state, normalized, task_spec)
    if reward_mode == "sparse":
        return _compute_sparse_reward(normalized, task_spec)
    raise ValueError(f"unsupported reward_mode '{reward_mode}'")


def _compute_legacy_reward(
    prev_state: RuntimeSnapshot,
    next_state: RuntimeSnapshot,
    normalized: dict[str, Any],
    task_config: TaskConfig | None,
    stuck_config: StuckPenaltyConfig,
) -> tuple[float, RewardTerms]:
    reward = 0.0
    breakdown: RewardTerms = {}
    events = normalized["events"]
    event_details = normalized["event_details"]

    def add(label: str, value: float) -> None:
        nonlocal reward
        reward += value
        breakdown[label] = breakdown.get(label, 0.0) + value

    if any(event.startswith("move_") for event in events):
        add("movement", _task_reward(task_config, "step", -0.01))
    if "blocked_wall" in events or "blocked_bounds" in events:
        add("blocked_movement", -0.02)
    if "blocked_locked" in events or "missing_requirement" in events:
        add("blocked_exit", -0.02)
    if "action_a_empty" in events or "action_b_empty" in events:
        add("empty_action", -0.01)
    if "room_transition" in events:
        add("room_transition", 0.1)
    if "pressed_button" in events:
        add("button_press", 0.1)
    if "trap_damage" in events:
        add("trap_damage", _task_reward(task_config, "damage", -0.5))
    if "monster_hit" in events:
        add("monster_hit", _task_reward(task_config, "damage", -0.4))
    if "monster_killed" in events:
        add("monster_kill", _task_reward(task_config, "monster_kill", 0.3))
    if "got_key" in events:
        add("got_key", _task_reward(task_config, "key", 0.4))
    if "door_unlocked" in events and any(
        detail.get("type") == "door_unlocked" and detail.get("trigger") != "all_monsters_defeated"
        for detail in event_details
    ):
        add("door_unlock", _task_reward(task_config, "door_unlock", 0.0))
    if "got_gold" in events:
        add("got_gold", 0.2)
    if "got_item" in events:
        add("got_item", 0.3)
    if "healed" in events:
        healed_amount = max(0, next_state.health - prev_state.health)
        add("healed", 0.2 if healed_amount > 0 else 0.05)
    if "task_finished" in events:
        add("task_finished", _task_reward(task_config, "finish", 10.0))
    if normalized["grant_victory_reward"]:
        add("victory", 1.0)
    if stuck_config.enabled and next_state.no_progress_steps >= stuck_config.steps:
        progress_events = {
            "door_unlocked",
            "got_gold",
            "got_item",
            "got_key",
            "healed",
            "monster_killed",
            "opened_chest",
            "pressed_button",
            "room_transition",
            "task_finished",
            "talked_npc",
            "victory",
        }
        if prev_state.player_position_px == next_state.player_position_px and prev_state.room_id == next_state.room_id:
            if not any(event in progress_events for event in events):
                add("stuck_penalty", stuck_config.reward)
    return reward, breakdown


def _compute_event_reward(
    prev_state: RuntimeSnapshot,
    next_state: RuntimeSnapshot,
    normalized: dict[str, Any],
    task_config: TaskConfig | None,
) -> tuple[float, RewardTerms]:
    del prev_state
    reward = 0.0
    terms: RewardTerms = {}
    counts = normalized["event_counts"]

    def add(name: str, value: float) -> None:
        nonlocal reward
        reward += value
        terms[name] = terms.get(name, 0.0) + value

    if counts.get("move_up", 0) or counts.get("move_down", 0) or counts.get("move_left", 0) or counts.get("move_right", 0):
        add("step_penalty", _task_reward(task_config, "step", -0.01))
    if counts.get("got_key", 0):
        add("picked_key", counts["got_key"] * _task_reward(task_config, "key", 0.4))
    if counts.get("door_unlocked", 0):
        add("opened_door", counts["door_unlocked"] * _task_reward(task_config, "door_unlock", 0.2))
    if counts.get("got_gold", 0):
        add("picked_coin", counts["got_gold"] * 0.2)
    if counts.get("monster_killed", 0):
        add("killed_monster", counts["monster_killed"] * _task_reward(task_config, "monster_kill", 0.3))
    if counts.get("trap_damage", 0):
        add("hit_trap", counts["trap_damage"] * _task_reward(task_config, "damage", -1.0))
    if counts.get("game_over", 0) or next_state.health <= 0:
        add("agent_dead", -1.0)
    if counts.get("task_finished", 0) or counts.get("victory", 0):
        add("reached_goal", _task_reward(task_config, "finish", 10.0))
    return reward, terms


def _compute_sparse_reward(
    normalized: dict[str, Any],
    task_config: TaskConfig | None,
) -> tuple[float, RewardTerms]:
    reward = 0.0
    terms: RewardTerms = {}
    if normalized["event_counts"].get("task_finished", 0) or normalized["event_counts"].get("victory", 0):
        reward = _task_reward(task_config, "finish", 10.0)
        terms["reached_goal"] = reward
    return reward, terms


def _normalize_events(events: Any) -> dict[str, Any]:
    if hasattr(events, "events") and hasattr(events, "event_details"):
        event_list = list(events.events)
        event_details = list(events.event_details)
        grant_victory_reward = bool(getattr(events, "grant_victory_reward", False))
    elif isinstance(events, dict):
        event_list = list(events.get("events", []))
        event_details = list(events.get("event_details", []))
        grant_victory_reward = bool(events.get("grant_victory_reward", False))
    else:
        event_list = list(events)
        event_details = []
        grant_victory_reward = False
    event_records = normalize_event_records(event_list, event_details)
    return {
        "events": event_list,
        "event_details": event_details,
        "event_records": event_records,
        "event_counts": event_records_to_counts(event_records),
        "grant_victory_reward": grant_victory_reward,
    }


def _task_reward(task_config: TaskConfig | None, field_name: str, default: float) -> float:
    if task_config is None:
        return default
    return float(getattr(task_config.reward, field_name, default))
