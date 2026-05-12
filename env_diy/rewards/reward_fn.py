from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.events import event_records_to_counts, normalize_event_records
from ..core.types import RewardTerms, RuntimeSnapshot, StuckPenaltyConfig
from ..maps.tasks import TaskConfig

DEFAULT_REWARD_MODE = "default"
SUPPORTED_REWARD_MODES = (DEFAULT_REWARD_MODE, "event", "sparse")


@dataclass(frozen=True)
class RewardRule:
    term: str
    events: tuple[str, ...]
    default_value: float
    task_field: str | None = None
    repeat_by_count: bool = False


DEFAULT_MODE_REWARD_RULES: tuple[RewardRule, ...] = (
    RewardRule("movement", ("move_up", "move_down", "move_left", "move_right"), -0.01, task_field="step"),
    RewardRule("blocked_movement", ("blocked_wall", "blocked_bounds"), -0.02),
    RewardRule("blocked_exit", ("blocked_locked", "missing_requirement"), -0.02),
    RewardRule("empty_action", ("action_a_empty", "action_b_empty"), -0.01),
    RewardRule("room_transition", ("room_transition",), 0.1),
    RewardRule("button_press", ("pressed_button",), 0.1),
    RewardRule("trap_damage", ("trap_damage",), -0.5, task_field="damage"),
    RewardRule("monster_hit", ("monster_hit",), -0.4, task_field="damage"),
    RewardRule("monster_kill", ("monster_killed",), 0.3, task_field="monster_kill"),
    RewardRule("got_key", ("got_key",), 0.4, task_field="key"),
    RewardRule("got_gold", ("got_gold",), 0.2),
    RewardRule("got_item", ("got_item",), 0.3),
    RewardRule("task_finished", ("task_finished",), 10.0, task_field="finish"),
)

EVENT_MODE_REWARD_RULES: tuple[RewardRule, ...] = (
    RewardRule("step_penalty", ("move_up", "move_down", "move_left", "move_right"), -0.01, task_field="step"),
    RewardRule("picked_key", ("got_key",), 0.4, task_field="key", repeat_by_count=True),
    RewardRule("opened_door", ("door_unlocked",), 0.2, task_field="door_unlock", repeat_by_count=True),
    RewardRule("picked_coin", ("got_gold",), 0.2, repeat_by_count=True),
    RewardRule("killed_monster", ("monster_killed",), 0.3, task_field="monster_kill", repeat_by_count=True),
    RewardRule("hit_trap", ("trap_damage",), -1.0, task_field="damage", repeat_by_count=True),
)

STUCK_PROGRESS_EVENTS = {
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


@dataclass(frozen=True)
class RewardConfig:
    reward_mode: str = DEFAULT_REWARD_MODE
    stuck_penalty: StuckPenaltyConfig = StuckPenaltyConfig(enabled=False, steps=30, reward=-0.01)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_mode", normalize_reward_mode(self.reward_mode))


def normalize_reward_mode(reward_mode: str) -> str:
    normalized = str(reward_mode)
    if normalized not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"unsupported reward_mode '{reward_mode}'")
    return normalized


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
    if reward_mode == DEFAULT_REWARD_MODE:
        return _compute_default_reward(prev_state, state, normalized, task_spec, reward_config.stuck_penalty)
    if reward_mode == "event":
        return _compute_event_reward(state, normalized, task_spec)
    if reward_mode == "sparse":
        return _compute_sparse_reward(normalized, task_spec)
    raise ValueError(f"unsupported reward_mode '{reward_mode}'")


def _compute_default_reward(
    prev_state: RuntimeSnapshot,
    next_state: RuntimeSnapshot,
    normalized: dict[str, Any],
    task_config: TaskConfig | None,
    stuck_config: StuckPenaltyConfig,
) -> tuple[float, RewardTerms]:
    reward = 0.0
    terms: RewardTerms = {}
    event_details = normalized["event_details"]

    reward = _apply_reward_rules(reward, terms, DEFAULT_MODE_REWARD_RULES, normalized["event_counts"], task_config)

    if _door_unlock_reward_applies(event_details):
        reward = _add_reward_term(reward, terms, "door_unlock", _task_reward(task_config, "door_unlock", 0.0))

    if "healed" in normalized["event_counts"]:
        healed_amount = max(0, next_state.health - prev_state.health)
        heal_reward = 0.2 if healed_amount > 0 else 0.05
        reward = _add_reward_term(reward, terms, "healed", heal_reward)

    if normalized["grant_victory_reward"]:
        reward = _add_reward_term(reward, terms, "victory", 1.0)

    if _stuck_penalty_applies(prev_state, next_state, normalized["events"], stuck_config):
        reward = _add_reward_term(reward, terms, "stuck_penalty", stuck_config.reward)

    return reward, terms


def _compute_event_reward(
    next_state: RuntimeSnapshot,
    normalized: dict[str, Any],
    task_config: TaskConfig | None,
) -> tuple[float, RewardTerms]:
    reward = 0.0
    terms: RewardTerms = {}
    reward = _apply_reward_rules(reward, terms, EVENT_MODE_REWARD_RULES, normalized["event_counts"], task_config)

    if normalized["event_counts"].get("game_over", 0) or next_state.health <= 0:
        reward = _add_reward_term(reward, terms, "agent_dead", -1.0)
    if normalized["event_counts"].get("task_finished", 0) or normalized["event_counts"].get("victory", 0):
        reward = _add_reward_term(reward, terms, "reached_goal", _task_reward(task_config, "finish", 10.0))

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


def _apply_reward_rules(
    reward: float,
    terms: RewardTerms,
    rules: tuple[RewardRule, ...],
    event_counts: dict[str, int],
    task_config: TaskConfig | None,
) -> float:
    for rule in rules:
        event_count = sum(int(event_counts.get(event_name, 0)) for event_name in rule.events)
        if event_count <= 0:
            continue
        multiplier = event_count if rule.repeat_by_count else 1
        value = _rule_value(rule, task_config) * multiplier
        reward = _add_reward_term(reward, terms, rule.term, value)
    return reward


def _rule_value(rule: RewardRule, task_config: TaskConfig | None) -> float:
    if rule.task_field is None:
        return float(rule.default_value)
    return _task_reward(task_config, rule.task_field, rule.default_value)


def _door_unlock_reward_applies(event_details: list[dict[str, Any]]) -> bool:
    return any(
        detail.get("type") == "door_unlocked" and detail.get("trigger") != "all_monsters_defeated"
        for detail in event_details
    )


def _stuck_penalty_applies(
    prev_state: RuntimeSnapshot,
    next_state: RuntimeSnapshot,
    events: list[str],
    stuck_config: StuckPenaltyConfig,
) -> bool:
    if not stuck_config.enabled or next_state.no_progress_steps < stuck_config.steps:
        return False
    if prev_state.player_position_px != next_state.player_position_px:
        return False
    if prev_state.room_id != next_state.room_id:
        return False
    return not any(event_name in STUCK_PROGRESS_EVENTS for event_name in events)


def _add_reward_term(reward: float, terms: RewardTerms, term: str, value: float) -> float:
    reward += value
    terms[term] = terms.get(term, 0.0) + float(value)
    return reward


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
