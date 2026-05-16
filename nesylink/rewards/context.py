from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_int(mapping: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class RewardContext:
    prev_info: Mapping[str, Any]
    info: Mapping[str, Any]
    action: int | None
    prev_agent: Mapping[str, Any]
    agent: Mapping[str, Any]
    prev_inventory: Mapping[str, Any]
    inventory: Mapping[str, Any]
    prev_entities: Mapping[str, Any]
    entities: Mapping[str, Any]
    event_counts: Mapping[str, Any]
    event_flags: Mapping[str, Any]
    game: Mapping[str, Any]


def build_reward_context(
    *,
    prev_info: dict[str, Any] | None,
    info: dict[str, Any],
    action: int | None = None,
) -> RewardContext:
    previous = prev_info or {}
    events = _mapping(info.get("events"))
    return RewardContext(
        prev_info=previous,
        info=info,
        action=action,
        prev_agent=_mapping(previous.get("agent")),
        agent=_mapping(info.get("agent")),
        prev_inventory=_mapping(previous.get("inventory")),
        inventory=_mapping(info.get("inventory")),
        prev_entities=_mapping(previous.get("entities")),
        entities=_mapping(info.get("entities")),
        event_counts=_mapping(events.get("counts")),
        event_flags=_mapping(events.get("flags")),
        game=_mapping(info.get("game")),
    )


def extract_reward_signals(context: RewardContext) -> dict[str, Any]:
    hp_delta = _nested_int(context.agent, "hp") - _nested_int(
        context.prev_agent,
        "hp",
        _nested_int(context.agent, "hp"),
    )
    gold_delta_raw = _nested_int(context.inventory, "gold") - _nested_int(
        context.prev_inventory,
        "gold",
        _nested_int(context.inventory, "gold"),
    )
    keys_delta_raw = _nested_int(context.inventory, "keys") - _nested_int(
        context.prev_inventory,
        "keys",
        _nested_int(context.inventory, "keys"),
    )
    monsters_remaining = _nested_int(context.entities, "monsters_remaining")
    prev_monsters_remaining = _nested_int(context.prev_entities, "monsters_remaining", monsters_remaining)

    monster_hit = _nested_int(context.event_counts, "monster_damaged")
    if monster_hit <= 0:
        monster_hit = _nested_int(context.event_counts, "action_attack")

    invalid_action = int(
        bool(context.event_flags.get("action_blocked", False))
        or bool(context.event_flags.get("action_no_effect", False))
        or _nested_int(context.event_counts, "action_blocked") > 0
        or _nested_int(context.event_counts, "action_no_effect") > 0
    )

    return {
        "step": 1,
        "hp_delta": hp_delta,
        "hp_loss": max(0, -hp_delta),
        "gold_delta": max(0, gold_delta_raw),
        "keys_delta": max(0, keys_delta_raw),
        "monster_hit": monster_hit,
        "monster_kill": _nested_int(context.event_counts, "monster_killed"),
        "door_opened": _nested_int(context.event_counts, "door_opened"),
        "chest_opened": _nested_int(context.event_counts, "chest_opened"),
        "room_changed": int(bool(context.game.get("room_changed", False))),
        "exit_reached": int(bool(context.game.get("exit_reached", False))),
        "death": int(bool(context.game.get("dead", False))),
        "invalid_action": invalid_action,
        "monsters_remaining": monsters_remaining,
        "prev_monsters_remaining": prev_monsters_remaining,
    }
