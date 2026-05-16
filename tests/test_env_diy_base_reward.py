from __future__ import annotations

from nesylink.rewards.context import build_reward_context, extract_reward_signals
from nesylink.rewards.base import BaseReward


class _TestReward(BaseReward):
    reward_name = "test_reward"
    reward_weights = {
        "step": -0.5,
        "hp_loss": -2.0,
        "gold_delta": 1.5,
        "keys_delta": 3.0,
        "monster_hit": 2.0,
        "monster_kill": 10.0,
        "door_opened": 4.0,
        "chest_opened": 1.0,
        "room_changed": 5.0,
        "exit_reached": 7.0,
        "death": -9.0,
        "invalid_action": -0.25,
    }


def _info(
    *,
    hp: int,
    gold: int,
    keys: int,
    monsters_remaining: int = 0,
    room_id: str = "room_a",
    counts: dict[str, int] | None = None,
    dead: bool = False,
    room_changed: bool = False,
    exit_reached: bool = False,
) -> dict:
    return {
        "episode": {"step_count": 1},
        "env": {"room_id": room_id},
        "agent": {"hp": hp},
        "inventory": {"gold": gold, "keys": keys, "items": [], "tools": [], "equipped": {}},
        "entities": {"monsters_remaining": monsters_remaining},
        "events": {"counts": counts or {}, "flags": {}, "records": [], "details": []},
        "game": {
            "dead": dead,
            "room_changed": room_changed,
            "exit_reached": exit_reached,
        },
    }


def test_base_reward_extracts_common_signals_and_reward_info() -> None:
    reward = _TestReward()
    reward.reset({}, _info(hp=6, gold=1, keys=0, monsters_remaining=2))

    value, reward_info = reward(
        {},
        _info(
            hp=4,
            gold=4,
            keys=1,
            monsters_remaining=1,
            room_id="room_b",
            counts={
                "monster_damaged": 1,
                "monster_killed": 1,
                "door_opened": 1,
                "chest_opened": 1,
                "action_blocked": 1,
            },
            room_changed=True,
            exit_reached=True,
        ),
        action=5,
    )

    signals = reward_info["reward_signals"]
    assert signals["step"] == 1
    assert signals["hp_delta"] == -2
    assert signals["hp_loss"] == 2
    assert signals["gold_delta"] == 3
    assert signals["keys_delta"] == 1
    assert signals["monster_hit"] == 1
    assert signals["monster_kill"] == 1
    assert signals["door_opened"] == 1
    assert signals["chest_opened"] == 1
    assert signals["room_changed"] == 1
    assert signals["exit_reached"] == 1
    assert signals["death"] == 0
    assert signals["invalid_action"] == 1
    assert reward_info["reward_name"] == "test_reward"
    assert reward_info["reward_weights"]["monster_kill"] == 10.0
    assert value == (
        -0.5
        + (-2.0 * 2)
        + (1.5 * 3)
        + (3.0 * 1)
        + (2.0 * 1)
        + (10.0 * 1)
        + (4.0 * 1)
        + (1.0 * 1)
        + (5.0 * 1)
        + (7.0 * 1)
        + (-0.25 * 1)
    )


def test_base_reward_handles_initial_call_without_previous_info() -> None:
    reward = _TestReward()

    value, reward_info = reward({}, _info(hp=5, gold=0, keys=0), action=0)

    assert value == -0.5
    assert reward_info["reward_signals"]["step"] == 1
    assert reward_info["reward_signals"]["hp_loss"] == 0
    assert reward_info["reward_signals"]["gold_delta"] == 0
    assert reward_info["reward_signals"]["keys_delta"] == 0


def test_base_reward_detects_death_from_game_flags() -> None:
    reward = _TestReward()
    reward.reset({}, _info(hp=1, gold=0, keys=0))

    value, reward_info = reward({}, _info(hp=0, gold=0, keys=0, dead=True), action=0)

    assert reward_info["reward_signals"]["death"] == 1
    assert value == -0.5 - 2.0 - 9.0


def test_reward_context_builds_structured_views_and_common_signals() -> None:
    context = build_reward_context(
        prev_info=_info(hp=6, gold=1, keys=0, monsters_remaining=2),
        info=_info(
            hp=4,
            gold=4,
            keys=1,
            monsters_remaining=1,
            room_id="room_b",
            counts={
                "monster_damaged": 1,
                "monster_killed": 1,
                "door_opened": 1,
                "chest_opened": 1,
                "action_blocked": 1,
            },
            room_changed=True,
            exit_reached=True,
        ),
        action=5,
    )

    signals = extract_reward_signals(context)
    assert context.agent["hp"] == 4
    assert context.prev_agent["hp"] == 6
    assert context.event_counts["monster_killed"] == 1
    assert context.game["room_changed"] is True
    assert signals["hp_loss"] == 2
    assert signals["gold_delta"] == 3
    assert signals["keys_delta"] == 1
    assert signals["monster_hit"] == 1
    assert signals["monster_kill"] == 1
    assert signals["door_opened"] == 1
    assert signals["chest_opened"] == 1
    assert signals["room_changed"] == 1
    assert signals["exit_reached"] == 1
    assert signals["invalid_action"] == 1
