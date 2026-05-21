from __future__ import annotations

from .base import BaseReward


class CustomReward(BaseReward):
    reward_name = "custom_reward"
    reward_weights = {
        "step": -0.01,
        "gold_delta": 1.0,
        "keys_delta": 5.0,
        "exit_reached": 50.0,
        "death": -20.0,
        "monster_kill": 10.0,
    }


def make_reward(**kwargs):
    return CustomReward(**kwargs)
