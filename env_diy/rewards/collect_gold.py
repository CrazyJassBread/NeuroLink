from __future__ import annotations

from .base import BaseReward


class CollectGoldReward(BaseReward):
    reward_name = "collect_gold"
    reward_weights = {
        "step": -0.01,
        "gold_delta": 1.0,
        "keys_delta": 3.0,
        "chest_opened": 2.0,
        "exit_reached": 20.0,
        "death": -10.0,
    }

    def check_termination(self, signals, obs, info, action=None):
        del obs, info, action
        if signals.get("exit_reached", 0) > 0:
            return True, "exit_reached"
        return False, None


def make_reward(**kwargs):
    return CollectGoldReward(**kwargs)
