from __future__ import annotations

from .base import BaseReward


class CollectKeyReward(BaseReward):
    reward_name = "collect_key"
    reward_weights = {
        "step": -0.01,
        "keys_delta": 5.0,
        "door_opened": 3.0,
        "exit_reached": 20.0,
        "death": -10.0,
        "invalid_action": -0.05,
    }

    def check_termination(self, signals, obs, info, action=None):
        del obs, info, action
        if signals.get("door_opened", 0) > 0:
            return True, "door_opened"
        return False, None


def make_reward(**kwargs):
    return CollectKeyReward(**kwargs)
