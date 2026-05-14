from __future__ import annotations

from .base import BaseReward


class SparseExitReward(BaseReward):
    reward_name = "sparse_exit"
    reward_weights = {
        "step": 0.0,
        "exit_reached": 1.0,
        "death": 0.0,
    }

    def check_termination(self, signals, obs, info, action=None):
        del obs, info, action
        if signals.get("exit_reached", 0) > 0:
            return True, "exit_reached"
        return False, None


def make_reward(**kwargs):
    return SparseExitReward(**kwargs)
