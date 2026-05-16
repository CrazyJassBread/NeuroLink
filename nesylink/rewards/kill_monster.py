from __future__ import annotations

from .base import BaseReward


class KillMonsterReward(BaseReward):
    reward_name = "kill_monster"
    reward_weights = {
        "step": -0.01,
        "monster_hit": 1.0,
        "monster_kill": 10.0,
        "hp_loss": -2.0,
        "death": -20.0,
        "exit_reached": 5.0,
    }

    def check_termination(self, signals, obs, info, action=None):
        del obs, info, action
        if signals.get("prev_monsters_remaining", 0) > 0 and signals.get("monsters_remaining", 0) == 0:
            return True, "all_monsters_defeated"
        return False, None


def make_reward(**kwargs):
    return KillMonsterReward(**kwargs)
