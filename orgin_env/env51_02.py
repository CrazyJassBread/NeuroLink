from __future__ import annotations
from .base_env import ZeldaEnv
from .config import ADDR_KEYS
from .reward import MonsterStats, estimate_room51_monsters, manhattan_distance, monster_kill_bonus

class Room51_Task1_Combat_Env(ZeldaEnv):
    """Room 51 combat-flavored key task with monster and exploration shaping."""

    def __init__(
        self,
        game_file: str,
        save_file: str,
        render_mode: str | None = None,
        goal_room: int | None = 51,
        **kwargs,
    ):
        super().__init__(
            game_file,
            save_file,
            goal_room=goal_room,
            render_mode=render_mode,
            **kwargs,
        )
        self.monsters = self._get_monsters()

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.monsters = self._get_monsters()
        return obs, info

    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    def get_distance(self) -> float:
        x, y = self._get_pos()
        target_x, target_y = 34, 45
        return float(manhattan_distance(x, y, target_x, target_y))

    def calculate_reward(self) -> tuple[float, bool]:
        reward = 0.0
        terminated = False

        if self.is_dead():
            reward -= 1.0
            return reward, True

        reward += 0.01 * self.is_hurt()
        reward += self._monster_kill_bonus()

        if self.rupee_gained():
            reward += 0.5

        if self.tile_explore_bonus():
            reward += 0.001

        if self.check_goal():
            reward += 10.0
            terminated = True
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.0001
            else:
                reward -= 0.0001 * self.get_distance()

        if self.outside_counter_tick(max_out=100):
            reward -= 0.1
            terminated = True

        return reward, terminated

    def _get_monsters(self) -> MonsterStats:
        return estimate_room51_monsters(self.emulator.game_area())

    def _monster_kill_bonus(self) -> float:
        current = self._get_monsters()
        bonus, self.monsters = monster_kill_bonus(
            previous=self.monsters,
            current=current,
            turtle_reward=5.0,
            slime_reward=2.0,
        )
        return bonus


# Backward-compatible class name.
Room51_Task1_Env = Room51_Task1_Combat_Env