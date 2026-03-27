from __future__ import annotations
from .base_env import ZeldaEnv
from .reward import estimate_room58_turtles

class Room58_Task2_Env(ZeldaEnv):
    def __init__(
        self,
        game_file: str,
        save_file: str,
        render_mode: str | None = None,
        goal_room: int | None = 58,
        **kwargs,
    ):
        super().__init__(
            game_file,
            save_file,
            goal_room=goal_room,
            render_mode=render_mode,
            **kwargs,
        )
        self.turtles = self._get_monsters()

    def _reset_extra(self, options=None):
        self.turtles = self._get_monsters()

    def check_goal(self) -> bool:
        _, y = self._get_pos()
        if self.cur_room != 58 or y == -16:
            return False
        else:
            return self.turtles == 0

    def calculate_reward(self) -> tuple[float, bool]:
        reward = 0.0
        terminated = False
        _, y = self._get_pos()
        if self.is_dead():
            reward -= 1.0
            return reward, True

        reward += 0.01 * self.is_hurt()

        if y != -16:
            reward += self._monster_kill_bonus()

        if self.check_goal():
            reward += 100.0
            terminated = True
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.1

        if self.outside_counter_tick(max_out=100):
            reward -= 1
            terminated = True

        return reward, terminated

    def _get_monsters(self):
        return estimate_room58_turtles(self.emulator.game_area())

    def _monster_kill_bonus(self) -> float:
        cur_turtles = self._get_monsters()
        bonus = 0.0
        if cur_turtles < self.turtles:
            bonus += 10.0
        self.turtles = cur_turtles
        return bonus