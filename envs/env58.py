from __future__ import annotations
from .base_env import ZeldaEnv
from .config import ADDR_KEYS
from .reward import manhattan_distance

class Room58_Task_Env(ZeldaEnv):
    """Room 58 task: approach key and pick it up."""

    def __init__(
        self,
        game_file: str,
        save_file: str,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(game_file, save_file, render_mode=render_mode, **kwargs)

        self.target_pos = (30, 45)
        self.goal_room = 58
        self.pre_distance = None
        self.cur_distance = None

    def _reset_extra(self, options=None):
        self.pre_distance = None
        self.cur_distance = None

    def update_after_action(self):
        self.pre_distance = self.cur_distance

    def update_before_action(self):
        self.cur_distance = self.get_distance()

    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    def get_distance(self) -> float:
        x, y = self._get_pos()
        target_x, target_y = self.target_pos
        return float(manhattan_distance(x, y, target_x, target_y))
    
    def get_close(self) -> bool:
        if self.cur_distance is not None and self.pre_distance is not None and self.cur_room == self.goal_room:
            return self.cur_distance < self.pre_distance
        return False

    def calculate_reward(self) -> tuple[float, bool]:
        reward = 0.0
        terminated = False

        if self.is_dead():
            reward -= 1.0
            return reward, True

        reward += 0.1 * self.is_hurt()

        if self.get_close():
            reward += 0.1

        if self.check_goal():
            reward += 10.0
            terminated = True
        else:
            if self.cur_room != self.goal_room:
                reward -= 0.1

        if self.outside_counter_tick(max_out = 400):
            reward -= 1.0
            terminated = True

        return reward, terminated