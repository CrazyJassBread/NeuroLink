from __future__ import annotations
from .base_env import ZeldaEnv
from .config import ADDR_KEYS
from .reward import manhattan_distance

class Room51_Task1_Env(ZeldaEnv):
    """Room 51 task: press switch, open chest, pick key."""

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
        self.flag = False
        self.pre_distance1 = self.get_distance(80, 45)
        self.pre_distance2 = self.get_distance(130, 40)

    def _reset_extra(self, options=None):
        self.flag = False

    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    def get_distance(self, target_x: int | None = None, target_y: int | None = None) -> float:
        x, y = self._get_pos()
        if target_x is not None and target_y is not None:
            return float(manhattan_distance(x, y, target_x, target_y))
        
        if not self.flag:
            target_x, target_y = 80, 45
        else:
            target_x, target_y = 130, 40
        return float(manhattan_distance(x, y, target_x, target_y))

    def update_flag(self) -> bool:
        x, y = self._get_pos()
        if not self.flag and abs(x - 80) < 3 and abs(y - 45) < 3:
            self.flag = True
        return self.flag

    def calculate_reward(self) -> tuple[float, bool]:
        reward = 0.0
        terminated = False

        if self.is_dead():
            reward -= 1.0
            return reward, True

        reward += 0.01 * self.is_hurt()

        if not self.flag and self.update_flag():
            reward += 5.0

        if self.tile_explore_bonus():
            reward += 0.01

        if self.check_goal():
            reward += 10.0
            terminated = True

        if not self.flag:
            reward -= (self.get_distance() / max(self.pre_distance1, 1e-6)) * 0.01
        else:
            reward -= (self.get_distance() / max(self.pre_distance2, 1e-6)) * 0.01

        if self.outside_counter_tick(max_out=100):
            reward -= 0.1
            terminated = True

        return reward, terminated