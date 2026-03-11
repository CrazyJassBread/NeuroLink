from __future__ import annotations
from typing import Tuple
import numpy as np
from .base_env import BaseEnv, ADDR_KEYS

class Room58_Task_Env(BaseEnv):
    """
    58 Task :
    - Goal 1: Pick up the key in room 58 and exit the room
    - Goal 2: kill the turtle monsters in room 58 (optional)
    """
    def __init__(self, game_file: str, save_file: str, render_mode: str | None = None):
        super().__init__(game_file, save_file, render_mode=render_mode)
        
        self.target_pos = (30,45) # the key position in room 58
        self.goal_room = 58 
        self.pre_distance = None
        self.cur_distance = None

        self.pre_turtles = self.get_monsters()
        self.cur_turtles = self.pre_turtles

    def _reset_extra(self, options=None):
        self.pre_distance = None
        self.cur_distance = None
        self.pre_turtles = self.get_monsters()
        self.cur_turtles = self.pre_turtles
    
    def update_after_action(self):
        self.pre_distance = self.cur_distance
        self.pre_turtles = self.cur_turtles

    def update_before_action(self):
        self.cur_distance = self.get_distance()
        self.cur_turtles = self.get_monsters()

    def check_goal(self) -> bool:
        return self.read_m(ADDR_KEYS) >= 1

    def get_distance(self) -> float:
        x, y = self._get_pos()
        target_x, target_y = self.target_pos
        return abs(target_x - x) + abs(target_y - y)
    
    def get_close(self) -> bool:
        if self.cur_distance != None and self.pre_distance != None and self.cur_room == self.goal_room:
            return self.cur_distance < self.pre_distance
        return False
    
    def get_monsters(self) -> int:
        # TODO: get the number of turtle monsters in room 58
        return 0

    def calculate_reward(self) -> Tuple[float, bool]:
        reward = 0.0
        terminated = False

        # the death reward
        if self.is_dead():
            reward -= 1.0
            return reward, True
        # the hurt reward
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