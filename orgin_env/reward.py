from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonsterStats:
    slimes: int = 0
    turtles: int = 0


class RewardTracker:
    """Shared reward state helpers used by task environments."""

    def __init__(self):
        self.pre_rupee = 0
        self.outside_counter = 0
        self.visited_tiles: set[tuple[int, int, int]] = set()

    def reset(self, room: int, rupee: int) -> None:
        self.pre_rupee = int(rupee)
        self.outside_counter = 0
        self.visited_tiles.clear()
        self.visited_tiles.add((int(room), -1, -1))

    def is_dead(self, cur_health: int) -> bool:
        return int(cur_health) == 0

    def health_delta(self, pre_health: int, cur_health: int) -> int:
        if not isinstance(pre_health, (int, float)) or not isinstance(cur_health, (int, float)):
            return 0
        if cur_health < pre_health:
            return int(cur_health - pre_health)
        return 0

    def rupee_gained(self, cur_rupee: int) -> bool:
        gained = int(cur_rupee) > self.pre_rupee
        if gained:
            self.pre_rupee = int(cur_rupee)
        return gained

    def outside_counter_tick(self, cur_room: int, goal_room: int | None, max_out: int = 100) -> bool:
        if goal_room is None:
            return False
        if cur_room != goal_room:
            self.outside_counter += 1
        else:
            self.outside_counter = 0
        if self.outside_counter >= max_out:
            self.outside_counter = 0
            return True
        return False

    def tile_explore_bonus(self, room: int, x: int, y: int) -> bool:
        key = (int(room), int(x), int(y))
        if key in self.visited_tiles:
            return False
        self.visited_tiles.add(key)
        return True


def manhattan_distance(x: int, y: int, target_x: int, target_y: int) -> int:
    return abs(int(target_x) - int(x)) + abs(int(target_y) - int(y))


def estimate_room51_monsters(game_area: np.ndarray) -> MonsterStats:
    sub_area = game_area[:20, :20]
    labels = np.digitize(sub_area, [85, 170])
    count_0 = int(np.count_nonzero(labels == 0))
    count_1 = int(np.count_nonzero(labels == 1))
    return MonsterStats(
        slimes=max((count_1 + 1) // 2, 0),
        turtles=max((count_0 - 1) // 4, 0),
    )


def estimate_room58_turtles(game_area: np.ndarray) -> int:
    sub_area = game_area[:20, :20]
    labels = np.digitize(sub_area, [85, 170])
    count_0 = int(np.count_nonzero(labels == 0))
    return max((count_0 - 1) // 4, 0)


def monster_kill_bonus(
    previous: MonsterStats,
    current: MonsterStats,
    turtle_reward: float,
    slime_reward: float,
) -> tuple[float, MonsterStats]:
    bonus = 0.0
    if current.turtles < previous.turtles:
        bonus += turtle_reward * float(previous.turtles - current.turtles)
    if current.slimes < previous.slimes:
        bonus += slime_reward * float(previous.slimes - current.slimes)
    return bonus, current
