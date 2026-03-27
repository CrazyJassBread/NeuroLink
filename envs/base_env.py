from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .config import (
    ADDR_CUR_HEALTH,
    ADDR_KEYS,
    ADDR_MAX_HEALTH,
    ADDR_ROOM_ID,
    ADDR_RUPEE,
    DEFAULT_ACTIONS,
    DEFAULT_RELEASE_ACTIONS,
    ObservationConfig,
    ZeldaEnvConfig,
    resolve_device,
)
from .emulator import EmulatorController
from .observation import ObservationProcessor
from .reward import RewardTracker


class ZeldaEnv(gym.Env, ABC):
    """Gymnasium-compatible Zelda environment composed of modular subsystems."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        game_file: str,
        save_file: str,
        goal_room: int | None = None,
        render_mode: str | None = None,
        device: torch.device | str | None = None,
        env_config: ZeldaEnvConfig | None = None,
        observation_config: ObservationConfig | None = None,
    ):
        super().__init__()

        self.device = resolve_device(device)
        self.game_file = game_file
        self.save_file = save_file
        self.render_mode = render_mode
        self.goal_room = goal_room
        self.cur_step = 0
        self.episode = 0

        self.env_config = env_config or ZeldaEnvConfig()
        self.obs_config = observation_config or self.env_config.observation

        self.emulator = EmulatorController(
            game_file=game_file,
            save_file=save_file,
            render_mode=render_mode,
        )
        self.reward_tracker = RewardTracker()
        self.observation_processor = ObservationProcessor(self.obs_config, self.device)

        self.valid_actions = list(DEFAULT_ACTIONS)
        self.release_actions = list(DEFAULT_RELEASE_ACTIONS)
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = self.observation_processor.observation_space()

        self.max_health = self.read_m(ADDR_MAX_HEALTH)
        self.pre_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_health = self.pre_health
        self.pre_rupee = self.read_m(ADDR_RUPEE)
        self.cur_rupee = self.pre_rupee
        self.cur_room = self.read_m(ADDR_ROOM_ID)

        self.reward_tracker.reset(self.cur_room, self.cur_rupee)

    @property
    def pyboy(self):
        """Compatibility accessor for existing task env code."""
        return self.emulator.pyboy

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.cur_step = 0
        self.episode += 1

        self.emulator.load_state(self.save_file)
        self.emulator.tick(1)

        self.cur_room = self.read_m(ADDR_ROOM_ID)
        self.pre_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_health = self.pre_health
        self.pre_rupee = self.read_m(ADDR_RUPEE)
        self.cur_rupee = self.pre_rupee

        self.reward_tracker.reset(self.cur_room, self.cur_rupee)
        self._reset_extra(options)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _reset_extra(self, options: dict[str, Any] | None):
        pass

    def run_action(self, action: int):
        self.emulator.send_press_release(
            self.valid_actions[action],
            self.release_actions[action],
            self.env_config.press_ticks,
            self.env_config.release_ticks,
        )

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action!"
        self.cur_step += 1

        self.pre_health = self.cur_health
        self.pre_rupee = self.cur_rupee

        self.update_before_action()
        self.run_action(action)
        self.update_after_action()

        self.cur_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_room = self.read_m(ADDR_ROOM_ID)
        self.cur_rupee = self.read_m(ADDR_RUPEE)

        reward, terminated = self.calculate_reward()
        truncated = self.cur_step >= self.env_config.max_steps

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def update_before_action(self):
        pass

    def update_after_action(self):
        pass

    def render(self, mode: str | None = None):
        return self.emulator.render(mode or self.render_mode)

    def close(self):
        self.emulator.close()

    def _get_obs(self):
        return self.observation_processor.process(self.emulator.screen())

    def _get_info(self):
        return {
            "goal": bool(self.check_goal()),
            "room": int(self.cur_room),
            "episode": int(self.episode),
            "step": int(self.cur_step),
        }

    @abstractmethod
    def check_goal(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self) -> tuple[float, bool]:
        raise NotImplementedError

    def _get_pos(self) -> tuple[int, int]:
        return self.emulator.sprite_pos(2)

    def is_dead(self) -> bool:
        return self.reward_tracker.is_dead(self.cur_health)

    def is_hurt(self) -> int:
        return self.reward_tracker.health_delta(self.pre_health, self.cur_health)

    def outside_counter_tick(self, max_out: int = 100) -> bool:
        return self.reward_tracker.outside_counter_tick(
            cur_room=self.cur_room,
            goal_room=self.goal_room,
            max_out=max_out,
        )

    def rupee_gained(self) -> bool:
        return self.reward_tracker.rupee_gained(self.cur_rupee)

    def tile_explore_bonus(self) -> bool:
        x, y = self._get_pos()
        return self.reward_tracker.tile_explore_bonus(self.cur_room, x, y)

    def read_m(self, addr: int) -> int:
        return self.emulator.read_memory(addr)


# Backward-compatible aliases for existing imports.
BaseEnv = ZeldaEnv
MAX_STEPS = ZeldaEnvConfig().max_steps