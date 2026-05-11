from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.constants import (
    ACTION_LABELS,
    GRID_HEIGHT,
    GRID_WIDTH,
    ITEM_NAME_TO_ID,
    MAP_PIXEL_HEIGHT,
    MAP_PIXEL_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TARGET_FPS,
)
from ..core.engine import DungeonEngine
from ..core.info import build_info
from ..core.observation import build_observation
from ..core.types import RewardTerms, StuckPenaltyConfig, TaskValidationResult
from ..entities import tile_from_position_px
from ..rendering.renderer import render_frame
from ..rewards.reward_fn import RewardConfig, compute_reward
from ..tasks.task_spec import TaskSpec
from ..tasks.validators import validate_task
from .registry import register_wrapper


class _BaseDungeonEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": TARGET_FPS}

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = True,
        move_speed_px: float = 1.0,
        stuck_penalty_enabled: bool = False,
        stuck_penalty_steps: int = 30,
        stuck_penalty: float = -0.01,
        action_repeat: int = 1,
        reward_mode: str = "legacy",
        use_validator_termination: bool = False,
    ):
        super().__init__()
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        self.render_mode = render_mode
        self.auto_reset_on_step = bool(auto_reset_on_step)
        self.action_repeat = int(action_repeat)
        self.native_action_repeat = self.action_repeat
        self.use_validator_termination = bool(use_validator_termination)
        self.stuck_penalty_config = StuckPenaltyConfig(
            enabled=bool(stuck_penalty_enabled),
            steps=max(1, int(stuck_penalty_steps)),
            reward=float(stuck_penalty),
        )
        self.reward_config = RewardConfig(
            reward_mode=str(reward_mode),
            stuck_penalty=self.stuck_penalty_config,
        )
        self.engine = DungeonEngine(room_file, move_speed_px=move_speed_px)
        self.task_spec = TaskSpec.from_task_config(self.engine.task_config)

        self.action_space = spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=8, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8),
                "player_position_px": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([MAP_PIXEL_WIDTH - 1.0, MAP_PIXEL_HEIGHT - 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "player_tile": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([GRID_WIDTH - 1, GRID_HEIGHT - 1], dtype=np.int32),
                    dtype=np.int32,
                ),
                "health": spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
                "gold": spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32),
                "keys": spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
                "inventory_ids": spaces.Box(
                    low=0,
                    high=max(ITEM_NAME_TO_ID.values()),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "monsters_position_px": spaces.Box(
                    low=-1.0,
                    high=max(SCREEN_WIDTH, SCREEN_HEIGHT),
                    shape=(self.engine.max_monster_slots, 2),
                    dtype=np.float32,
                ),
                "monsters_tile": spaces.Box(
                    low=-1,
                    high=max(GRID_WIDTH, GRID_HEIGHT),
                    shape=(self.engine.max_monster_slots, 2),
                    dtype=np.int32,
                ),
                "monsters_active_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.engine.max_monster_slots,),
                    dtype=np.uint8,
                ),
                "monsters_hp": spaces.Box(
                    low=0,
                    high=99,
                    shape=(self.engine.max_monster_slots,),
                    dtype=np.int32,
                ),
            }
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.engine.reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info(
            events=["reset"],
            event_details=[],
            reward_terms={},
            validator_result=TaskValidationResult(),
            legacy_terminated=False,
            inner_steps=0,
        )
        return observation, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), "Invalid action!"

        auto_reset = False
        if self.engine.runtime.pending_reset and self.auto_reset_on_step:
            self.reset()
            auto_reset = True
        elif self.engine.runtime.pending_reset:
            raise RuntimeError("Episode terminated. Call reset() before step().")

        total_reward = 0.0
        reward_terms: RewardTerms = {}
        merged_events: list[str] = []
        merged_event_details: list[dict[str, Any]] = []
        legacy_result = None
        inner_steps = 0

        for _ in range(self.action_repeat):
            prev_state = self.engine.runtime.snapshot()
            result = self.engine.step(action)
            next_state = self.engine.runtime.snapshot()
            reward, inner_terms = compute_reward(
                prev_state,
                next_state,
                result,
                task_spec=self.engine.task_config,
                config=self.reward_config,
            )
            total_reward += reward
            _merge_reward_terms(reward_terms, inner_terms)
            merged_events.extend(result.events)
            merged_event_details.extend(result.event_details)
            legacy_result = result
            inner_steps += 1
            if result.terminated or result.truncated:
                break

        assert legacy_result is not None
        validator_result = validate_task(
            self.task_spec,
            self.engine.runtime,
            merged_events,
            merged_event_details,
            legacy_done=legacy_result.terminated,
        )
        terminated = legacy_result.terminated
        if self.use_validator_termination and validator_result.validator_matches_legacy:
            terminated = validator_result.validator_done

        observation = self._get_obs()
        info = self._get_info(
            events=merged_events,
            event_details=merged_event_details,
            reward_terms=reward_terms,
            validator_result=validator_result,
            legacy_terminated=legacy_result.terminated,
            auto_reset=auto_reset,
            inner_steps=inner_steps,
        )
        if not validator_result.validator_matches_legacy:
            info["validator_mismatch"] = True
        return observation, total_reward, terminated, legacy_result.truncated, info

    def render(self) -> np.ndarray:
        return render_frame(self.engine.runtime.room, self.engine.runtime.player)

    def close(self) -> None:
        return None

    def hud_lines(self) -> tuple[str, str]:
        return self.engine.hud_lines()

    def _get_obs(self) -> dict[str, np.ndarray]:
        return build_observation(
            self.engine.runtime.room,
            self.engine.runtime.player,
            self.engine.max_monster_slots,
        )

    def _get_info(
        self,
        *,
        events: list[str],
        event_details: list[dict[str, Any]],
        reward_terms: RewardTerms,
        validator_result: TaskValidationResult,
        legacy_terminated: bool,
        auto_reset: bool = False,
        inner_steps: int = 1,
    ) -> dict[str, Any]:
        task_id = self.engine.task_config.task_id if self.engine.task_config is not None else None
        task_type = self.engine.task_config.task_type if self.engine.task_config is not None else None
        return build_info(
            self.engine.runtime,
            events=events,
            event_details=event_details,
            reward_terms=reward_terms,
            map_id=self.engine.map_id,
            movement_pixels=self.engine.move_speed_px,
            action_repeat=self.action_repeat,
            inner_steps=inner_steps,
            legacy_terminated=legacy_terminated,
            validator_result=validator_result,
            auto_reset=auto_reset,
            task_id=task_id,
            task_type=task_type,
        )

    def _player_tile(self) -> tuple[int, int]:
        player = self.engine.runtime.player
        return tile_from_position_px(player.position_px, player.size_px)


class DungeonEnv(_BaseDungeonEnv):
    """Legacy compatibility wrapper.

    This class preserves old convenience attributes and auto-reset behavior for existing scripts.
    New code should prefer `make_env(api="gym")`, which returns `GymDungeonEnv`.
    """

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = True,
        move_speed_px: float = 1.0,
        stuck_penalty_enabled: bool = False,
        stuck_penalty_steps: int = 30,
        stuck_penalty: float = -0.01,
        action_repeat: int = 1,
        reward_mode: str = "legacy",
        use_validator_termination: bool = False,
    ):
        super().__init__(
            room_file,
            render_mode=render_mode,
            auto_reset_on_step=auto_reset_on_step,
            move_speed_px=move_speed_px,
            stuck_penalty_enabled=stuck_penalty_enabled,
            stuck_penalty_steps=stuck_penalty_steps,
            stuck_penalty=stuck_penalty,
            action_repeat=action_repeat,
            reward_mode=reward_mode,
            use_validator_termination=use_validator_termination,
        )

    @property
    def room_manager(self):
        return self.engine.room_manager

    @property
    def task_config(self):
        return self.engine.task_config

    @property
    def max_monster_slots(self) -> int:
        return self.engine.max_monster_slots

    @property
    def room_coord(self) -> tuple[int, int]:
        return self.engine.runtime.room_coord

    @room_coord.setter
    def room_coord(self, value: tuple[int, int]) -> None:
        self.engine.runtime.room_coord = value

    @property
    def room(self):
        return self.engine.runtime.room

    @room.setter
    def room(self, value) -> None:
        self.engine.runtime.room = value

    @property
    def player(self):
        return self.engine.runtime.player

    @player.setter
    def player(self, value) -> None:
        self.engine.runtime.player = value

    @property
    def episode(self) -> int:
        return self.engine.runtime.episode

    @property
    def step_count(self) -> int:
        return self.engine.runtime.step_count

    @property
    def pending_reset(self) -> bool:
        return self.engine.runtime.pending_reset

    @property
    def last_message(self) -> str:
        return self.engine.runtime.last_message


class GymDungeonEnv(DungeonEnv):
    """Canonical Gymnasium wrapper.

    The legacy convenience attributes still exist during the current compatibility window, but
    new code should treat this as a standard Gym env and rely on `reset/step/render/close`.
    """

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = False,
        move_speed_px: float = 1.0,
        stuck_penalty_enabled: bool = False,
        stuck_penalty_steps: int = 30,
        stuck_penalty: float = -0.01,
        action_repeat: int = 1,
        reward_mode: str = "legacy",
        use_validator_termination: bool = True,
    ):
        super().__init__(
            room_file,
            render_mode=render_mode,
            auto_reset_on_step=auto_reset_on_step,
            move_speed_px=move_speed_px,
            stuck_penalty_enabled=stuck_penalty_enabled,
            stuck_penalty_steps=stuck_penalty_steps,
            stuck_penalty=stuck_penalty,
            action_repeat=action_repeat,
            reward_mode=reward_mode,
            use_validator_termination=use_validator_termination,
        )


def _merge_reward_terms(total: RewardTerms, incoming: RewardTerms) -> None:
    for name, value in incoming.items():
        total[name] = total.get(name, 0.0) + float(value)


register_wrapper("gym", GymDungeonEnv)
