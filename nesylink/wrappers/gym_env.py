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
from ..core.mechanics.engine import DungeonEngine
from ..core.info import build_info
from ..core.observation import build_observation
from ..core.rendering import render_frame
from ..core.state import tile_from_position_px
from ..core.world.loader import load_map
from ..rewards.loader import load_reward
from .registry import register_wrapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUNGEON_CONFIG = PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "prototype" / "dungeon.json"


class DefaultSeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._default_seed = int(seed)

    def reset(self, *, seed: int | None = None, options=None):
        return self.env.reset(
            seed=self._default_seed if seed is None else seed,
            options=options,
        )


def resolve_config_path(
    config_path: str | Path | None = None,
    *,
    default_config: Path = DEFAULT_DUNGEON_CONFIG,
) -> Path:
    dungeon_config = Path(config_path) if config_path is not None else default_config
    if not dungeon_config.is_absolute():
        dungeon_config = PROJECT_ROOT / dungeon_config
    return dungeon_config


def make_gym_env(
    config_path: str | Path | None = None,
    *,
    map_id: str | None = None,
    map_path: str | Path | None = None,
    reward_id: str | None = None,
    reward_module: str | None = None,
    reward_kwargs: dict[str, float] | None = None,
    render_mode: str | None = None,
    action_repeat: int = 1,
    max_steps: int | None = None,
    auto_reset_on_step: bool = False,
    move_speed_px: float = 1.0,
    **kwargs: Any,
) -> gym.Env:
    if action_repeat < 1:
        raise ValueError("action_repeat must be >= 1")

    resolved_path = load_map(
        map_id=map_id,
        map_path=(
            map_path
            if map_path is not None
            else resolve_config_path(config_path)
            if config_path is not None
            else None
        ),
    )
    reward_fn = load_reward(
        reward_id=reward_id,
        reward_module=reward_module,
        reward_kwargs=reward_kwargs,
    )
    return GymDungeonEnv(
        resolved_path,
        reward_fn=reward_fn,
        render_mode=render_mode,
        action_repeat=action_repeat,
        max_steps=max_steps,
        auto_reset_on_step=auto_reset_on_step,
        move_speed_px=move_speed_px,
        **kwargs,
    )


def seed_action_space(env: gym.Env, seed: int) -> gym.Env:
    env.action_space.seed(seed)
    return env


def with_default_seed(env: gym.Env, seed: int) -> gym.Env:
    wrapped = DefaultSeedWrapper(env, seed)
    return seed_action_space(wrapped, seed)


class BaseGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": TARGET_FPS}

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = True,
        move_speed_px: float = 1.0,
        action_repeat: int = 1,
        reward_fn: Any | None = None,
        max_steps: int | None = None,
        **_: Any,
    ):
        super().__init__()
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        self.render_mode = render_mode
        self.auto_reset_on_step = bool(auto_reset_on_step)
        self.action_repeat = int(action_repeat)
        self.native_action_repeat = self.action_repeat
        self.reward_fn = reward_fn
        self.max_steps = None if max_steps is None else int(max_steps)
        self.last_reward_info: dict[str, Any] = {}
        self.engine = DungeonEngine(room_file, move_speed_px=move_speed_px)

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
            events=[],
            event_details=[],
            engine_terminated=False,
            terminal_reason=None,
            inner_steps=0,
            debug_message=None,
        )
        if self.reward_fn is not None:
            self.reward_fn.reset(observation, info)
            info["reward"] = self.reward_fn.build_reward_info(signals={})
            self.last_reward_info = dict(info["reward"])
        return observation, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), "Invalid action!"

        if self.engine.runtime.pending_reset and self.auto_reset_on_step:
            self.reset()
        elif self.engine.runtime.pending_reset:
            raise RuntimeError("Episode terminated. Call reset() before step().")

        merged_events: list[str] = []
        merged_event_details: list[dict[str, Any]] = []
        engine_result = None
        inner_steps = 0

        for _ in range(self.action_repeat):
            result = self.engine.step(action)
            merged_events.extend(result.events)
            merged_event_details.extend(result.event_details)
            engine_result = result
            inner_steps += 1
            if result.terminated or result.truncated:
                break

        assert engine_result is not None

        observation = self._get_obs()
        info = self._get_info(
            events=merged_events,
            event_details=merged_event_details,
            engine_terminated=engine_result.terminated,
            terminal_reason=engine_result.terminated_reason,
            inner_steps=inner_steps,
        )
        if self.reward_fn is not None:
            reward, reward_info = self.reward_fn(observation, info, action)
        else:
            reward = 0.0
            reward_info = {
                "reward_name": "base",
                "reward_signals": {},
                "reward_weights": {},
                "terminated": False,
                "terminated_reason": None,
            }

        terminated = bool(engine_result.terminated or reward_info.get("terminated", False))
        if terminated and not self.engine.runtime.pending_reset:
            self.engine.runtime.pending_reset = True
        if info.get("terminal_reason") is None and reward_info.get("terminated_reason") is not None:
            info["terminal_reason"] = reward_info["terminated_reason"]

        truncated = bool(
            not terminated
            and self.max_steps is not None
            and info["episode"]["step_count"] >= self.max_steps
        )
        info["reward"] = reward_info
        self.last_reward_info = dict(reward_info)
        return observation, float(reward), terminated, truncated, info

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
        engine_terminated: bool,
        terminal_reason: str | None,
        inner_steps: int = 1,
        debug_message: str | None | object = ...,
    ) -> dict[str, Any]:
        return build_info(
            self.engine.runtime,
            events=events,
            event_details=event_details,
            map_id=self.engine.map_id,
            movement_pixels=self.engine.move_speed_px,
            action_repeat=self.action_repeat,
            inner_steps=inner_steps,
            engine_terminated=engine_terminated,
            terminal_reason=terminal_reason,
            debug_message=debug_message,
        )

    def _player_tile(self) -> tuple[int, int]:
        player = self.engine.runtime.player
        return tile_from_position_px(player.position_px, player.size_px)


class DungeonEnv(BaseGameEnv):
    """Compatibility wrapper.

    This class preserves convenience attributes and auto-reset behavior for existing scripts.
    New code should prefer `make_env(api="gym")`, which returns `GymDungeonEnv`.
    """

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = True,
        move_speed_px: float = 1.0,
        action_repeat: int = 1,
        reward_fn: Any | None = None,
        max_steps: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            room_file,
            render_mode=render_mode,
            auto_reset_on_step=auto_reset_on_step,
            move_speed_px=move_speed_px,
            action_repeat=action_repeat,
            reward_fn=reward_fn,
            max_steps=max_steps,
            **kwargs,
        )

    @property
    def room_manager(self):
        return self.engine.room_manager

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

    The compatibility convenience attributes still exist during the current transition window, but
    new code should treat this as a standard Gym env and rely on `reset/step/render/close`.
    """

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = False,
        move_speed_px: float = 1.0,
        action_repeat: int = 1,
        reward_fn: Any | None = None,
        max_steps: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            room_file,
            render_mode=render_mode,
            auto_reset_on_step=auto_reset_on_step,
            move_speed_px=move_speed_px,
            action_repeat=action_repeat,
            reward_fn=reward_fn,
            max_steps=max_steps,
            **kwargs,
        )


register_wrapper("gym", GymDungeonEnv)
