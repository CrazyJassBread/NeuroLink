from __future__ import annotations

import importlib
import warnings
from typing import Any

import gymnasium as gym
import numpy as np

from rl.config.schema import EnvironmentConfig

from .base import build_render_mode


def _load_envpool_module():
    try:
        return importlib.import_module("envpool")
    except Exception as exc:  # noqa: BLE001 - surface the backend failure with context.
        raise ImportError(
            "Failed to import envpool. On this repository's current macOS setup, "
            "the installed envpool wheel may fail at import time because optional "
            "backends such as procgen pull in missing native libraries. Install a "
            "working envpool build and the `minigrid` package before running this wrapper."
        ) from exc


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    return image.astype(np.float32).reshape(-1) / 255.0


def _normalize_direction(direction: np.ndarray | int | None) -> np.ndarray:
    if direction is None:
        return np.empty((0,), dtype=np.float32)
    direction_arr = np.asarray(direction)
    if direction_arr.ndim > 0:
        direction_arr = direction_arr[0]
    return np.asarray([float(direction_arr) / 3.0], dtype=np.float32)


def _unbatch_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _unbatch_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_unbatch_value(item) for item in value)
    if isinstance(value, list):
        return [_unbatch_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        head = value[0]
        if isinstance(head, np.ndarray) and head.ndim == 0:
            return head.item()
        return head
    return value


class MiniGridEnvPoolAdapter(gym.Env[np.ndarray, int]):
    """Adapt EnvPool's batched MiniGrid API to a single-env Gymnasium API."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        *,
        task_id: str = "MiniGrid-Empty-5x5-v0",
        seed: int = 0,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        envpool_env: Any | None = None,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self._seed = seed
        self.render_mode = render_mode
        self._needs_reset = True
        self._env = envpool_env or self._build_envpool_env(
            task_id=task_id,
            seed=seed,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )
        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._infer_observation_shape(),
            dtype=np.float32,
        )

    def _build_envpool_env(
        self,
        *,
        task_id: str,
        seed: int,
        max_episode_steps: int | None,
        render_mode: str | None,
    ):
        envpool = _load_envpool_module()
        kwargs: dict[str, Any] = {
            "num_envs": 1,
            "seed": seed,
        }
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        return envpool.make_gymnasium(task_id, **kwargs)

    def _infer_observation_shape(self) -> tuple[int]:
        image_space = self._env.observation_space["image"]
        flat_image = int(np.prod(image_space.shape))
        direction_dim = 1 if "direction" in self._env.observation_space.spaces else 0
        return (flat_image + direction_dim,)

    def _transform_observation(self, obs: dict[str, Any]) -> np.ndarray:
        pieces = [_normalize_image(obs["image"])]
        if "direction" in obs:
            pieces.append(_normalize_direction(obs["direction"]))
        return np.concatenate(pieces, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("MiniGridEnvPoolAdapter does not support Gymnasium reset options.")
        if seed is not None and seed != self._seed:
            warnings.warn(
                "EnvPool seeds are fixed at construction time. "
                "reset(seed=...) is ignored; create a new adapter with the requested seed.",
                stacklevel=2,
            )
        obs, info = self._env.reset()
        self._needs_reset = False
        return self._transform_observation(obs), _unbatch_value(info)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("Environment reached a terminal state; call reset() before step().")
        batched_action = np.asarray([action], dtype=np.int64)
        obs, reward, terminated, truncated, info = self._env.step(batched_action)
        terminated_flag = bool(np.asarray(terminated)[0])
        truncated_flag = bool(np.asarray(truncated)[0])
        self._needs_reset = terminated_flag or truncated_flag
        return (
            self._transform_observation(obs),
            float(np.asarray(reward)[0]),
            terminated_flag,
            truncated_flag,
            _unbatch_value(info),
        )

    def render(self) -> np.ndarray | None:
        frame = self._env.render()
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.ndim == 4:
            return frame[0]
        return frame

    def close(self) -> None:
        self._env.close()


def make_minigrid_env(
    *,
    task_id: str = "MiniGrid-Empty-5x5-v0",
    seed: int = 0,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
) -> MiniGridEnvPoolAdapter:
    return MiniGridEnvPoolAdapter(
        task_id=task_id,
        seed=seed,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )


def build_minigrid_env(
    config: EnvironmentConfig,
    *,
    seed: int,
    render: bool | None = None,
) -> MiniGridEnvPoolAdapter:
    render_enabled = config.render if render is None else render
    task_id = config.task_id or "MiniGrid-Empty-5x5-v0"
    return make_minigrid_env(
        task_id=task_id,
        seed=seed,
        max_episode_steps=config.max_episode_steps,
        render_mode=build_render_mode(render_enabled),
    )
