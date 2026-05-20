import functools
import sys
from pathlib import Path

import elements
import embodied
import numpy as np

from .gym_env import DEFAULT_DUNGEON_CONFIG, make_gym_env, seed_action_space


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class NesyLinkEnv(embodied.Env):
    LOG_EVENTS = (
        "chest_opened",
        "key_collected",
        "gold_collected",
        "item_collected",
        "agent_healed",
        "button_pressed",
        "room_changed",
        "door_opened",
        "action_blocked",
        "trap_triggered",
        "monster_damaged",
        "action_shield",
        "shield_block",
        "monster_killed",
        "agent_dead",
        "exit_reached",
    )

    OBS_KEYS = (
        "grid",
        "player_position_px",
        "player_tile",
        "health",
        "gold",
        "keys",
        "inventory_ids",
        "monsters_position_px",
        "monsters_tile",
        "monsters_active_mask",
        "monsters_hp",
    )

    def __init__(
        self,
        task="default",
        config_path=None,
        image=True,
        image_size=(64, 64),
        length=500,
        logs=True,
        seed=None,
        move_speed_px=1,
        agent_noop_enabled=False,
        stuck_penalty_enabled=False,
        stuck_penalty_steps=30,
        stuck_penalty=-0.01,
    ):
        del task
        del stuck_penalty_enabled, stuck_penalty_steps, stuck_penalty
        from nesylink.core.constants import ACTION_NOOP

        self._config_path = self._resolve_config_path(config_path)
        self._env = make_gym_env(
            map_path=self._config_path,
            render_mode="rgb_array",
            auto_reset_on_step=False,
            move_speed_px=move_speed_px,
            max_steps=length,
        )
        self._image = bool(image)
        self._image_size = tuple(image_size)
        self._length = int(length) if length else 0
        self._logs = bool(logs)
        self._seed = seed
        self._agent_noop_enabled = bool(agent_noop_enabled)
        self._noop_action = ACTION_NOOP
        self._last_noop_mapped = False
        self._episode = 0
        self._step = 0
        self._done = True
        self._visited_rooms = set()

        if seed is not None:
            seed_action_space(self._env, seed)

    @functools.cached_property
    def obs_space(self):
        spaces = {
            "vector": elements.Space(np.float32, (self._vector_size(),)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        if self._image:
            spaces["image"] = elements.Space(np.uint8, self._image_size + (3,))
        if self._logs:
            spaces.update(
                {
                    "log/reward_raw": elements.Space(np.float32),
                    "log/discount": elements.Space(np.float32),
                    "log/health": elements.Space(np.float32),
                    "log/gold": elements.Space(np.float32),
                    "log/keys": elements.Space(np.float32),
                    "log/step": elements.Space(np.float32),
                    "log/dungeon_episode": elements.Space(np.float32),
                    "log/room_x": elements.Space(np.float32),
                    "log/room_y": elements.Space(np.float32),
                    "log/visited_rooms": elements.Space(np.float32),
                    "log/success": elements.Space(np.float32),
                    "log/has_key": elements.Space(np.float32),
                    "log/no_progress_steps": elements.Space(np.float32),
                    "log/noop_mapped": elements.Space(np.float32),
                    **{f"log/{event}": elements.Space(np.float32) for event in self.LOG_EVENTS},
                }
            )
        return spaces

    @functools.cached_property
    def act_space(self):
        action_count = self._env.action_space.n if self._agent_noop_enabled else self._env.action_space.n - 1
        return {
            "action": elements.Space(np.int32, (), 0, action_count),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if bool(action["reset"]) or self._done:
            return self._reset()

        raw_action = self._agent_action(int(np.asarray(action["action"]).item()))
        obs, reward, terminated, truncated, info = self._env.step(raw_action)
        self._step += 1

        time_limit = bool(self._length and self._step >= self._length)
        is_terminal = bool(terminated)
        is_last = bool(terminated or truncated or time_limit)
        self._done = is_last
        return self._obs(
            obs,
            reward,
            info,
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def close(self):
        self._env.close()

    def _reset(self):
        seed = None if self._seed is None else self._seed + self._episode
        obs, info = self._env.reset(seed=seed)
        self._episode += 1
        self._step = 0
        self._done = False
        self._last_noop_mapped = False
        self._visited_rooms = {info.get("env", {}).get("room_id", "")}
        return self._obs(obs, 0.0, info, is_first=True)

    def _agent_action(self, raw_action):
        if self._agent_noop_enabled:
            self._last_noop_mapped = False
            return raw_action
        self._last_noop_mapped = raw_action == self._noop_action
        return raw_action + 1

    def _obs(
        self,
        obs,
        reward,
        info,
        *,
        is_first=False,
        is_last=False,
        is_terminal=False,
    ):
        self._visited_rooms.add(info.get("env", {}).get("room_id", ""))
        result = {
            "vector": self._vector(obs),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        }
        if self._image:
            result["image"] = self._render_image()
        if self._logs:
            result.update(self._log_obs(info, reward, is_terminal))
        return result

    def _vector(self, obs):
        parts = []
        for key in self.OBS_KEYS:
            value = obs.get(key)
            if value is None:
                continue
            parts.append(np.asarray(value, np.float32).reshape(-1))
        return np.concatenate(parts, 0).astype(np.float32)

    def _vector_size(self):
        size = 0
        for key in self.OBS_KEYS:
            space = self._env.observation_space.spaces.get(key)
            if space is not None:
                size += int(np.prod(space.shape))
        return size

    def _render_image(self):
        image = self._env.render()
        if image.shape[:2] == self._image_size:
            return image.astype(np.uint8)
        from PIL import Image

        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self._image_size[1], self._image_size[0]), Image.NEAREST)
        return np.asarray(pil_image, dtype=np.uint8)

    def _log_obs(self, info, reward, is_terminal):
        event_counts = info.get("events", {}).get("counts", {})
        room_coord = info.get("env", {}).get("room_coord", (0, 0))
        inventory = info.get("inventory", {})
        episode_info = info.get("episode", {})
        success = bool(info.get("reward", {}).get("terminated", False))
        logs = {
            "log/reward_raw": np.float32(reward),
            "log/discount": np.float32(0.0 if is_terminal else 1.0),
            "log/health": np.float32(info.get("agent", {}).get("hp", 0)),
            "log/gold": np.float32(inventory.get("gold", 0)),
            "log/keys": np.float32(inventory.get("keys", 0)),
            "log/step": np.float32(episode_info.get("step_count", self._step)),
            "log/dungeon_episode": np.float32(episode_info.get("id", self._episode)),
            "log/room_x": np.float32(room_coord[0]),
            "log/room_y": np.float32(room_coord[1]),
            "log/visited_rooms": np.float32(len(self._visited_rooms)),
            "log/success": np.float32(success),
            "log/has_key": np.float32(1.0 if inventory.get("keys", 0) > 0 else 0.0),
            "log/no_progress_steps": np.float32(episode_info.get("no_progress_steps", 0)),
            "log/noop_mapped": np.float32(1.0 if self._last_noop_mapped else 0.0),
        }
        logs.update(
            {
                f"log/{event}": np.float32(1.0 if int(event_counts.get(event, 0)) > 0 else 0.0)
                for event in self.LOG_EVENTS
            }
        )
        return logs

    @staticmethod
    def _resolve_config_path(config_path):
        if config_path is None:
            return DEFAULT_DUNGEON_CONFIG
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
