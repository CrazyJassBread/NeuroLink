import functools
import sys
from pathlib import Path

import elements
import embodied
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / 'env_diy' / 'map_data' / 'dungeons' / 'prototype' / 'dungeon.json'

if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))


class EnvDIY(embodied.Env):

  LOG_EVENTS = (
      'opened_chest',
      'got_key',
      'got_gold',
      'got_item',
      'healed',
      'pressed_button',
      'room_transition',
      'door_unlocked',
      'blocked_locked',
      'missing_requirement',
      'trap_damage',
      'monster_hit',
      'monster_damaged',
      'shield_block',
      'monster_killed',
      'game_over',
      'victory',
  )

  OBS_KEYS = (
      'grid',
      'player_position_px',
      'player_tile',
      'health',
      'gold',
      'keys',
      'inventory_ids',
      'monsters_position_px',
      'monsters_tile',
      'monsters_active_mask',
      'monsters_hp',
  )

  def __init__(
      self,
      task='default',
      config_path=None,
      image=True,
      image_size=(64, 64),
      length=500,
      logs=True,
      seed=None,
  ):
    del task
    from env_diy.envs import DungeonEnv

    self._config_path = self._resolve_config_path(config_path)
    self._env = DungeonEnv(
        self._config_path,
        render_mode='rgb_array',
        auto_reset_on_step=False)
    self._image = bool(image)
    self._image_size = tuple(image_size)
    self._length = int(length) if length else 0
    self._logs = bool(logs)
    self._seed = seed
    self._episode = 0
    self._step = 0
    self._done = True
    self._visited_rooms = set()

    if seed is not None:
      self._env.action_space.seed(seed)

  @functools.cached_property
  def obs_space(self):
    spaces = {
        'vector': elements.Space(np.float32, (self._vector_size(),)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self._image:
      spaces['image'] = elements.Space(np.uint8, self._image_size + (3,))
    if self._logs:
      spaces.update({
          'log/reward_raw': elements.Space(np.float32),
          'log/discount': elements.Space(np.float32),
          'log/health': elements.Space(np.float32),
          'log/gold': elements.Space(np.float32),
          'log/keys': elements.Space(np.float32),
          'log/step': elements.Space(np.float32),
          'log/dungeon_episode': elements.Space(np.float32),
          'log/room_x': elements.Space(np.float32),
          'log/room_y': elements.Space(np.float32),
          'log/visited_rooms': elements.Space(np.float32),
          'log/success': elements.Space(np.float32),
          **{f'log/{event}': elements.Space(np.float32) for event in self.LOG_EVENTS},
      })
    return spaces

  @functools.cached_property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if bool(action['reset']) or self._done:
      return self._reset()

    raw_action = int(np.asarray(action['action']).item())
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
        is_terminal=is_terminal)

  def close(self):
    self._env.close()

  def _reset(self):
    seed = None if self._seed is None else self._seed + self._episode
    obs, info = self._env.reset(seed=seed)
    self._episode += 1
    self._step = 0
    self._done = False
    self._visited_rooms = {info.get('room_id', '')}
    return self._obs(obs, 0.0, info, is_first=True)

  def _obs(
      self,
      obs,
      reward,
      info,
      *,
      is_first=False,
      is_last=False,
      is_terminal=False):
    self._visited_rooms.add(info.get('room_id', ''))
    result = {
        'vector': self._vector(obs),
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
    }
    if self._image:
      result['image'] = self._render_image()
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
    # PIL expects size as (width, height), while config uses (height, width).
    pil_image = pil_image.resize((self._image_size[1], self._image_size[0]), Image.NEAREST)
    return np.asarray(pil_image, dtype=np.uint8)

  def _log_obs(self, info, reward, is_terminal):
    events = set(info.get('events', ()))
    room_coord = info.get('room_coord', (0, 0))
    success = bool(info.get('victory', False))
    logs = {
        'log/reward_raw': np.float32(reward),
        'log/discount': np.float32(0.0 if is_terminal else 1.0),
        'log/health': np.float32(info.get('health', 0)),
        'log/gold': np.float32(info.get('gold', 0)),
        'log/keys': np.float32(info.get('keys', 0)),
        'log/step': np.float32(info.get('step', self._step)),
        'log/dungeon_episode': np.float32(info.get('episode', self._episode)),
        'log/room_x': np.float32(room_coord[0]),
        'log/room_y': np.float32(room_coord[1]),
        'log/visited_rooms': np.float32(len(self._visited_rooms)),
        'log/success': np.float32(success),
    }
    logs.update({
        f'log/{event}': np.float32(1.0 if event in events else 0.0)
        for event in self.LOG_EVENTS
    })
    return logs

  @staticmethod
  def _resolve_config_path(config_path):
    if config_path is None:
      return DEFAULT_CONFIG
    path = Path(config_path)
    if not path.is_absolute():
      path = PROJECT_ROOT / path
    return path
