from __future__ import annotations

import warnings
from pathlib import Path

import gymnasium as gym

from env_diy.env import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"


class _CheckerInfoAdapter(gym.Wrapper):
    """Strip monotonic diagnostic fields that are intentionally non-deterministic across resets.

    Gymnasium's determinism check compares info dicts across fresh resets. We keep episode counters
    in normal runtime info, but remove them only for checker parity.
    """

    INFO_KEYS_TO_DROP = {"episode_id", "episode"}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, self._normalized_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._normalized_info(info)

    def _normalized_info(self, info):
        return {key: value for key, value in info.items() if key not in self.INFO_KEYS_TO_DROP}


def main() -> int:
    env = _CheckerInfoAdapter(make_env(DEFAULT_CONFIG, api="gym"))
    try:
        from gymnasium.utils.env_checker import check_env as gym_check_env

        gym_check_env(env)
    finally:
        env.close()

    try:
        from stable_baselines3.common.env_checker import check_env as sb3_check_env
    except ImportError:
        warnings.warn("stable_baselines3 is not installed; skipping SB3 env checker", stacklevel=1)
        return 0

    env = _CheckerInfoAdapter(make_env(DEFAULT_CONFIG, api="gym"))
    try:
        sb3_check_env(env, warn=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
