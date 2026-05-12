from __future__ import annotations

import importlib
import unittest
import warnings
from pathlib import Path

from env_diy.env import DungeonEnv as LegacyDungeonEnv
from env_diy.wrappers import GymDungeonEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DUNGEON = (
    PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"
)


class EnvDIYAPITests(unittest.TestCase):
    def test_make_env_returns_gym_wrapper(self) -> None:
        from env_diy.env import make_env

        env = make_env(STRUCTURED_DUNGEON, api="gym")
        try:
            self.assertIsInstance(env, LegacyDungeonEnv)
            self.assertIsInstance(env, GymDungeonEnv)
            obs, info = env.reset(seed=0)
            self.assertIn("grid", obs)
            self.assertIsInstance(info, dict)
        finally:
            env.close()

    def test_make_env_uses_non_autoreset_gym_default(self) -> None:
        from env_diy.env import make_env

        env = make_env(STRUCTURED_DUNGEON, api="gym")
        try:
            self.assertFalse(env.auto_reset_on_step)
        finally:
            env.close()

    def test_legacy_dungeon_env_keeps_autoreset_compatibility(self) -> None:
        env = LegacyDungeonEnv(STRUCTURED_DUNGEON)
        try:
            self.assertTrue(env.auto_reset_on_step)
        finally:
            env.close()

    def test_make_env_rejects_unknown_api(self) -> None:
        from env_diy.env import make_env

        with self.assertRaises(ValueError):
            make_env(STRUCTURED_DUNGEON, api="unknown")

    def test_root_dungeon_env_is_deprecated_compat_export(self) -> None:
        module = importlib.import_module("env_diy")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            root_env_cls = module.DungeonEnv
        self.assertIs(root_env_cls, LegacyDungeonEnv)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_envs_namespace_is_deprecated_compat_export(self) -> None:
        module = importlib.import_module("env_diy.envs")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            legacy_env_cls = module.DungeonEnv
        self.assertIs(legacy_env_cls, LegacyDungeonEnv)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_envs_dungeon_env_module_warns_and_forwards(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            module = importlib.import_module("env_diy.envs.dungeon_env")
        self.assertIs(module.DungeonEnv, LegacyDungeonEnv)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))


if __name__ == "__main__":
    unittest.main()
