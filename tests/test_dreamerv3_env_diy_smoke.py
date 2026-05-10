from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DREAMER_ROOT = PROJECT_ROOT / "world_model" / "dreamerv3"


class _FakeSpace:
    def __init__(self, dtype, shape=(), low=None, high=None):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape) if isinstance(shape, (list, tuple)) else (() if shape == () else (shape,))
        self.low = low
        self.high = high
        self.discrete = np.issubdtype(self.dtype, np.integer) or self.dtype == np.dtype(bool)

    def sample(self):
        if self.discrete:
            high = 2 if self.high is None else int(self.high)
            return np.asarray(0 if high <= 0 else high - 1, self.dtype).reshape(self.shape)
        return np.zeros(self.shape, self.dtype)

    def __contains__(self, value):
        value = np.asarray(value)
        return value.shape == self.shape


class _FakeConfig(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            value = _FakeConfig(value)
            self[name] = value
        return value


class _IdentityWrapper:
    def __init__(self, env, *args, **kwargs):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()


def _install_dreamer_fakes(monkeypatch):
    elements = types.ModuleType("elements")
    elements.Space = _FakeSpace
    elements.Config = _FakeConfig
    elements.Path = Path
    elements.timestamp = lambda: "test"
    elements.print = print
    elements.Flags = lambda *args, **kwargs: None
    elements.Counter = lambda *args, **kwargs: None

    embodied = types.ModuleType("embodied")
    embodied.__path__ = [str(DREAMER_ROOT / "embodied")]

    class Env:
        pass

    wrappers = types.SimpleNamespace(
        NormalizeAction=_IdentityWrapper,
        UnifyDtypes=_IdentityWrapper,
        CheckSpaces=_IdentityWrapper,
        ClipAction=_IdentityWrapper,
    )
    embodied.Env = Env
    embodied.wrappers = wrappers
    embodied.run = types.SimpleNamespace()
    embodied.replay = types.SimpleNamespace()
    embodied.streams = types.SimpleNamespace()
    embodied.RandomAgent = object

    portal = types.ModuleType("portal")
    portal.setup = lambda *args, **kwargs: None

    ruamel = types.ModuleType("ruamel")
    yaml_mod = types.ModuleType("ruamel.yaml")
    yaml_mod.YAML = lambda *args, **kwargs: None
    ruamel.yaml = yaml_mod

    monkeypatch.setitem(sys.modules, "elements", elements)
    monkeypatch.setitem(sys.modules, "embodied", embodied)
    monkeypatch.setitem(sys.modules, "portal", portal)
    monkeypatch.setitem(sys.modules, "ruamel", ruamel)
    monkeypatch.setitem(sys.modules, "ruamel.yaml", yaml_mod)
    monkeypatch.syspath_prepend(str(DREAMER_ROOT))
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))


def _import_env_diy(monkeypatch):
    _install_dreamer_fakes(monkeypatch)
    sys.modules.pop("embodied.envs.env_diy", None)
    return importlib.import_module("embodied.envs.env_diy")


def test_env_diy_reset_and_step_expose_dreamerv3_fields(monkeypatch):
    env_diy = _import_env_diy(monkeypatch)
    env = env_diy.EnvDIY("default", image=True, image_size=(64, 64), length=10, logs=True, seed=0)
    try:
        obs = env.step({"reset": np.array(True), "action": np.array(0, np.int32)})
        assert obs["is_first"]
        assert not obs["is_last"]
        assert not obs["is_terminal"]
        assert obs["reward"].dtype == np.float32
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert obs["vector"].dtype == np.float32
        assert obs["vector"].ndim == 1
        assert "log/health" in obs
        assert "log/reward_raw" in obs

        obs = env.step({"reset": np.array(False), "action": np.array(0, np.int32)})
        assert not obs["is_first"]
        assert obs["reward"].dtype == np.float32
        assert obs["vector"].shape == env.obs_space["vector"].shape
    finally:
        env.close()


def test_env_diy_time_limit_is_non_terminal_last_step(monkeypatch):
    env_diy = _import_env_diy(monkeypatch)
    env = env_diy.EnvDIY("default", image=False, length=2, logs=True, seed=0)
    try:
        obs = env.step({"reset": np.array(True), "action": np.array(0, np.int32)})
        assert obs["is_first"]

        obs = env.step({"reset": np.array(False), "action": np.array(0, np.int32)})
        assert not obs["is_last"]

        obs = env.step({"reset": np.array(False), "action": np.array(0, np.int32)})
        assert obs["is_last"]
        assert not obs["is_terminal"]
        assert obs["log/discount"] == np.float32(1.0)
    finally:
        env.close()


def test_env_diy_training_noop_is_mapped_to_valid_action(monkeypatch):
    env_diy = _import_env_diy(monkeypatch)
    env = env_diy.EnvDIY(
        "default",
        image=False,
        length=10,
        logs=True,
        seed=0,
        move_speed_px=1,
        agent_noop_enabled=False,
    )
    try:
        assert env.act_space["action"].high == 6
        obs = env.step({"reset": np.array(True), "action": np.array(0, np.int32)})
        before = env._env.player.position_px

        obs = env.step({"reset": np.array(False), "action": np.array(0, np.int32)})

        assert not obs["is_first"]
        assert env._env.player.position_px != before
        assert "log/noop_mapped" in obs
        assert obs["log/noop_mapped"] == np.float32(1.0)
    finally:
        env.close()


def test_dreamerv3_make_env_routes_env_diy_task(monkeypatch):
    _install_dreamer_fakes(monkeypatch)
    sys.modules.pop("dreamerv3.main", None)
    main = importlib.import_module("dreamerv3.main")

    config = _FakeConfig(
        task="env_diy_default",
        seed=0,
        logdir="/tmp/dreamerv3-env-diy-test",
        env={
            "env_diy": {
                "config_path": "env_diy/map_data/dungeons/prototype/dungeon.json",
                "image": False,
                "length": 5,
                "logs": True,
                "use_seed": True,
                "move_speed_px": 4,
                "agent_noop_enabled": False,
                "stuck_penalty_enabled": False,
            },
        },
    )
    env = main.make_env(config, 0)
    try:
        obs = env.step({"reset": np.array(True), "action": np.array(0, np.int32)})
        assert obs["is_first"]
        assert "vector" in obs
    finally:
        env.close()
