from __future__ import annotations

import sys
import types

import pytest

from env_diy.rewards.base import BaseReward
from env_diy.rewards.loader import load_reward, resolve_reward_module


def test_load_reward_supports_builtin_reward_id() -> None:
    reward = load_reward(reward_id="sparse_exit")

    assert isinstance(reward, BaseReward)
    assert reward.reward_name == "sparse_exit"


def test_load_reward_supports_make_reward_import_path() -> None:
    module_name = "tests.fake_make_reward_module"
    module = types.ModuleType(module_name)

    class FakeReward(BaseReward):
        reward_name = "fake_reward"

    def make_reward(**kwargs):
        return FakeReward(**kwargs)

    module.make_reward = make_reward
    sys.modules[module_name] = module

    try:
        reward = load_reward(reward_module=module_name)
    finally:
        sys.modules.pop(module_name, None)

    assert isinstance(reward, BaseReward)
    assert reward.reward_name == "fake_reward"


def test_load_reward_passes_reward_kwargs_to_module_factory() -> None:
    module_name = "tests.fake_kwargs_reward_module"
    module = types.ModuleType(module_name)
    seen_kwargs: dict[str, float] = {}

    class FakeReward(BaseReward):
        reward_name = "fake_kwargs"

    def make_reward(**kwargs):
        seen_kwargs.update(kwargs)
        return FakeReward(**kwargs)

    module.make_reward = make_reward
    sys.modules[module_name] = module

    try:
        reward = load_reward(reward_module=module_name, reward_kwargs={"step": -0.25, "gold_delta": 2.0})
    finally:
        sys.modules.pop(module_name, None)

    assert reward.weights["step"] == -0.25
    assert reward.weights["gold_delta"] == 2.0
    assert seen_kwargs == {"step": -0.25, "gold_delta": 2.0}


def test_resolve_reward_module_requires_factory() -> None:
    module_name = "tests.bad_reward_module"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module

    try:
        with pytest.raises(ValueError, match="make_reward"):
            resolve_reward_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
