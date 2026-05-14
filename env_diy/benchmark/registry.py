from __future__ import annotations

from typing import Any

import gymnasium as gym

from ..env import make_env
from .specs import BenchmarkSuiteSpec, BenchmarkTaskSpec
from .suites import NESYLINK_V0


_SUITES: dict[str, BenchmarkSuiteSpec] = {
    NESYLINK_V0.suite_id: NESYLINK_V0,
}


class _DefaultSeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._default_seed = int(seed)

    def reset(self, *, seed: int | None = None, options=None):
        return self.env.reset(
            seed=self._default_seed if seed is None else seed,
            options=options,
        )


def list_suites() -> list[str]:
    return sorted(_SUITES)


def get_suite(suite_id: str) -> BenchmarkSuiteSpec:
    try:
        return _SUITES[suite_id]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark suite '{suite_id}'") from exc


def list_tasks(suite_id: str) -> list[BenchmarkTaskSpec]:
    return list(get_suite(suite_id).tasks)


def get_task_spec(suite_id: str, task_id: str) -> BenchmarkTaskSpec:
    for task in get_suite(suite_id).tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown task '{task_id}' in suite '{suite_id}'")


def make_benchmark_env(
    suite_id: str,
    task_id: str,
    seed: int | None = None,
    **kwargs: Any,
):
    spec = get_task_spec(suite_id, task_id)
    env = make_env(
        map_path=spec.map_path,
        api="gym",
        reward_id=spec.reward_id,
        reward_module=spec.reward_module,
        max_steps=spec.max_episode_steps,
        **kwargs,
    )
    if seed is not None:
        env = _DefaultSeedWrapper(env, seed)
        env.action_space.seed(seed)
    return env
