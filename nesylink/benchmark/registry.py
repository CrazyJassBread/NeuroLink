from __future__ import annotations

from typing import Any

from ..wrappers.gym_env import make_gym_env, with_default_seed
from .specs import BenchmarkSuiteSpec, BenchmarkTaskSpec
from .suites import NESYLINK_V0


_SUITES: dict[str, BenchmarkSuiteSpec] = {
    NESYLINK_V0.suite_id: NESYLINK_V0,
}


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
    env = make_gym_env(
        map_path=spec.map_path,
        reward_id=spec.reward_id,
        reward_module=spec.reward_module,
        max_steps=spec.max_episode_steps,
        **kwargs,
    )
    if seed is not None:
        env = with_default_seed(env, seed)
    return env
