from __future__ import annotations

from typing import Any

from ..env import make_env
from ..rewards import RewardWrapper
from ..rewards.reward_fn import AvoidTrapReward, KeyDoorReward, KillMonsterReward
from ..tasks import get_task_spec as get_runtime_task_spec
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
    env = make_env(spec.config_path, api="gym", **kwargs)
    if spec.task_registry_id is not None:
        task_spec = get_runtime_task_spec(spec.task_registry_id)
        reward_cls = {
            "avoid_traps_room_001": AvoidTrapReward,
            "kill_monsters_room_001": KillMonsterReward,
            "key_door_room_001": KeyDoorReward,
        }[task_spec.task_id]
        env = RewardWrapper(env, reward_fn=reward_cls(task_spec))
    if seed is not None:
        env.action_space.seed(seed)
    return env
