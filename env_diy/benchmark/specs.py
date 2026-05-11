from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    suite_id: str
    task_id: str
    map_id: str
    task_rooms: tuple[str, ...]
    config_path: Path
    difficulty: str
    max_episode_steps: int
    default_reward_mode: str
    supported_reward_modes: tuple[str, ...]
    observation_mode: str
    action_mode: str
    success_condition: str
    failure_condition: str


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    suite_id: str
    tasks: tuple[BenchmarkTaskSpec, ...]
