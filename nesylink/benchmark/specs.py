from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    suite_id: str
    task_id: str
    map_id: str
    map_path: Path
    reward_id: str | None
    reward_module: str | None
    difficulty: str
    max_episode_steps: int
    observation_mode: str
    action_mode: str
    objective: str


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    suite_id: str
    tasks: tuple[BenchmarkTaskSpec, ...]
