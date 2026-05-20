from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "rl" / "outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "default_experiment"
    seed: int = 0
    device: str = "auto"
    output_dir: Path = DEFAULT_OUTPUT_ROOT
    resume: bool = False


@dataclass(frozen=True)
class EnvironmentConfig:
    id: str
    task_id: str | None = None
    map_path: Path | None = None
    reward_id: str | None = None
    reward_module: str | None = None
    render: bool = False
    num_envs: int = 1
    action_repeat: int = 1
    max_episode_steps: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    total_timesteps: int = 50_000
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = True
    episodes: int = 5
    deterministic: bool = True
    save_model: bool = True
    save_metrics: bool = True
    render: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    experiment: ExperimentConfig
    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
