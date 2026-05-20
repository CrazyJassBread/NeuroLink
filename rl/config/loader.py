from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    DEFAULT_OUTPUT_ROOT,
    PROJECT_ROOT,
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)


def load_training_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> TrainingConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"training config must deserialize to a mapping: {config_path}")

    merged = _deep_copy_mapping(raw)
    for override in overrides:
        _apply_override(merged, override)
    return _build_training_config(merged)


def _build_training_config(payload: dict[str, Any]) -> TrainingConfig:
    experiment_payload = _expect_mapping(payload, "experiment")
    environment_payload = _expect_mapping(payload, "environment")
    algorithm_payload = _expect_mapping(payload, "algorithm")
    evaluation_payload = dict(payload.get("evaluation", {}) or {})

    experiment = ExperimentConfig(
        name=str(experiment_payload.get("name", "default_experiment")),
        seed=int(experiment_payload.get("seed", 0)),
        device=str(experiment_payload.get("device", "auto")),
        output_dir=_resolve_path(experiment_payload.get("output_dir"), DEFAULT_OUTPUT_ROOT),
        resume=bool(experiment_payload.get("resume", False)),
    )
    environment = EnvironmentConfig(
        id=str(environment_payload["id"]),
        task_id=_optional_str(environment_payload.get("task_id")),
        map_path=_optional_path(environment_payload.get("map_path")),
        reward_id=_optional_str(environment_payload.get("reward_id")),
        reward_module=_optional_str(environment_payload.get("reward_module")),
        render=bool(environment_payload.get("render", False)),
        num_envs=int(environment_payload.get("num_envs", 1)),
        action_repeat=int(environment_payload.get("action_repeat", 1)),
        max_episode_steps=_optional_int(environment_payload.get("max_episode_steps")),
        params=_collect_extra(
            environment_payload,
            {
                "id",
                "task_id",
                "map_path",
                "reward_id",
                "reward_module",
                "render",
                "num_envs",
                "action_repeat",
                "max_episode_steps",
            },
        ),
    )
    algorithm = AlgorithmConfig(
        name=str(algorithm_payload["name"]),
        total_timesteps=int(algorithm_payload.get("total_timesteps", 50_000)),
        params=_collect_extra(algorithm_payload, {"name", "total_timesteps"}),
    )
    evaluation = EvaluationConfig(
        enabled=bool(evaluation_payload.get("enabled", True)),
        episodes=int(evaluation_payload.get("episodes", 5)),
        deterministic=bool(evaluation_payload.get("deterministic", True)),
        save_model=bool(evaluation_payload.get("save_model", True)),
        save_metrics=bool(evaluation_payload.get("save_metrics", True)),
        render=bool(evaluation_payload.get("render", False)),
        params=_collect_extra(
            evaluation_payload,
            {"enabled", "episodes", "deterministic", "save_model", "save_metrics", "render"},
        ),
    )
    return TrainingConfig(
        experiment=experiment,
        environment=environment,
        algorithm=algorithm,
        evaluation=evaluation,
    )


def _expect_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"training config requires a '{key}' mapping")
    return dict(value)


def _apply_override(payload: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must be in key=value form: {override}")
    dotted_key, raw_value = override.split("=", 1)
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError(f"override key is empty: {override}")
    node = payload
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"override path collides with non-mapping key '{key}' in {override}")
        node = child
    node[keys[-1]] = yaml.safe_load(raw_value)


def _deep_copy_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_mapping(value)
        else:
            result[key] = value
    return result


def _collect_extra(payload: dict[str, Any], known_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in known_keys}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value)


def _resolve_path(value: Any, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("path value is required")
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
