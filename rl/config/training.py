from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "rl" / "outputs"

SUPPORTED_METHODS = ("ppo", "dqn", "a3c")

TASK_ROOM_CONFIGS: dict[str, tuple[Path, str | None, str | None]] = {
    "prototype": (PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "prototype" / "dungeon.json", None, None),
    "combat_training": (
        PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "combat_training" / "dungeon.json",
        None,
        None,
    ),
    "evasion_training": (
        PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "evasion_training" / "dungeon.json",
        None,
        None,
    ),
    "chest_training": (
        PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "chest_training" / "dungeon.json",
        None,
        None,
    ),
    "avoid_traps": (
        PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "avoid_traps" / "room_001.json",
        "sparse_exit",
        None,
    ),
    "kill_monsters": (
        PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "kill_monsters" / "room_001.json",
        "kill_monster",
        None,
    ),
    "key_door": (
        PROJECT_ROOT / "nesylink" / "map_data" / "dungeons" / "key_door" / "room_001.json",
        "collect_key",
        None,
    ),
}


@dataclass(frozen=True)
class TrainingTarget:
    name: str
    map_path: Path
    reward_id: str | None = None
    reward_module: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    method: str = "ppo"
    episodes: int = 5
    max_steps: int = 500
    total_timesteps: int = 50_000
    gpu: int | None = None
    seed: int = 0
    action_repeat: int = 1
    task_rooms: tuple[str, ...] = ("prototype",)
    config_path: Path | None = None
    output_dir: Path = OUTPUT_ROOT
    render: bool = False
    skip_train: bool = False

    @property
    def device(self) -> str:
        if self.gpu is None:
            return "auto"
        if self.gpu < 0:
            return "cpu"
        return f"cuda:{self.gpu}"


DEFAULT_TRAINING_CONFIG = TrainingConfig()


def apply_overrides(config: TrainingConfig = DEFAULT_TRAINING_CONFIG, **overrides: object) -> TrainingConfig:
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean_overrides)


def resolve_task_rooms(
    task_rooms: Sequence[str],
    *,
    config_path: str | Path | None = None,
) -> list[TrainingTarget]:
    if config_path is not None:
        path = _resolve_project_path(config_path)
        name = path.stem if path.name != "dungeon.json" else path.parent.name
        return [TrainingTarget(name=name, map_path=path, reward_id=None, reward_module=None)]

    if not task_rooms:
        raise ValueError("at least one task room must be selected")

    targets: list[TrainingTarget] = []
    for room_name in task_rooms:
        if room_name not in TASK_ROOM_CONFIGS:
            allowed = ", ".join(sorted(TASK_ROOM_CONFIGS))
            raise ValueError(f"unsupported task room '{room_name}', allowed: {allowed}")
        room_config_path, reward_id, reward_module = TASK_ROOM_CONFIGS[room_name]
        targets.append(
            TrainingTarget(
                name=room_name,
                map_path=room_config_path,
                reward_id=reward_id,
                reward_module=reward_module,
            )
        )
    return targets


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
