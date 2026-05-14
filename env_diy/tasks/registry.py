from __future__ import annotations

from pathlib import Path

from .task_spec import TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"

_REGISTRY: dict[str, TaskSpec] = {}


def register_task_spec(*aliases: str, spec: TaskSpec) -> None:
    for alias in aliases:
        _REGISTRY[alias] = spec


def get_task_spec(task_id: str) -> TaskSpec:
    try:
        return _REGISTRY[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown task spec '{task_id}', available: {available}") from exc


def list_task_specs() -> list[str]:
    return sorted(_REGISTRY)


register_task_spec(
    "avoid_traps",
    "avoid_traps_room_001",
    spec=TaskSpec(
        task_id="avoid_traps_room_001",
        map_path=DUNGEON_ROOT / "avoid_traps" / "room_001.json",
        target_exit="north_exit",
        reward_weights={
            "trap_triggered": -1.0,
            "exit_reached": 10.0,
        },
        success_condition="Reach the target exit without triggering a trap during the episode.",
        failure_condition="Die or reach the target exit after a trap trigger.",
    ),
)
register_task_spec(
    "kill_monsters",
    "kill_monsters_room_001",
    spec=TaskSpec(
        task_id="kill_monsters_room_001",
        map_path=DUNGEON_ROOT / "kill_monsters" / "room_001.json",
        reward_weights={
            "monster_killed": 0.5,
            "room_cleared": 10.0,
        },
        success_condition="Defeat all monsters in the room.",
        failure_condition="Die before the room is cleared.",
    ),
)
register_task_spec(
    "key_door",
    "key_door_room_001",
    spec=TaskSpec(
        task_id="key_door_room_001",
        map_path=DUNGEON_ROOT / "key_door" / "room_001.json",
        target_exit="north_exit",
        reward_weights={
            "key_collected": 0.5,
            "door_opened": 0.5,
            "exit_reached": 10.0,
        },
        success_condition="Collect the key, open the door, and reach the target exit.",
        failure_condition="Die before clearing the room.",
    ),
)
