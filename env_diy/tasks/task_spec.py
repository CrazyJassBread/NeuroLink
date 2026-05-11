from __future__ import annotations

from dataclasses import dataclass, field

from ..maps.tasks import TaskConfig


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    room_id: str
    objective_type: str
    target_exit: str | None = None
    target_monsters: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_task_config(cls, task_config: TaskConfig | None) -> TaskSpec | None:
        if task_config is None:
            return None
        return cls(
            task_id=task_config.task_id,
            task_type=task_config.task_type,
            room_id=task_config.room_id,
            objective_type=task_config.objective.objective_type,
            target_exit=task_config.objective.target_exit,
            target_monsters=task_config.objective.target_monsters,
        )
