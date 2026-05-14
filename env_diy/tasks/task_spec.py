from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    map_path: Path
    target_exit: str | None = None
    target_monsters: tuple[str, ...] = ()
    reward_weights: dict[str, float] = field(default_factory=dict)
    success_condition: str = ""
    failure_condition: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskOutcome:
    success: bool = False
    failure: bool = False
    terminated_reason: str | None = None
    progress: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardFn(Protocol):
    last_outcome: TaskOutcome
    last_reward_terms: dict[str, float]

    def reset(self) -> None: ...

    def __call__(
        self,
        prev_obs: Any,
        prev_info: dict[str, Any],
        obs: Any,
        info: dict[str, Any],
        action: int,
    ) -> tuple[float, bool]: ...
