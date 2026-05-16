from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeSnapshot:
    room_id: str
    room_coord: tuple[int, int]
    player_position_px: tuple[float, float]
    player_tile: tuple[int, int]
    health: int
    gold: int
    keys: int
    items: tuple[str, ...]
    no_progress_steps: int
    step_count: int
    episode_id: int
    seed: int | None


@dataclass(frozen=True)
class StuckPenaltyConfig:
    enabled: bool
    steps: int
    reward: float


@dataclass
class EngineStepResult:
    events: list[str] = field(default_factory=list)
    event_details: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    auto_reset: bool = False
    shield_active: bool = False
    move_direction: str | None = None
    progress_start_pos: tuple[float, float] | None = None
    progress_start_room_id: str | None = None
    last_message: str = ""
    terminated_reason: str | None = None


RewardBreakdown = dict[str, float]
RewardTerms = dict[str, float]
