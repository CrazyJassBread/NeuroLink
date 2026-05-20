from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquipmentUseResult:
    used: bool = False
    item_name: str | None = None
    pose: str | None = None
    duration_ticks: int = 0
