from __future__ import annotations

from typing import Any

from ..constants import SHIELD_RAISE_DURATION_TICKS
from .types import EquipmentUseResult


def use_shield(engine: Any, result: Any) -> EquipmentUseResult:
    runtime = engine.runtime
    runtime.player.start_action(
        item_name="shield",
        pose="raise",
        facing=runtime.player.facing,
        ticks=SHIELD_RAISE_DURATION_TICKS,
    )
    runtime.last_message = "SHIELD"
    result.events.append("action_shield")
    return EquipmentUseResult(
        used=True,
        item_name="shield",
        pose="raise",
        duration_ticks=SHIELD_RAISE_DURATION_TICKS,
    )
