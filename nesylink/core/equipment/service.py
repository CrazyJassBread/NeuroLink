from __future__ import annotations

from typing import Any

from ..state import EquipmentSlot, PlayerState, ToolType
from .registry import EQUIPMENT_HANDLERS
from .types import EquipmentUseResult
from .weapons import sword_attack_rect


def trigger_equipment(engine: Any, slot: EquipmentSlot, result: Any) -> EquipmentUseResult:
    runtime = engine.runtime
    item_name = runtime.player.equipped_tool(slot)
    handler = EQUIPMENT_HANDLERS.get(item_name)
    if handler is None or item_name in {ToolType.NONE.value, ToolType.INTERACT.value}:
        runtime.last_message = f"{slot.value} NO EFFECT"
        result.events.append("action_no_effect")
        return EquipmentUseResult()
    return handler(engine, result)


def active_block_item(player: PlayerState) -> str | None:
    if player.action_item == "shield" and player.action_ticks_remaining > 0:
        return "shield"
    return None


def current_attack_hitbox(player: PlayerState) -> tuple[float, float, float, float] | None:
    if player.action_item != "sword" or player.action_ticks_remaining <= 0:
        return None
    return sword_attack_rect(player.position_px, player.action_facing or player.facing)
