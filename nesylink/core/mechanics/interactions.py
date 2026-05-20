from __future__ import annotations

from typing import Any

from ..equipment import trigger_equipment
from ..state import EquipmentSlot
from ..state import is_adjacent, tile_to_top_left_px


def handle_equipped_action(engine: Any, slot: EquipmentSlot, result: Any) -> bool:
    return trigger_equipment(engine, slot, result).used


def try_interaction(engine: Any, result: Any) -> bool:
    runtime = engine.runtime
    player_tile = runtime.snapshot().player_tile

    for chest in runtime.room.chests.values():
        if not chest.is_open and is_adjacent(player_tile, chest.pos):
            chest.is_open = True
            apply_loot(runtime, chest.loot, result)
            result.events.append("chest_opened")
            return True

    for npc in runtime.room.npcs.values():
        if is_adjacent(player_tile, npc.pos):
            runtime.last_message = npc.text.upper()[:24]
            result.events.append("talked_npc")
            return True

    return False


def apply_loot(runtime: Any, loot: dict, result: Any) -> None:
    loot_kind = str(loot.get("kind", "gold"))
    amount = int(loot.get("amount", 1))

    if loot_kind == "key":
        runtime.player.keys += max(1, amount)
        runtime.last_message = "GOT KEY"
        result.events.append("key_collected")
        return
    if loot_kind == "heal":
        healed = min(runtime.player.max_health, runtime.player.health + max(1, amount))
        runtime.player.health = healed
        runtime.last_message = "HEALED"
        result.events.append("agent_healed")
        return
    if loot_kind == "item":
        item_name = str(loot.get("item_id", "item"))
        if item_name not in runtime.player.items:
            runtime.player.items.append(item_name)
        runtime.last_message = f"GOT {item_name}".upper()[:24]
        result.events.append("item_collected")
        return

    runtime.player.gold += max(1, amount)
    runtime.last_message = "GOT GOLD"
    result.events.append("gold_collected")


def resolve_tile_effects(engine: Any, result: Any) -> None:
    runtime = engine.runtime
    player_tile = runtime.snapshot().player_tile

    button = runtime.room.button_at(player_tile)
    if button is not None and not button.is_pressed:
        button.is_pressed = True
        runtime.last_message = button.message.upper()[:24]
        result.events.append("button_pressed")

    trap = runtime.room.trap_at(player_tile)
    if trap is not None:
        runtime.player.health = max(0, runtime.player.health - trap.damage)
        respawn_name = trap.respawn_to if trap.respawn_to in runtime.room.spawns else runtime.room.default_spawn_name
        if runtime.player.health > 0:
            runtime.player.position_px = tile_to_top_left_px(runtime.room.spawns[respawn_name])
        runtime.last_message = f"TRAP -{trap.damage}HP"
        result.events.append("trap_triggered")
        result.events.append("agent_damaged")
        result.event_details.append(
            {
                "type": "trap_triggered",
                "trap_id": trap.trap_id,
                "damage": trap.damage,
                "respawn_to": respawn_name,
            }
        )
        if trap.single_use:
            trap.is_active = False
