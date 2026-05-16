from __future__ import annotations

from typing import Any

from ..constants import MONSTER_STUN_TICKS
from ..state import EquipmentSlot, ToolType, aabb_overlap, is_adjacent, tile_to_top_left_px
from .combat import remove_defeated_monster


def handle_equipped_action(engine: Any, slot: EquipmentSlot, result: Any) -> bool:
    runtime = engine.runtime
    tool = runtime.player.equipped_tool(slot)
    if slot == EquipmentSlot.A and tool == ToolType.INTERACT.value:
        runtime.last_message = "INTERACT"
        handle_action_a(engine, result)
        return False
    if slot == EquipmentSlot.B and tool == ToolType.SHIELD.value:
        runtime.last_message = "SHIELD"
        return True

    runtime.last_message = f"{slot.value} NO EFFECT"
    result.events.append("action_no_effect")
    return False


def handle_action_a(engine: Any, result: Any) -> None:
    runtime = engine.runtime
    player_tile = runtime.snapshot().player_tile

    for chest in runtime.room.chests.values():
        if not chest.is_open and is_adjacent(player_tile, chest.pos):
            chest.is_open = True
            apply_loot(runtime, chest.loot, result)
            result.events.append("chest_opened")
            return

    for npc in runtime.room.npcs.values():
        if is_adjacent(player_tile, npc.pos):
            runtime.last_message = npc.text.upper()[:24]
            result.events.append("talked_npc")
            return

    if handle_monster_attack(engine, result):
        return

    runtime.last_message = "A NO EFFECT"
    result.events.append("action_no_effect")


def handle_monster_attack(engine: Any, result: Any) -> bool:
    runtime = engine.runtime
    player_tile = runtime.snapshot().player_tile
    for monster in list(runtime.room.monsters.values()):
        if not (
            is_adjacent(player_tile, monster.tile_pos)
            or aabb_overlap(runtime.player.position_px, runtime.player.size_px, monster.position_px, monster.size_px)
        ):
            continue
        monster.hp -= 1
        monster.stun_ticks_remaining = MONSTER_STUN_TICKS
        if monster.hp <= 0:
            remove_defeated_monster(engine, monster, result, killed_by="attack")
            runtime.last_message = f"ATTACK KILL {monster.monster_type.upper()}"
            return True

        runtime.last_message = f"ATTACK HIT ({monster.hp}HP LEFT)"
        result.events.append("action_attack")
        result.events.append("monster_damaged")
        result.event_details.append(
            {
                "type": "monster_damaged",
                "monster_id": monster.monster_id,
                "monster_type": monster.monster_type,
                "monster_hp_remaining": monster.hp,
                "damaged_by": "attack",
            }
        )
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
