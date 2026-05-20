from __future__ import annotations

from typing import Any

from ..constants import MONSTER_STUN_TICKS, SWORD_SWING_DURATION_TICKS, TILE_SIZE
from ..state import entity_rect
from .types import EquipmentUseResult


def use_sword(engine: Any, result: Any) -> EquipmentUseResult:
    runtime = engine.runtime
    runtime.player.start_action(
        item_name="sword",
        pose="swing",
        facing=runtime.player.facing,
        ticks=SWORD_SWING_DURATION_TICKS,
    )
    result.events.append("action_attack")

    attack_rect = sword_attack_rect(runtime.player.position_px, runtime.player.facing)
    from ..mechanics.combat import apply_monster_knockback, remove_defeated_monster

    for monster in list(runtime.room.monsters.values()):
        if not rects_overlap(attack_rect, entity_rect(monster.position_px, monster.size_px)):
            continue

        monster.hp -= 1
        knockback_applied_px = apply_monster_knockback(engine, monster)
        monster.stun_ticks_remaining = MONSTER_STUN_TICKS
        if monster.hp <= 0:
            remove_defeated_monster(engine, monster, result, killed_by="sword")
            runtime.last_message = f"SWORD KILL {monster.monster_type.upper()}"
            break

        runtime.last_message = f"SWORD HIT ({monster.hp}HP LEFT)"
        result.events.append("monster_damaged")
        result.event_details.append(
            {
                "type": "monster_damaged",
                "monster_id": monster.monster_id,
                "monster_type": monster.monster_type,
                "monster_hp_remaining": monster.hp,
                "damaged_by": "sword",
                "monster_knockback_px": TILE_SIZE,
                "knockback_applied_px": knockback_applied_px,
                "monster_stun_ticks": MONSTER_STUN_TICKS,
            }
        )
        break
    else:
        runtime.last_message = "SWORD"

    return EquipmentUseResult(
        used=True,
        item_name="sword",
        pose="swing",
        duration_ticks=SWORD_SWING_DURATION_TICKS,
    )


def sword_attack_rect(position_px: tuple[float, float], facing: str) -> tuple[float, float, float, float]:
    left, top = position_px
    if facing == "up":
        return left, top - TILE_SIZE, left + TILE_SIZE, top
    if facing == "down":
        return left, top + TILE_SIZE, left + TILE_SIZE, top + TILE_SIZE * 2
    if facing == "left":
        return left - TILE_SIZE, top, left, top + TILE_SIZE
    return left + TILE_SIZE, top, left + TILE_SIZE * 2, top + TILE_SIZE


def rects_overlap(
    left_rect: tuple[float, float, float, float],
    right_rect: tuple[float, float, float, float],
) -> bool:
    left_l, left_t, left_r, left_b = left_rect
    right_l, right_t, right_r, right_b = right_rect
    return not (
        left_r <= right_l
        or left_l >= right_r
        or left_b <= right_t
        or left_t >= right_b
    )
