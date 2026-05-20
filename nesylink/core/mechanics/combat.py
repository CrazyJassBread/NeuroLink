from __future__ import annotations

import math
from typing import Any

from ..constants import MONSTER_HIT_KNOCKBACK_PX, MONSTER_KILL_GOLD_REWARD, MONSTER_STUN_TICKS, TILE_SIZE
from ..monsters import MonsterState, update_monster
from ..state import aabb_overlap, entity_center_px, move_with_tile_collisions


def update_monsters(engine: Any, result: Any) -> None:
    del result
    runtime = engine.runtime
    occupied_tiles = {monster.tile_pos for monster in runtime.room.monsters.values()}
    for monster in runtime.room.monsters.values():
        if monster.stun_ticks_remaining > 0:
            monster.stun_ticks_remaining -= 1
            monster.last_move_delta_px = (0.0, 0.0)
            continue
        occupied_tiles.discard(monster.tile_pos)
        update_monster(monster, runtime.player.position_px, runtime.room.walls, occupied_tiles)
        occupied_tiles.add(monster.tile_pos)


def resolve_monster_contact(engine: Any, result: Any) -> None:
    runtime = engine.runtime
    from ..equipment import active_block_item

    shield_active = active_block_item(runtime.player) == "shield"
    for monster in runtime.room.monsters.values():
        if monster.stun_ticks_remaining > 0:
            continue
        if not aabb_overlap(
            runtime.player.position_px,
            runtime.player.size_px,
            monster.position_px,
            monster.size_px,
        ):
            continue
        if shield_active:
            knockback_applied_px = apply_monster_knockback(engine, monster)
            monster.stun_ticks_remaining = MONSTER_STUN_TICKS
            runtime.last_message = "SHIELD BLOCK"
            result.events.append("shield_block")
            result.event_details.append(
                {
                    "type": "shield_block",
                    "monster_id": monster.monster_id,
                    "monster_type": monster.monster_type,
                    "damage_prevented": monster.damage,
                    "monster_hp_remaining": monster.hp,
                    "monster_knockback_px": MONSTER_HIT_KNOCKBACK_PX,
                    "knockback_applied_px": knockback_applied_px,
                    "monster_stun_ticks": MONSTER_STUN_TICKS,
                }
            )
            break

        runtime.player.health = max(0, runtime.player.health - monster.damage)
        knockback_applied_px = apply_monster_knockback(engine, monster)
        monster.stun_ticks_remaining = MONSTER_STUN_TICKS
        runtime.last_message = f"HIT -{monster.damage}HP"
        result.events.append("agent_damaged")
        result.event_details.append(
            {
                "type": "agent_damaged",
                "monster_id": monster.monster_id,
                "damage": monster.damage,
                "monster_hp_remaining": monster.hp,
                "monster_knockback_px": MONSTER_HIT_KNOCKBACK_PX,
                "knockback_applied_px": knockback_applied_px,
                "monster_stun_ticks": MONSTER_STUN_TICKS,
            }
        )
        break


def remove_defeated_monster(engine: Any, monster: MonsterState, result: Any, *, killed_by: str) -> None:
    runtime = engine.runtime
    if monster.monster_id in runtime.room.monsters:
        del runtime.room.monsters[monster.monster_id]
    runtime.player.gold += MONSTER_KILL_GOLD_REWARD
    result.events.append("monster_killed")
    result.event_details.append(
        {
            "type": "monster_killed",
            "monster_id": monster.monster_id,
            "monster_type": monster.monster_type,
            "gold_reward": MONSTER_KILL_GOLD_REWARD,
            "killed_by": killed_by,
        }
    )
    unlock_all_monster_defeated_exits(runtime, result)


def unlock_all_monster_defeated_exits(runtime: Any, result: Any) -> None:
    if runtime.room.monsters:
        return
    for exit_cfg in runtime.room.exits:
        if not exit_cfg.requires.get("all_monsters_defeated"):
            continue
        exit_state = runtime.room.exit_state(exit_cfg)
        if exit_state.unlocked:
            continue
        exit_state.unlocked = True
        exit_state.opened = True
        runtime.last_message = "ALL MONSTERS DEFEATED - DOOR OPENED"
        result.events.append("door_opened")
        result.event_details.append(
            {
                "type": "door_opened",
                "exit_id": exit_cfg.exit_id,
                "trigger": "all_monsters_defeated",
            }
        )


def apply_monster_knockback(engine: Any, monster: MonsterState) -> float:
    runtime = engine.runtime
    knockback_dx, knockback_dy = monster_knockback_vector(runtime, monster)
    other_monster_tiles = {
        other.tile_pos
        for other in runtime.room.monsters.values()
        if other.monster_id != monster.monster_id
    }
    world_blockers = runtime.room.blocking_tiles() | other_monster_tiles
    previous_position = monster.position_px
    for distance in (float(MONSTER_HIT_KNOCKBACK_PX), 12.0, 8.0, 4.0):
        candidate_position = move_with_tile_collisions(
            previous_position,
            monster.size_px,
            (knockback_dx * distance, knockback_dy * distance),
            world_blockers,
        )
        moved_px = math.hypot(
            candidate_position[0] - previous_position[0],
            candidate_position[1] - previous_position[1],
        )
        if moved_px + 1e-6 >= distance:
            monster.position_px = candidate_position
            monster.last_move_delta_px = (
                monster.position_px[0] - previous_position[0],
                monster.position_px[1] - previous_position[1],
            )
            return distance

    apply_player_separation(runtime, monster)
    monster.last_move_delta_px = (0.0, 0.0)
    return 0.0


def apply_player_separation(runtime: Any, monster: MonsterState) -> None:
    player_center = entity_center_px(runtime.player.position_px, runtime.player.size_px)
    monster_center = entity_center_px(monster.position_px, monster.size_px)
    dx = player_center[0] - monster_center[0]
    dy = player_center[1] - monster_center[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        dx, dy = 0.0, -1.0
        distance = 1.0
    sep_dx = (dx / distance) * TILE_SIZE
    sep_dy = (dy / distance) * TILE_SIZE
    runtime.player.position_px = move_with_tile_collisions(
        runtime.player.position_px,
        runtime.player.size_px,
        (sep_dx, sep_dy),
        runtime.room.blocking_tiles(),
    )


def monster_knockback_vector(runtime: Any, monster: MonsterState) -> tuple[float, float]:
    player_center = entity_center_px(runtime.player.position_px, runtime.player.size_px)
    monster_center = entity_center_px(monster.position_px, monster.size_px)
    dx = monster_center[0] - player_center[0]
    dy = monster_center[1] - player_center[1]
    distance = math.hypot(dx, dy)

    if distance <= 1e-6:
        last_move_x, last_move_y = monster.last_move_delta_px
        move_length = math.hypot(last_move_x, last_move_y)
        if move_length > 1e-6:
            dx = -last_move_x / move_length
            dy = -last_move_y / move_length
        else:
            dx, dy = 1.0, 0.0
        distance = 1.0

    return dx / distance, dy / distance
