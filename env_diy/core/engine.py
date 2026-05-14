from __future__ import annotations

import math
from pathlib import Path
import random

from ..core.constants import (
    ACTION_A,
    ACTION_B,
    ACTION_NOOP,
    MAP_PIXEL_HEIGHT,
    MAP_PIXEL_WIDTH,
    MESSAGE_DEFAULT,
    MONSTER_HIT_KNOCKBACK_PX,
    MONSTER_KILL_GOLD_REWARD,
    MONSTER_STUN_TICKS,
    MOVE_ACTION_TO_DIRECTION,
    PLAYER_SPEED_PX_PER_TICK,
    TILE_SIZE,
)
from ..entities import (
    EquipmentSlot,
    PlayerState,
    ToolType,
    aabb_overlap,
    entity_center_px,
    is_adjacent,
    move_with_tile_collisions,
    tile_to_top_left_px,
)
from ..entities.monsters import MonsterState, update_monster
from ..maps.rooms import (
    ExitConfig,
    RoomManager,
    RoomState,
    direction_from_entry_name,
    first_valid_entry_spawn_tile,
)
from .runtime import RuntimeState
from .types import EngineStepResult


MOVE_TO_EXIT_DIRECTION = {
    "up": "north",
    "down": "south",
    "left": "west",
    "right": "east",
}


class DungeonEngine:
    def __init__(self, room_file: str | Path, *, move_speed_px: float = PLAYER_SPEED_PX_PER_TICK):
        self.room_manager = RoomManager(room_file)
        self.map_id = self.room_manager.room_file.stem
        self.move_speed_px = max(1, int(move_speed_px))
        self.max_monster_slots = max(1, self.room_manager.max_monsters)
        self.world_completion_via_exit = any(
            exit_cfg.complete_task
            for room in self.room_manager.room_templates.values()
            for exit_cfg in room.exits
        )
        self.seed: int | None = None
        self.rng = random.Random()
        self.runtime = self._build_initial_runtime()

    def reset(self, *, seed: int | None = None) -> RuntimeState:
        self.seed = seed
        self.rng = random.Random(seed)
        previous_episode = self.runtime.episode
        self.room_manager.reset_room_cache()
        self.runtime = self._build_initial_runtime()
        self.runtime.episode = previous_episode + 1
        self.runtime.seed = seed
        return self.runtime

    def step(self, action: int) -> EngineStepResult:
        runtime = self.runtime
        result = EngineStepResult(
            progress_start_pos=runtime.player.position_px,
            progress_start_room_id=runtime.room.room_id,
        )
        runtime.step_count += 1

        move_direction = MOVE_ACTION_TO_DIRECTION.get(action)
        if move_direction is not None:
            result.move_direction = move_direction
            self._handle_move(move_direction, result)
        elif action == ACTION_A:
            result.events.append("action_interact")
            result.shield_active = self._handle_equipped_action(EquipmentSlot.A, result)
        elif action == ACTION_B:
            result.events.append("action_shield")
            result.shield_active = self._handle_equipped_action(EquipmentSlot.B, result)
        elif action == ACTION_NOOP:
            runtime.last_message = "WAIT"
            result.events.append("noop")

        if runtime.player.health > 0 and move_direction is not None:
            self._resolve_transition(move_direction, result)
        if runtime.player.health > 0:
            self._resolve_tile_effects(result)
        if runtime.player.health > 0:
            self._update_monsters(result)
        if runtime.player.health > 0:
            self._resolve_monster_contact(result, shield_active=result.shield_active)

        if runtime.player.health <= 0:
            result.terminated = True
            runtime.pending_reset = True
            runtime.last_message = "GAME OVER"
            result.events.append("agent_dead")
            result.terminated_reason = "agent_dead"
        elif "environment_completed" in result.events or (not self.world_completion_via_exit and self._all_chests_opened()):
            result.terminated = True
            runtime.pending_reset = True
            runtime.last_message = "WORLD COMPLETE"
            if "environment_completed" not in result.events:
                result.events.append("environment_completed")
            result.terminated_reason = "world_completed"

        if self._step_made_progress(result.progress_start_pos, result.progress_start_room_id, result.events):
            runtime.no_progress_steps = 0
        else:
            runtime.no_progress_steps += 1

        result.last_message = runtime.last_message
        return result

    def hud_lines(self) -> tuple[str, str]:
        runtime = self.runtime
        room_text = f"R:{runtime.room.room_id} HP:{runtime.player.health} G:{runtime.player.gold}"
        items = ",".join(runtime.player.items) if runtime.player.items else "-"
        equipment = (
            f"A:{runtime.player.equipped_tool_label(EquipmentSlot.A.value)} "
            f"B:{runtime.player.equipped_tool_label(EquipmentSlot.B.value)}"
        )
        return room_text, f"I:{items} {equipment}"

    def _build_initial_runtime(self) -> RuntimeState:
        room_coord = self.room_manager.start_room
        room = self.room_manager.get_room(room_coord)
        player = PlayerState(position_px=tile_to_top_left_px(room.spawns[room.default_spawn_name]))
        return RuntimeState(
            room_manager=self.room_manager,
            room_coord=room_coord,
            room=room,
            player=player,
            episode=0,
            step_count=0,
            pending_reset=False,
            last_message=MESSAGE_DEFAULT,
            no_progress_steps=0,
            seed=self.seed,
        )

    def _handle_move(self, direction: str, result: EngineStepResult) -> None:
        runtime = self.runtime
        step_dx, step_dy = {
            "up": (0.0, -1.0),
            "down": (0.0, 1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
        }[direction]
        previous_position = runtime.player.position_px
        current_position = previous_position
        blocked_position: tuple[float, float] | None = None
        for _ in range(self.move_speed_px):
            proposed_position = (
                current_position[0] + step_dx,
                current_position[1] + step_dy,
            )
            next_position = move_with_tile_collisions(
                current_position,
                runtime.player.size_px,
                (step_dx, step_dy),
                runtime.room.blocking_tiles(),
            )
            if next_position == current_position:
                blocked_position = proposed_position
                break
            current_position = next_position

        runtime.player.position_px = current_position
        if runtime.player.position_px == previous_position:
            blocked_reason = "bounds"
            if blocked_position is not None and not self._within_map_bounds(blocked_position):
                runtime.last_message = "EDGE BLOCKED"
            else:
                runtime.last_message = "BLOCKED"
                blocked_reason = "wall"
            result.events.append("action_blocked")
            result.event_details.append(
                {
                    "type": "action_blocked",
                    "reason": blocked_reason,
                    "direction": direction,
                }
            )
            return

        runtime.last_message = f"MOVE {direction.upper()}"
        result.events.append(f"move_{direction}")

    def _resolve_transition(self, direction: str, result: EngineStepResult) -> None:
        runtime = self.runtime
        player_tile = runtime.snapshot().player_tile
        exit_direction = MOVE_TO_EXIT_DIRECTION[direction]
        exit_config = runtime.room.exit_at(player_tile, exit_direction)
        if exit_config is None or not self._player_is_flush_with_edge(direction):
            return
        self._apply_exit(exit_config, result)

    def _apply_exit(self, exit_config: ExitConfig, result: EngineStepResult) -> None:
        runtime = self.runtime
        allowed, blocked_reason = self._can_use_exit(exit_config)
        if not allowed:
            runtime.last_message = exit_config.blocked_message
            result.events.append("action_blocked")
            result.event_details.append(
                {
                    "type": "action_blocked",
                    "reason": blocked_reason,
                    "exit_id": exit_config.exit_id,
                    "direction": exit_config.direction,
                }
            )
            return

        from_room_id = runtime.room.room_id
        exit_state = runtime.room.exit_state(exit_config)
        if exit_config.exit_type == "locked_key" and not exit_state.unlocked:
            key_consumed = False
            if bool(exit_config.requires.get("consume_key", False)):
                runtime.player.keys -= int(exit_config.requires.get("key_count", 1))
                key_consumed = True
            exit_state.unlocked = True
            exit_state.opened = True
            result.events.append("door_opened")
            result.event_details.append(
                {
                    "type": "door_opened",
                    "room_id": from_room_id,
                    "direction": exit_config.direction,
                    "exit_id": exit_config.exit_id,
                    "key_consumed": key_consumed,
                }
            )

        target_coord = self.room_manager.coord_for_room_id(exit_config.target_room_id)
        target_room = self.room_manager.get_room(target_coord)
        spawn_tile = self._entry_spawn_tile(target_room, exit_config.target_entry)
        runtime.room_coord = target_coord
        runtime.room = target_room
        runtime.player.position_px = tile_to_top_left_px(spawn_tile)
        runtime.last_message = exit_config.success_message
        result.events.append("exit_reached")
        result.events.append("room_changed")
        result.event_details.append(
            {
                "type": "exit_reached",
                "from_room": from_room_id,
                "to_room": runtime.room.room_id,
                "exit_id": exit_config.exit_id,
                "direction": exit_config.direction,
                "target_entry": exit_config.target_entry,
                "spawn_px": [runtime.player.position_px[0], runtime.player.position_px[1]],
            }
        )
        if exit_config.complete_task:
            result.events.append("environment_completed")

    def _can_use_exit(self, exit_config: ExitConfig) -> tuple[bool, str]:
        runtime = self.runtime
        if exit_config.exit_type == "normal":
            return True, ""
        if exit_config.exit_type == "locked_key":
            if runtime.room.exit_state(exit_config).unlocked:
                return True, ""
            required_keys = int(exit_config.requires.get("key_count", 1))
            if runtime.player.keys < required_keys:
                return False, "locked"
            return True, ""

        button_id = exit_config.requires.get("button_pressed")
        if button_id is not None:
            button = runtime.room.buttons.get(button_id)
            if button is None or not button.is_pressed:
                return False, "missing_requirement"
        item_name = exit_config.requires.get("item")
        if item_name is not None and item_name not in runtime.player.items:
            return False, "missing_requirement"
        if exit_config.requires.get("all_monsters_defeated") and len(runtime.room.monsters) > 0:
            return False, "missing_requirement"
        return True, ""

    def _entry_spawn_tile(self, room: RoomState, target_entry: str) -> tuple[int, int]:
        entry_direction = direction_from_entry_name(target_entry)
        if entry_direction is not None:
            spawn_tile = first_valid_entry_spawn_tile(entry_direction, room.walls)
            if spawn_tile is None:
                raise RuntimeError(
                    f"room '{room.room_id}' has no valid spawn tile for entry '{target_entry}'"
                )
            return spawn_tile
        return room.spawns[target_entry]

    def _handle_equipped_action(self, slot: EquipmentSlot, result: EngineStepResult) -> bool:
        runtime = self.runtime
        tool = runtime.player.equipped_tool(slot)
        if slot == EquipmentSlot.A and tool == ToolType.INTERACT.value:
            runtime.last_message = "INTERACT"
            self._handle_action_a(result)
            return False
        if slot == EquipmentSlot.B and tool == ToolType.SHIELD.value:
            runtime.last_message = "SHIELD"
            return True

        runtime.last_message = f"{slot.value} NO EFFECT"
        result.events.append("action_no_effect")
        return False

    def _handle_action_a(self, result: EngineStepResult) -> None:
        runtime = self.runtime
        player_tile = runtime.snapshot().player_tile

        for chest in runtime.room.chests.values():
            if not chest.is_open and is_adjacent(player_tile, chest.pos):
                chest.is_open = True
                self._apply_loot(chest.loot, result)
                result.events.append("chest_opened")
                return

        for npc in runtime.room.npcs.values():
            if is_adjacent(player_tile, npc.pos):
                runtime.last_message = npc.text.upper()[:24]
                result.events.append("talked_npc")
                return

        if self._handle_monster_attack(result):
            return

        runtime.last_message = "A NO EFFECT"
        result.events.append("action_no_effect")

    def _handle_monster_attack(self, result: EngineStepResult) -> bool:
        runtime = self.runtime
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
                self._remove_defeated_monster(monster, result, killed_by="attack")
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

    def _apply_loot(self, loot: dict, result: EngineStepResult) -> None:
        runtime = self.runtime
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

    def _resolve_tile_effects(self, result: EngineStepResult) -> None:
        runtime = self.runtime
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

    def _update_monsters(self, result: EngineStepResult) -> None:
        runtime = self.runtime
        occupied_tiles = {monster.tile_pos for monster in runtime.room.monsters.values()}
        for monster in runtime.room.monsters.values():
            if monster.stun_ticks_remaining > 0:
                monster.stun_ticks_remaining -= 1
                monster.last_move_delta_px = (0.0, 0.0)
                continue
            occupied_tiles.discard(monster.tile_pos)
            update_monster(monster, runtime.player.position_px, runtime.room.walls, occupied_tiles)
            occupied_tiles.add(monster.tile_pos)
    def _resolve_monster_contact(self, result: EngineStepResult, *, shield_active: bool) -> None:
        runtime = self.runtime
        monster_to_remove: str | None = None
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
                monster.hp -= 1
                knockback_applied_px = self._apply_monster_knockback(monster)
                monster.stun_ticks_remaining = MONSTER_STUN_TICKS
                if monster.hp <= 0:
                    monster_to_remove = monster.monster_id
                    runtime.player.gold += MONSTER_KILL_GOLD_REWARD
                    runtime.last_message = f"SHIELD KILL {monster.monster_type.upper()} +{MONSTER_KILL_GOLD_REWARD}G"
                    result.events.append("monster_killed")
                    result.event_details.append(
                        {
                            "type": "monster_killed",
                            "monster_id": monster.monster_id,
                            "monster_type": monster.monster_type,
                            "gold_reward": MONSTER_KILL_GOLD_REWARD,
                            "killed_by": "shield",
                        }
                    )
                else:
                    runtime.last_message = f"SHIELD BLOCK ({monster.hp}HP LEFT)"
                    result.events.append("action_shield")
                    result.event_details.append(
                        {
                            "type": "action_shield",
                            "monster_id": monster.monster_id,
                            "damage_prevented": monster.damage,
                            "monster_hp_remaining": monster.hp,
                            "monster_knockback_px": MONSTER_HIT_KNOCKBACK_PX,
                            "knockback_applied_px": knockback_applied_px,
                            "monster_stun_ticks": MONSTER_STUN_TICKS,
                        }
                    )
                break

            runtime.player.health = max(0, runtime.player.health - monster.damage)
            monster.hp -= 1
            knockback_applied_px = self._apply_monster_knockback(monster)
            monster.stun_ticks_remaining = MONSTER_STUN_TICKS
            if monster.hp <= 0:
                monster_to_remove = monster.monster_id
                runtime.player.gold += MONSTER_KILL_GOLD_REWARD
                runtime.last_message = f"KILLED {monster.monster_type.upper()} +{MONSTER_KILL_GOLD_REWARD}G"
                result.events.append("monster_killed")
                result.event_details.append(
                    {
                        "type": "monster_killed",
                        "monster_id": monster.monster_id,
                        "monster_type": monster.monster_type,
                        "gold_reward": MONSTER_KILL_GOLD_REWARD,
                        "damage_taken": monster.damage,
                    }
                )
            else:
                runtime.last_message = f"HIT -{monster.damage}HP ({monster.hp}HP LEFT)"
                result.events.append("agent_damaged")
                result.events.append("monster_damaged")
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

        if monster_to_remove is not None:
            del runtime.room.monsters[monster_to_remove]
            self._unlock_all_monster_defeated_exits(result)

    def _remove_defeated_monster(
        self,
        monster: MonsterState,
        result: EngineStepResult,
        *,
        killed_by: str,
    ) -> None:
        runtime = self.runtime
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
        self._unlock_all_monster_defeated_exits(result)

    def _unlock_all_monster_defeated_exits(self, result: EngineStepResult) -> None:
        runtime = self.runtime
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

    def _step_made_progress(
        self,
        start_pos: tuple[float, float] | None,
        start_room_id: str | None,
        events: list[str],
    ) -> bool:
        runtime = self.runtime
        if start_pos is not None and runtime.player.position_px != start_pos:
            return True
        if start_room_id is not None and runtime.room.room_id != start_room_id:
            return True
        progress_events = {
            "door_opened",
            "key_collected",
            "gold_collected",
            "item_collected",
            "agent_healed",
            "monster_killed",
            "chest_opened",
            "button_pressed",
            "room_changed",
            "exit_reached",
            "talked_npc",
            "environment_completed",
        }
        return any(event in progress_events for event in events)

    def _all_chests_opened(self) -> bool:
        total_chests = 0
        for coord in self.room_manager.room_templates:
            room = self.room_manager.get_room(coord)
            for chest in room.chests.values():
                total_chests += 1
                if not chest.is_open:
                    return False
        return total_chests > 0

    def _apply_monster_knockback(self, monster: MonsterState) -> float:
        runtime = self.runtime
        knockback_dx, knockback_dy = self._monster_knockback_vector(monster)
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

        self._apply_player_separation(monster)
        monster.last_move_delta_px = (0.0, 0.0)
        return 0.0

    def _apply_player_separation(self, monster: MonsterState) -> None:
        runtime = self.runtime
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

    def _monster_knockback_vector(self, monster: MonsterState) -> tuple[float, float]:
        runtime = self.runtime
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

    def _player_is_flush_with_edge(self, direction: str) -> bool:
        runtime = self.runtime
        epsilon = 1e-6
        left, top = runtime.player.position_px
        if direction == "left":
            return left <= epsilon
        if direction == "right":
            return left >= MAP_PIXEL_WIDTH - runtime.player.size_px - epsilon
        if direction == "up":
            return top <= epsilon
        return top >= MAP_PIXEL_HEIGHT - runtime.player.size_px - epsilon

    @staticmethod
    def _within_map_bounds(position_px: tuple[float, float]) -> bool:
        return (
            0.0 <= position_px[0] <= MAP_PIXEL_WIDTH - TILE_SIZE
            and 0.0 <= position_px[1] <= MAP_PIXEL_HEIGHT - TILE_SIZE
        )
