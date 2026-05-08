from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.constants import (
    ACTION_A,
    ACTION_B,
    ACTION_LABELS,
    ACTION_NOOP,
    GRID_HEIGHT,
    GRID_WIDTH,
    ITEM_NAME_TO_ID,
    MAP_PIXEL_HEIGHT,
    MAP_PIXEL_WIDTH,
    MESSAGE_DEFAULT,
    MONSTER_HIT_KNOCKBACK_PX,
    MONSTER_KILL_GOLD_REWARD,
    MONSTER_STUN_TICKS,
    MOVE_ACTION_TO_DIRECTION,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    TARGET_FPS,
)
from ..entities import (
    EquipmentSlot,
    PlayerState,
    ToolType,
    aabb_overlap,
    entity_center_px,
    inventory_item_codes,
    is_adjacent,
    move_with_tile_collisions,
    tile_from_position_px,
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
from ..rendering.renderer import render_frame
from .observation import room_observation


MOVE_TO_EXIT_DIRECTION = {
    "up": "north",
    "down": "south",
    "left": "west",
    "right": "east",
}


class DungeonEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": TARGET_FPS}

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = True,
        move_speed_px: int = 4,
        stuck_penalty_enabled: bool = False,
        stuck_penalty_steps: int = 30,
        stuck_penalty: float = -0.01,
    ):
        super().__init__()
        self.room_manager = RoomManager(room_file)
        self.render_mode = render_mode
        self.auto_reset_on_step = auto_reset_on_step
        self.move_speed_px = max(1, int(move_speed_px))
        self.stuck_penalty_enabled = bool(stuck_penalty_enabled)
        self.stuck_penalty_steps = max(1, int(stuck_penalty_steps))
        self.stuck_penalty = float(stuck_penalty)
        self.no_progress_steps = 0
        self.max_monster_slots = max(1, self.room_manager.max_monsters)

        self.action_space = spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=8, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8),
                "player_position_px": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([MAP_PIXEL_WIDTH - 1.0, MAP_PIXEL_HEIGHT - 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "player_tile": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([GRID_WIDTH - 1, GRID_HEIGHT - 1], dtype=np.int32),
                    dtype=np.int32,
                ),
                "health": spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
                "gold": spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32),
                "keys": spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
                "inventory_ids": spaces.Box(
                    low=0,
                    high=max(ITEM_NAME_TO_ID.values()),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "monsters_position_px": spaces.Box(
                    low=-1.0,
                    high=max(SCREEN_WIDTH, SCREEN_HEIGHT),
                    shape=(self.max_monster_slots, 2),
                    dtype=np.float32,
                ),
                "monsters_tile": spaces.Box(
                    low=-1,
                    high=max(GRID_WIDTH, GRID_HEIGHT),
                    shape=(self.max_monster_slots, 2),
                    dtype=np.int32,
                ),
                "monsters_active_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_monster_slots,),
                    dtype=np.uint8,
                ),
                "monsters_hp": spaces.Box(
                    low=0,
                    high=99,
                    shape=(self.max_monster_slots,),
                    dtype=np.int32,
                ),
            }
        )

        self.room_coord: tuple[int, int] = self.room_manager.start_room
        self.room: RoomState = self.room_manager.get_room(self.room_coord)
        self.player = PlayerState(position_px=tile_to_top_left_px(self.room.spawns[self.room.default_spawn_name]))
        self.episode = 0
        self.step_count = 0
        self.pending_reset = False
        self.last_message = MESSAGE_DEFAULT

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.room_manager.reset_room_cache()
        self.room_coord = self.room_manager.start_room
        self.room = self.room_manager.get_room(self.room_coord)
        self.player = PlayerState(position_px=tile_to_top_left_px(self.room.spawns[self.room.default_spawn_name]))
        self.pending_reset = False
        self.last_message = MESSAGE_DEFAULT
        self.step_count = 0
        self.episode += 1
        self.no_progress_steps = 0
        return self._get_obs(), self._get_info(events=["reset"], event_details=[])

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), "Invalid action!"

        auto_reset = False
        if self.pending_reset and self.auto_reset_on_step:
            self.reset()
            auto_reset = True
        elif self.pending_reset:
            raise RuntimeError("Episode terminated. Call reset() before step().")

        progress_start_pos = self.player.position_px
        progress_start_room = self.room.room_id
        self.step_count += 1
        events: list[str] = []
        event_details: list[dict[str, Any]] = []
        reward = 0.0
        terminated = False
        truncated = False
        shield_active = False

        move_direction = MOVE_ACTION_TO_DIRECTION.get(action)
        if move_direction is not None:
            reward += self._handle_move(move_direction, events)
        elif action == ACTION_A:
            events.append("action_a")
            action_reward, shield_active = self._handle_equipped_action(EquipmentSlot.A, events)
            reward += action_reward
        elif action == ACTION_B:
            events.append("action_b")
            action_reward, shield_active = self._handle_equipped_action(EquipmentSlot.B, events)
            reward += action_reward
        elif action == ACTION_NOOP:
            self.last_message = "WAIT"
            events.append("noop")

        if self.player.health > 0 and move_direction is not None:
            reward += self._resolve_transition(move_direction, events, event_details)
        if self.player.health > 0:
            reward += self._resolve_tile_effects(events)
        if self.player.health > 0:
            reward += self._update_monsters(events)
        if self.player.health > 0:
            reward += self._resolve_monster_contact(events, event_details, shield_active=shield_active)

        if self.player.health <= 0:
            terminated = True
            self.pending_reset = True
            self.last_message = "GAME OVER"
            events.append("game_over")
        elif self._all_chests_opened():
            terminated = True
            self.pending_reset = True
            self.last_message = "VICTORY"
            events.append("victory")
            reward += 1.0

        if self._step_made_progress(progress_start_pos, progress_start_room, events):
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
            if self.stuck_penalty_enabled and self.no_progress_steps >= self.stuck_penalty_steps:
                reward += self.stuck_penalty

        observation = self._get_obs()
        info = self._get_info(events=events, event_details=event_details, auto_reset=auto_reset)
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return render_frame(self.room, self.player)

    def close(self) -> None:
        return None

    def hud_lines(self) -> tuple[str, str]:
        room_text = f"R:{self.room.room_id} HP:{self.player.health} G:{self.player.gold}"
        items = ",".join(self.player.items) if self.player.items else "-"
        equipment = (
            f"A:{self.player.equipped_tool_label(EquipmentSlot.A.value)} "
            f"B:{self.player.equipped_tool_label(EquipmentSlot.B.value)}"
        )
        return room_text, f"I:{items} {equipment}"

    def _handle_move(self, direction: str, events: list[str]) -> float:
        step_dx, step_dy = {
            "up": (0.0, -1.0),
            "down": (0.0, 1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
        }[direction]
        previous_position = self.player.position_px
        current_position = previous_position
        blocked_position: tuple[float, float] | None = None
        for _ in range(self.move_speed_px):
            proposed_position = (
                current_position[0] + step_dx,
                current_position[1] + step_dy,
            )
            next_position = move_with_tile_collisions(
                current_position,
                self.player.size_px,
                (step_dx, step_dy),
                self.room.blocking_tiles(),
            )
            if next_position == current_position:
                blocked_position = proposed_position
                break
            current_position = next_position

        self.player.position_px = current_position

        if self.player.position_px == previous_position:
            if blocked_position is not None and not self._within_map_bounds(blocked_position):
                self.last_message = "EDGE BLOCKED"
                events.append("blocked_bounds")
            else:
                self.last_message = "BLOCKED"
                events.append("blocked_wall")
            return -0.02

        self.last_message = f"MOVE {direction.upper()}"
        events.append(f"move_{direction}")
        return -0.01

    def _resolve_transition(
        self,
        direction: str,
        events: list[str],
        event_details: list[dict[str, Any]],
    ) -> float:
        player_tile = self._player_tile()
        exit_direction = MOVE_TO_EXIT_DIRECTION[direction]
        exit_config = self.room.exit_at(player_tile, exit_direction)
        if exit_config is None or not self._player_is_flush_with_edge(direction):
            return 0.0
        return self._apply_exit(exit_config, events, event_details)

    def _apply_exit(
        self,
        exit_config: ExitConfig,
        events: list[str],
        event_details: list[dict[str, Any]],
    ) -> float:
        allowed, blocked_event = self._can_use_exit(exit_config)
        if not allowed:
            self.last_message = exit_config.blocked_message
            events.append(blocked_event)
            return -0.02

        from_room_id = self.room.room_id
        exit_state = self.room.exit_state(exit_config)
        if exit_config.exit_type == "locked_key" and not exit_state.unlocked:
            key_consumed = False
            if bool(exit_config.requires.get("consume_key", False)):
                self.player.keys -= int(exit_config.requires.get("key_count", 1))
                events.append("used_key")
                key_consumed = True
            exit_state.unlocked = True
            exit_state.opened = True
            events.append("door_unlocked")
            event_details.append(
                {
                    "type": "door_unlocked",
                    "room_id": from_room_id,
                    "direction": exit_config.direction,
                    "key_consumed": key_consumed,
                }
            )

        target_coord = self.room_manager.coord_for_room_id(exit_config.target_room_id)
        target_room = self.room_manager.get_room(target_coord)
        spawn_tile = self._entry_spawn_tile(target_room, exit_config.target_entry)
        self.room_coord = target_coord
        self.room = target_room
        self.player.position_px = tile_to_top_left_px(spawn_tile)
        self.last_message = exit_config.success_message
        events.append("room_transition")
        event_details.append(
            {
                "type": "room_transition",
                "from_room": from_room_id,
                "to_room": self.room.room_id,
                "exit_direction": exit_config.direction,
                "target_entry": exit_config.target_entry,
                "spawn_px": [self.player.position_px[0], self.player.position_px[1]],
            }
        )
        return 0.1

    def _can_use_exit(self, exit_config: ExitConfig) -> tuple[bool, str]:
        if exit_config.exit_type == "normal":
            return True, ""
        if exit_config.exit_type == "locked_key":
            if self.room.exit_state(exit_config).unlocked:
                return True, ""
            required_keys = int(exit_config.requires.get("key_count", 1))
            if self.player.keys < required_keys:
                return False, "blocked_locked"
            return True, ""

        button_id = exit_config.requires.get("button_pressed")
        if button_id is not None:
            button = self.room.buttons.get(button_id)
            if button is None or not button.is_pressed:
                return False, "missing_requirement"
        item_name = exit_config.requires.get("item")
        if item_name is not None and item_name not in self.player.items:
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

    def _handle_equipped_action(self, slot: EquipmentSlot, events: list[str]) -> tuple[float, bool]:
        tool = self.player.equipped_tool(slot)
        if slot == EquipmentSlot.A and tool == ToolType.INTERACT.value:
            self.last_message = "INTERACT"
            return self._handle_action_a(events), False
        if slot == EquipmentSlot.B and tool == ToolType.SHIELD.value:
            self.last_message = "SHIELD"
            events.append("shield")
            return 0.0, True

        self.last_message = f"{slot.value} NO EFFECT"
        events.append(f"action_{slot.value.lower()}_empty")
        return -0.01, False

    def _handle_action_a(self, events: list[str]) -> float:
        player_tile = self._player_tile()

        for chest in self.room.chests.values():
            if not chest.is_open and is_adjacent(player_tile, chest.pos):
                chest.is_open = True
                reward = self._apply_loot(chest.loot, events)
                events.append("opened_chest")
                return reward

        for npc in self.room.npcs.values():
            if is_adjacent(player_tile, npc.pos):
                self.last_message = npc.text.upper()[:24]
                events.append("talked_npc")
                return 0.0

        self.last_message = "A NO EFFECT"
        events.append("action_a_empty")
        return -0.01

    def _apply_loot(self, loot: dict[str, Any], events: list[str]) -> float:
        loot_kind = str(loot.get("kind", "gold"))
        amount = int(loot.get("amount", 1))

        if loot_kind == "key":
            self.player.keys += max(1, amount)
            self.last_message = "GOT KEY"
            events.append("got_key")
            return 0.4
        if loot_kind == "heal":
            healed = min(self.player.max_health, self.player.health + max(1, amount))
            actual = healed - self.player.health
            self.player.health = healed
            self.last_message = "HEALED"
            events.append("healed")
            return 0.2 if actual > 0 else 0.05
        if loot_kind == "item":
            item_name = str(loot.get("item_id", "item"))
            if item_name not in self.player.items:
                self.player.items.append(item_name)
            self.last_message = f"GOT {item_name}".upper()[:24]
            events.append("got_item")
            return 0.3

        self.player.gold += max(1, amount)
        self.last_message = "GOT GOLD"
        events.append("got_gold")
        return 0.2

    def _resolve_tile_effects(self, events: list[str]) -> float:
        reward = 0.0
        player_tile = self._player_tile()

        button = self.room.button_at(player_tile)
        if button is not None and not button.is_pressed:
            button.is_pressed = True
            self.last_message = button.message.upper()[:24]
            events.append("pressed_button")
            reward += 0.1

        trap = self.room.trap_at(player_tile)
        if trap is not None:
            self.player.health = max(0, self.player.health - trap.damage)
            respawn_name = trap.respawn_to if trap.respawn_to in self.room.spawns else self.room.default_spawn_name
            if self.player.health > 0:
                self.player.position_px = tile_to_top_left_px(self.room.spawns[respawn_name])
            self.last_message = f"TRAP -{trap.damage}HP"
            events.append("trap_damage")
            reward -= 0.5
            if trap.single_use:
                trap.is_active = False

        return reward

    def _update_monsters(self, events: list[str]) -> float:
        reward = 0.0
        occupied_tiles = {monster.tile_pos for monster in self.room.monsters.values()}
        for monster in self.room.monsters.values():
            if monster.stun_ticks_remaining > 0:
                monster.stun_ticks_remaining -= 1
                monster.last_move_delta_px = (0.0, 0.0)
                continue
            occupied_tiles.discard(monster.tile_pos)
            update_monster(monster, self.player.position_px, self.room.walls, occupied_tiles)
            occupied_tiles.add(monster.tile_pos)
        if self.room.monsters:
            events.append("monsters_updated")
        return reward

    def _resolve_monster_contact(
        self,
        events: list[str],
        event_details: list[dict[str, Any]],
        *,
        shield_active: bool = False,
    ) -> float:
        reward = 0.0
        monster_to_remove: str | None = None
        for monster in self.room.monsters.values():
            if monster.stun_ticks_remaining > 0:
                continue
            if not aabb_overlap(
                self.player.position_px,
                self.player.size_px,
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
                    self.player.gold += MONSTER_KILL_GOLD_REWARD
                    self.last_message = f"SHIELD KILL {monster.monster_type.upper()} +{MONSTER_KILL_GOLD_REWARD}G"
                    events.append("monster_killed")
                    event_details.append(
                        {
                            "type": "monster_killed",
                            "monster_id": monster.monster_id,
                            "monster_type": monster.monster_type,
                            "gold_reward": MONSTER_KILL_GOLD_REWARD,
                            "killed_by": "shield",
                        }
                    )
                    reward += 0.3
                else:
                    self.last_message = f"SHIELD BLOCK ({monster.hp}HP LEFT)"
                    events.append("shield_block")
                    event_details.append(
                        {
                            "type": "shield_block",
                            "monster_id": monster.monster_id,
                            "damage_prevented": monster.damage,
                            "monster_hp_remaining": monster.hp,
                            "monster_knockback_px": MONSTER_HIT_KNOCKBACK_PX,
                            "knockback_applied_px": knockback_applied_px,
                            "monster_stun_ticks": MONSTER_STUN_TICKS,
                        }
                    )
                break
            # Player takes damage from monster
            self.player.health = max(0, self.player.health - monster.damage)
            # Monster also takes damage from contact
            monster.hp -= 1
            knockback_applied_px = self._apply_monster_knockback(monster)
            monster.stun_ticks_remaining = MONSTER_STUN_TICKS
            reward -= 0.4
            if monster.hp <= 0:
                # Monster killed
                monster_to_remove = monster.monster_id
                self.player.gold += MONSTER_KILL_GOLD_REWARD
                self.last_message = f"KILLED {monster.monster_type.upper()} +{MONSTER_KILL_GOLD_REWARD}G"
                events.append("monster_killed")
                event_details.append(
                    {
                        "type": "monster_killed",
                        "monster_id": monster.monster_id,
                        "monster_type": monster.monster_type,
                        "gold_reward": MONSTER_KILL_GOLD_REWARD,
                        "damage_taken": monster.damage,
                    }
                )
                reward += 0.3
            else:
                self.last_message = f"HIT -{monster.damage}HP ({monster.hp}HP LEFT)"
                events.append("monster_hit")
                events.append("monster_damaged")
                event_details.append(
                    {
                        "type": "monster_collision",
                        "monster_id": monster.monster_id,
                        "damage": monster.damage,
                        "monster_hp_remaining": monster.hp,
                        "monster_knockback_px": MONSTER_HIT_KNOCKBACK_PX,
                        "knockback_applied_px": knockback_applied_px,
                        "monster_stun_ticks": MONSTER_STUN_TICKS,
                    }
                )
            # Only process one collision per step
            break
        if monster_to_remove is not None:
            del self.room.monsters[monster_to_remove]
        return reward

    def _get_obs(self) -> dict[str, np.ndarray]:
        grid = room_observation(self.room, self.player)
        player_tile = self._player_tile()
        monster_positions = np.full((self.max_monster_slots, 2), -1.0, dtype=np.float32)
        monster_tiles = np.full((self.max_monster_slots, 2), -1, dtype=np.int32)
        monster_mask = np.zeros((self.max_monster_slots,), dtype=np.uint8)
        monster_hp = np.zeros((self.max_monster_slots,), dtype=np.int32)

        for index, monster in enumerate(self.room.monsters.values()):
            if index >= self.max_monster_slots:
                break
            monster_positions[index] = np.asarray(monster.position_px, dtype=np.float32)
            monster_tiles[index] = np.asarray(monster.tile_pos, dtype=np.int32)
            monster_mask[index] = 1
            monster_hp[index] = monster.hp

        return {
            "grid": grid,
            "player_position_px": np.asarray(self.player.position_px, dtype=np.float32),
            "player_tile": np.asarray(player_tile, dtype=np.int32),
            "health": np.asarray([self.player.health], dtype=np.int32),
            "gold": np.asarray([self.player.gold], dtype=np.int32),
            "keys": np.asarray([self.player.keys], dtype=np.int32),
            "inventory_ids": np.asarray(inventory_item_codes(self.player.items), dtype=np.int32),
            "monsters_position_px": monster_positions,
            "monsters_tile": monster_tiles,
            "monsters_active_mask": monster_mask,
            "monsters_hp": monster_hp,
        }

    def _step_made_progress(
        self,
        start_pos: tuple[float, float],
        start_room_id: str,
        events: list[str],
    ) -> bool:
        if self.player.position_px != start_pos:
            return True
        if self.room.room_id != start_room_id:
            return True
        progress_events = {
            "door_unlocked",
            "got_key",
            "got_gold",
            "got_item",
            "healed",
            "monster_killed",
            "opened_chest",
            "pressed_button",
            "room_transition",
            "talked_npc",
            "victory",
        }
        return any(event in progress_events for event in events)

    def _get_info(
        self,
        *,
        events: list[str],
        event_details: list[dict[str, Any]],
        auto_reset: bool = False,
    ) -> dict[str, Any]:
        info = {
            "room_id": self.room.room_id,
            "room_coord": self.room.coord,
            "health": self.player.health,
            "gold": self.player.gold,
            "keys": self.player.keys,
            "items": list(self.player.items),
            "tools": list(self.player.tools),
            "equipped": dict(self.player.equipped),
            "message": self.last_message,
            "events": events,
            "event_details": event_details,
            "episode": self.episode,
            "step": self.step_count,
            "player_position_px": self.player.position_px,
            "player_tile": self._player_tile(),
            "agent_pos": self.player.position_px,
            "has_key": self.player.keys > 0,
            "key_count": self.player.keys,
            "picked_key": "got_key" in events,
            "unlocked_door": "door_unlocked" in events,
            "entered_new_room": "room_transition" in events,
            "task_success": "victory" in events,
            "no_progress_steps": self.no_progress_steps,
        }
        if auto_reset:
            info["auto_reset"] = True
        if "victory" in events:
            info["victory"] = True
        if "game_over" in events:
            info["game_over"] = True
        return info

    def _all_chests_opened(self) -> bool:
        total_chests = 0
        for coord in self.room_manager.room_templates:
            room = self.room_manager.get_room(coord)
            for chest in room.chests.values():
                total_chests += 1
                if not chest.is_open:
                    return False
        return total_chests > 0

    def _player_tile(self) -> tuple[int, int]:
        return tile_from_position_px(self.player.position_px, self.player.size_px)

    def _apply_monster_knockback(self, monster: MonsterState) -> float:
        knockback_dx, knockback_dy = self._monster_knockback_vector(monster)
        other_monster_tiles = {
            other.tile_pos
            for other in self.room.monsters.values()
            if other.monster_id != monster.monster_id
        }
        world_blockers = self.room.blocking_tiles() | other_monster_tiles
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

        # Fallback: monster couldn't be knocked back (stuck against wall/edge).
        # Move the player away from the monster instead to break overlap.
        self._apply_player_separation(monster)
        monster.last_move_delta_px = (0.0, 0.0)
        return 0.0

    def _apply_player_separation(self, monster: MonsterState) -> None:
        """Move the player away from the monster when knockback fails."""
        player_center = entity_center_px(self.player.position_px, self.player.size_px)
        monster_center = entity_center_px(monster.position_px, monster.size_px)
        dx = player_center[0] - monster_center[0]
        dy = player_center[1] - monster_center[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            dx, dy = 0.0, -1.0
            distance = 1.0
        sep_dx = (dx / distance) * TILE_SIZE
        sep_dy = (dy / distance) * TILE_SIZE
        self.player.position_px = move_with_tile_collisions(
            self.player.position_px,
            self.player.size_px,
            (sep_dx, sep_dy),
            self.room.blocking_tiles(),
        )

    def _monster_knockback_vector(self, monster: MonsterState) -> tuple[float, float]:
        player_center = entity_center_px(self.player.position_px, self.player.size_px)
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
        epsilon = 1e-6
        left, top = self.player.position_px
        if direction == "left":
            return left <= epsilon
        if direction == "right":
            return left >= MAP_PIXEL_WIDTH - self.player.size_px - epsilon
        if direction == "up":
            return top <= epsilon
        return top >= MAP_PIXEL_HEIGHT - self.player.size_px - epsilon

    @staticmethod
    def _within_map_bounds(position_px: tuple[float, float]) -> bool:
        return (
            0.0 <= position_px[0] <= MAP_PIXEL_WIDTH - TILE_SIZE
            and 0.0 <= position_px[1] <= MAP_PIXEL_HEIGHT - TILE_SIZE
        )
