from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .constants import (
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
    MOVE_ACTION_TO_DIRECTION,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
)
from .entities import (
    PlayerState,
    aabb_overlap,
    entity_center_px,
    inventory_item_codes,
    is_adjacent,
    move_with_tile_collisions,
    tile_from_position_px,
    tile_to_top_left_px,
)
from .monsters import update_monster
from .observation import room_observation
from .renderer import render_frame
from .room import RoomManager, RoomState, TransitionConfig


class DungeonEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        room_file: str | Path,
        render_mode: str | None = None,
        auto_reset_on_step: bool = True,
    ):
        super().__init__()
        self.room_manager = RoomManager(room_file)
        self.render_mode = render_mode
        self.auto_reset_on_step = auto_reset_on_step
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
        return self._get_obs(), self._get_info(events=["reset"])

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), "Invalid action!"

        auto_reset = False
        if self.pending_reset and self.auto_reset_on_step:
            self.reset()
            auto_reset = True
        elif self.pending_reset:
            raise RuntimeError("Episode terminated. Call reset() before step().")

        self.step_count += 1
        events: list[str] = []
        reward = 0.0
        terminated = False
        truncated = False

        move_direction = MOVE_ACTION_TO_DIRECTION.get(action)
        if move_direction is not None:
            reward += self._handle_move(move_direction, events)
        elif action == ACTION_A:
            self.last_message = "INTERACT"
            events.append("action_a")
            reward += self._handle_action_a(events)
        elif action == ACTION_B:
            self.last_message = "B DEFEND"
            events.append("action_b")
        elif action == ACTION_NOOP:
            self.last_message = "WAIT"
            events.append("noop")

        if self.player.health > 0 and move_direction is not None:
            reward += self._resolve_transition(move_direction, events)
        if self.player.health > 0:
            reward += self._resolve_tile_effects(events)
        if self.player.health > 0:
            reward += self._update_monsters(events)
        if self.player.health > 0:
            reward += self._resolve_monster_contact(events)

        if self.player.health <= 0:
            terminated = True
            self.pending_reset = True
            self.last_message = "GAME OVER"
            events.append("game_over")

        observation = self._get_obs()
        info = self._get_info(events=events, auto_reset=auto_reset)
        if terminated:
            info["game_over"] = True
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return render_frame(self.room, self.player)

    def close(self) -> None:
        return None

    def hud_lines(self) -> tuple[str, str]:
        room_text = f"ROOM {self.room.room_id}  HP {self.player.health}/{self.player.max_health}"
        status = (
            f"G {self.player.gold}  K {self.player.keys}  "
            f"A {self.player.action_a_label}  B {self.player.action_b_label}"
        )
        return room_text, f"{status}  {self.last_message}"

    def _handle_move(self, direction: str, events: list[str]) -> float:
        dx, dy = {
            "up": (0.0, -self.player.speed_px_per_step),
            "down": (0.0, self.player.speed_px_per_step),
            "left": (-self.player.speed_px_per_step, 0.0),
            "right": (self.player.speed_px_per_step, 0.0),
        }[direction]
        previous_position = self.player.position_px
        proposed_position = (
            previous_position[0] + dx,
            previous_position[1] + dy,
        )
        self.player.position_px = move_with_tile_collisions(
            previous_position,
            self.player.size_px,
            (dx, dy),
            self.room.blocking_tiles(),
        )

        if self.player.position_px == previous_position:
            if not self._within_map_bounds(proposed_position):
                self.last_message = "EDGE BLOCKED"
                events.append("blocked_bounds")
            else:
                self.last_message = "BLOCKED"
                events.append("blocked_wall")
            return -0.02

        self.last_message = f"MOVE {direction.upper()}"
        events.append(f"move_{direction}")
        return -0.01

    def _resolve_transition(self, direction: str, events: list[str]) -> float:
        player_tile = self._player_tile()
        transition = self.room.transition_at(player_tile, direction)
        if transition is None or not self._player_is_flush_with_edge(direction):
            return 0.0
        return self._apply_transition(transition, events)

    def _apply_transition(self, transition: TransitionConfig, events: list[str]) -> float:
        if transition.requires_key > self.player.keys:
            self.last_message = transition.locked_message
            events.append("blocked_locked")
            return -0.02

        if transition.requires_key > 0:
            self.player.keys -= transition.requires_key
            events.append("used_key")

        self.room_coord = self.room_manager.coord_for_room_id(transition.target_room_id)
        self.room = self.room_manager.get_room(self.room_coord)
        spawn_tile = self.room.spawns[transition.target_spawn]
        self.player.position_px = tile_to_top_left_px(spawn_tile)
        self.last_message = transition.success_message
        events.append("room_transition")
        return 0.1

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
            occupied_tiles.discard(monster.tile_pos)
            update_monster(monster, self.player.position_px, self.room.walls, occupied_tiles)
            occupied_tiles.add(monster.tile_pos)
        if self.room.monsters:
            events.append("monsters_updated")
        return reward

    def _resolve_monster_contact(self, events: list[str]) -> float:
        for monster in self.room.monsters.values():
            if aabb_overlap(
                self.player.position_px,
                self.player.size_px,
                monster.position_px,
                monster.size_px,
            ):
                self.player.health = max(0, self.player.health - monster.damage)
                self.last_message = f"HIT -{monster.damage}HP"
                events.append("monster_hit")
                return -0.4
        return 0.0

    def _get_obs(self) -> dict[str, np.ndarray]:
        grid = room_observation(self.room, self.player)
        player_tile = self._player_tile()
        monster_positions = np.full((self.max_monster_slots, 2), -1.0, dtype=np.float32)
        monster_tiles = np.full((self.max_monster_slots, 2), -1, dtype=np.int32)
        monster_mask = np.zeros((self.max_monster_slots,), dtype=np.uint8)

        for index, monster in enumerate(self.room.monsters.values()):
            if index >= self.max_monster_slots:
                break
            monster_positions[index] = np.asarray(monster.position_px, dtype=np.float32)
            monster_tiles[index] = np.asarray(monster.tile_pos, dtype=np.int32)
            monster_mask[index] = 1

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
        }

    def _get_info(self, *, events: list[str], auto_reset: bool = False) -> dict[str, Any]:
        info = {
            "room_id": self.room.room_id,
            "room_coord": self.room.coord,
            "health": self.player.health,
            "gold": self.player.gold,
            "keys": self.player.keys,
            "items": list(self.player.items),
            "message": self.last_message,
            "events": events,
            "episode": self.episode,
            "step": self.step_count,
            "player_position_px": self.player.position_px,
            "player_tile": self._player_tile(),
        }
        if auto_reset:
            info["auto_reset"] = True
        return info

    def _player_tile(self) -> tuple[int, int]:
        return tile_from_position_px(self.player.position_px, self.player.size_px)

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
