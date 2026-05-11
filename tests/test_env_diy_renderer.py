from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from env_diy.core.constants import (
    COLOR_EXIT_LOCKED,
    HUD_PIXEL_Y,
    INTERNAL_HEIGHT,
    INTERNAL_WIDTH,
    TILE_SIZE,
)
from env_diy.entities import ButtonState, ChestState, NPCState, PlayerState, TrapState
from env_diy.entities.monsters import MonsterState
from env_diy.envs import DungeonEnv
from env_diy.maps import ExitConfig, ExitRuntimeState, RoomState
from env_diy.rendering import render_frame
from env_diy.rendering.sprites import draw_exit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DUNGEON = (
    PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"
)


class RendererTests(unittest.TestCase):
    def test_render_canvas_size_is_160x160(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()

        frame = env.render()

        self.assertEqual(frame.shape, (INTERNAL_HEIGHT, INTERNAL_WIDTH, 3))

    def test_legacy_import_paths_still_work(self) -> None:
        from env_diy.env import DungeonEnv as FacadeDungeonEnv

        self.assertIs(FacadeDungeonEnv, DungeonEnv)

    def test_hud_area_exists_and_status_text_is_available(self) -> None:
        env = DungeonEnv(STRUCTURED_DUNGEON)
        env.reset()
        env.player.gold = 12
        env.player.items = ["key", "bow"]

        frame = env.render()
        line_1, line_2 = env.hud_lines()
        hud_area = frame[HUD_PIXEL_Y:, :]
        map_area = frame[:HUD_PIXEL_Y, :]

        self.assertIn("R:", line_1)
        self.assertIn("HP:", line_1)
        self.assertIn("G:12", line_1)
        self.assertIn("I:key,bow", line_2)
        self.assertIn("A:interact", line_2)
        self.assertIn("B:shield", line_2)
        self.assertGreater(_unique_color_count(hud_area), 3)
        self.assertFalse(np.array_equal(hud_area[:16], map_area[:16]))

    def test_player_and_monster_render_as_distinct_icons(self) -> None:
        room = _base_room()
        room.monsters["monster_1"] = MonsterState(
            monster_id="monster_1",
            monster_type="chaser",
            position_px=(48.0, 16.0),
        )
        player = PlayerState(position_px=(16.0, 16.0))

        frame = render_frame(room, player)
        player_tile = _tile_crop(frame, 1, 1)
        monster_tile = _tile_crop(frame, 3, 1)

        self.assertGreater(_unique_color_count(player_tile), 4)
        self.assertGreater(_unique_color_count(monster_tile), 4)
        self.assertFalse(np.array_equal(player_tile, monster_tile))

    def test_chest_open_state_and_loot_icons_render_differently(self) -> None:
        closed = _base_room()
        closed.chests["gold_chest"] = ChestState(
            chest_id="gold_chest",
            pos=(2, 2),
            loot={"kind": "gold", "amount": 5},
            is_open=False,
        )
        opened = _base_room()
        opened.chests["gold_chest"] = ChestState(
            chest_id="gold_chest",
            pos=(2, 2),
            loot={"kind": "gold", "amount": 5},
            is_open=True,
        )

        closed_frame = render_frame(closed, PlayerState(position_px=(16.0, 16.0)))
        opened_frame = render_frame(opened, PlayerState(position_px=(16.0, 16.0)))

        self.assertGreater(_unique_color_count(_tile_crop(closed_frame, 2, 2)), 4)
        self.assertFalse(np.array_equal(_tile_crop(closed_frame, 2, 2), _tile_crop(opened_frame, 2, 2)))

    def test_key_coin_and_heal_loot_icons_are_distinct(self) -> None:
        room = _base_room()
        room.chests["key_chest"] = ChestState("key_chest", (2, 2), {"kind": "key"})
        room.chests["coin_chest"] = ChestState("coin_chest", (4, 2), {"kind": "gold"})
        room.chests["heal_chest"] = ChestState("heal_chest", (6, 2), {"kind": "heal"})

        frame = render_frame(room, PlayerState(position_px=(16.0, 16.0)))
        key_tile = _tile_crop(frame, 2, 2)
        coin_tile = _tile_crop(frame, 4, 2)
        heal_tile = _tile_crop(frame, 6, 2)

        self.assertFalse(np.array_equal(key_tile, coin_tile))
        self.assertFalse(np.array_equal(key_tile, heal_tile))
        self.assertFalse(np.array_equal(coin_tile, heal_tile))

    def test_trap_and_button_pressed_states_render_without_crashing(self) -> None:
        room = _base_room()
        room.traps["trap_1"] = TrapState("trap_1", (2, 4))
        room.buttons["button_up"] = ButtonState("button_up", (4, 4), is_pressed=False)
        room.buttons["button_down"] = ButtonState("button_down", (6, 4), is_pressed=True)

        frame = render_frame(room, PlayerState(position_px=(16.0, 16.0)))
        trap_tile = _tile_crop(frame, 2, 4)
        unpressed_tile = _tile_crop(frame, 4, 4)
        pressed_tile = _tile_crop(frame, 6, 4)

        self.assertGreater(_unique_color_count(trap_tile), 3)
        self.assertFalse(np.array_equal(unpressed_tile, pressed_tile))
        self.assertFalse(np.array_equal(trap_tile, unpressed_tile))

    def test_exit_types_render_as_distinct_connected_two_tile_icons(self) -> None:
        room = _base_room()
        room.exits = [
            ExitConfig("west_exit", "west", ((0, 3), (0, 4)), "room_b", "from_east", "normal"),
            ExitConfig("east_exit", "east", ((9, 3), (9, 4)), "room_b", "from_west", "locked_key"),
            ExitConfig("south_exit", "south", ((4, 7), (5, 7)), "room_b", "from_north", "conditional"),
        ]

        frame = render_frame(room, PlayerState(position_px=(16.0, 16.0)))
        normal_exit = frame[3 * TILE_SIZE : 5 * TILE_SIZE, 0:TILE_SIZE]
        locked_exit = frame[3 * TILE_SIZE : 5 * TILE_SIZE, 9 * TILE_SIZE : 10 * TILE_SIZE]
        conditional_exit = frame[7 * TILE_SIZE : 8 * TILE_SIZE, 4 * TILE_SIZE : 6 * TILE_SIZE]

        self.assertGreater(_unique_color_count(normal_exit), 3)
        self.assertGreater(_unique_color_count(locked_exit), 3)
        self.assertGreater(_unique_color_count(conditional_exit), 3)
        self.assertFalse(np.array_equal(normal_exit, locked_exit))
        self.assertFalse(np.array_equal(normal_exit[:16], conditional_exit))
        self.assertFalse(np.array_equal(locked_exit[:16], conditional_exit[:, :16]))

    def test_locked_exit_opened_runtime_state_renders_differently(self) -> None:
        room = _base_room()
        exit_config = ExitConfig("east_exit", "east", ((9, 3), (9, 4)), "room_b", "west", "locked_key")
        room.exits = [exit_config]
        room.exit_states[exit_config.exit_id] = ExitRuntimeState(unlocked=False, opened=False)

        locked_frame = render_frame(room, PlayerState(position_px=(16.0, 16.0)))

        room.exit_states[exit_config.exit_id] = ExitRuntimeState(unlocked=True, opened=True)
        opened_frame = render_frame(room, PlayerState(position_px=(16.0, 16.0)))

        locked_exit = locked_frame[3 * TILE_SIZE : 5 * TILE_SIZE, 9 * TILE_SIZE : 10 * TILE_SIZE]
        opened_exit = opened_frame[3 * TILE_SIZE : 5 * TILE_SIZE, 9 * TILE_SIZE : 10 * TILE_SIZE]
        self.assertFalse(np.array_equal(locked_exit, opened_exit))

    def test_locked_opened_sprite_is_distinct_from_other_exit_types(self) -> None:
        tiles = ((4, 7), (5, 7))
        locked_frame = np.zeros((INTERNAL_HEIGHT, INTERNAL_WIDTH, 3), dtype=np.uint8)
        opened_frame = np.zeros((INTERNAL_HEIGHT, INTERNAL_WIDTH, 3), dtype=np.uint8)
        normal_frame = np.zeros((INTERNAL_HEIGHT, INTERNAL_WIDTH, 3), dtype=np.uint8)
        conditional_frame = np.zeros((INTERNAL_HEIGHT, INTERNAL_WIDTH, 3), dtype=np.uint8)

        draw_exit(locked_frame, tiles, "locked_key", COLOR_EXIT_LOCKED, opened=False)
        draw_exit(opened_frame, tiles, "locked_key", COLOR_EXIT_LOCKED, opened=True)
        draw_exit(normal_frame, tiles, "normal", COLOR_EXIT_LOCKED, opened=True)
        draw_exit(conditional_frame, tiles, "conditional", COLOR_EXIT_LOCKED, opened=True)

        self.assertFalse(np.array_equal(locked_frame, opened_frame))
        self.assertFalse(np.array_equal(opened_frame, normal_frame))
        self.assertFalse(np.array_equal(opened_frame, conditional_frame))

    def test_all_supported_room_entities_render_headless(self) -> None:
        room = _base_room()
        room.walls = {(0, 0)}
        room.chests["closed_chest"] = ChestState("closed_chest", (2, 1), {"kind": "key"})
        room.chests["open_chest"] = ChestState("open_chest", (3, 1), {"kind": "heal"}, is_open=True)
        room.npcs["npc_1"] = NPCState("npc_1", (4, 1), "hello")
        room.traps["trap_1"] = TrapState("trap_1", (5, 1))
        room.buttons["button_up"] = ButtonState("button_up", (6, 1), is_pressed=False)
        room.buttons["button_down"] = ButtonState("button_down", (7, 1), is_pressed=True)
        room.monsters["chaser"] = MonsterState("chaser", "chaser", (32.0, 64.0))
        room.monsters["ambusher"] = MonsterState("ambusher", "ambusher", (64.0, 64.0))
        room.monsters["patroller"] = MonsterState("patroller", "patroller", (96.0, 64.0))
        room.exits = [
            ExitConfig("west_exit", "west", ((0, 3), (0, 4)), "room_b", "from_east", "normal"),
            ExitConfig("east_exit", "east", ((9, 3), (9, 4)), "room_b", "from_west", "locked_key"),
            ExitConfig("south_exit", "south", ((4, 7), (5, 7)), "room_b", "from_north", "conditional"),
        ]

        frame = render_frame(room, PlayerState(position_px=(16.0, 16.0)))

        self.assertEqual(frame.shape, (INTERNAL_HEIGHT, INTERNAL_WIDTH, 3))
        self.assertGreater(_unique_color_count(frame), 20)


def _base_room() -> RoomState:
    return RoomState(
        room_id="room_test",
        coord=(0, 0),
        width=10,
        height=8,
        spawns={"default": (1, 1)},
        default_spawn_name="default",
        walls=set(),
        chests={},
        npcs={},
        traps={},
        buttons={},
        monsters={},
        exits=[],
    )


def _tile_crop(frame: np.ndarray, col: int, row: int) -> np.ndarray:
    top = row * TILE_SIZE
    left = col * TILE_SIZE
    return frame[top : top + TILE_SIZE, left : left + TILE_SIZE]


def _unique_color_count(frame: np.ndarray) -> int:
    return len(np.unique(frame.reshape(-1, 3), axis=0))


if __name__ == "__main__":
    unittest.main()
