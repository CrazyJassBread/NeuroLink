from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from env_diy.core.constants import GRID_HEIGHT, GRID_WIDTH
from env_diy.maps import MapValidationError, RoomManager, exit_tiles_for_direction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DUNGEON = (
    PROJECT_ROOT / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"
)


class RoomManagerLoadingTests(unittest.TestCase):
    def test_structured_dungeon_loads(self) -> None:
        manager = RoomManager(STRUCTURED_DUNGEON)
        room = manager.get_room((0, 0))

        self.assertEqual(manager.start_room, (0, 0))
        self.assertEqual(manager.start_room_id, "room_0_0")
        self.assertEqual(room.width, GRID_WIDTH)
        self.assertEqual(room.height, GRID_HEIGHT)
        self.assertEqual(len(room.exits), 3)
        self.assertTrue(any(exit_config.exit_type == "locked_key" for exit_config in room.exits))
        self.assertTrue(any(exit_config.exit_type == "conditional" for exit_config in room.exits))

    def test_exit_tiles_match_fixed_two_tile_rules(self) -> None:
        self.assertEqual(exit_tiles_for_direction("north"), ((4, 0), (5, 0)))
        self.assertEqual(exit_tiles_for_direction("south"), ((4, 7), (5, 7)))
        self.assertEqual(exit_tiles_for_direction("west"), ((0, 3), (0, 4)))
        self.assertEqual(exit_tiles_for_direction("east"), ((9, 3), (9, 4)))

    def test_reset_room_cache_rebuilds_state(self) -> None:
        manager = RoomManager(STRUCTURED_DUNGEON)
        room = manager.get_room((0, 0))
        first_chest = next(iter(room.chests.values()))
        first_chest.is_open = True

        manager.reset_room_cache()
        rebuilt = manager.get_room((0, 0))
        rebuilt_chest = next(iter(rebuilt.chests.values()))

        self.assertIsNot(room, rebuilt)
        self.assertFalse(rebuilt_chest.is_open)

    def test_row_8_entity_rejected_as_hud_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dungeon_path = self._write_dungeon(
                root,
                start_room="room_a",
                rooms=[
                    {
                        "file": "rooms/room_a.json",
                        "payload": {
                            "id": "room_a",
                            "coord": [0, 0],
                            "layout": self._empty_layout(),
                            "spawns": {"default": [1, 1]},
                            "objects": [
                                {"id": "npc_1", "kind": "npc", "pos": [1, 8], "text": "bad"}
                            ],
                        },
                    }
                ],
            )

            with self.assertRaises(MapValidationError) as ctx:
                RoomManager(dungeon_path)

        self.assertIn("objects[0].pos", str(ctx.exception))
        self.assertIn("HUD", str(ctx.exception))

    def test_invalid_exit_target_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dungeon_path = self._write_dungeon(
                root,
                start_room="room_a",
                rooms=[
                    {
                        "file": "rooms/room_a.json",
                        "payload": {
                            "id": "room_a",
                            "coord": [0, 0],
                            "layout": self._empty_layout(),
                            "spawns": {"default": [0, 4]},
                            "exits": [
                                {
                                    "id": "west_exit",
                                    "direction": "west",
                                    "target_room": "missing_room",
                                    "target_entry": "default",
                                    "type": "normal",
                                }
                            ],
                        },
                    }
                ],
            )

            with self.assertRaises(MapValidationError) as ctx:
                RoomManager(dungeon_path)

        self.assertIn("target_room", str(ctx.exception))
        self.assertIn("missing_room", str(ctx.exception))

    def test_missing_target_entry_defaults_to_opposite_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dungeon_path = self._write_dungeon(
                root,
                start_room="room_a",
                rooms=[
                    {
                        "file": "rooms/room_a.json",
                        "payload": {
                            "id": "room_a",
                            "coord": [0, 0],
                            "layout": self._empty_layout(),
                            "spawns": {"default": [1, 1]},
                            "exits": [
                                {
                                    "id": "east_exit",
                                    "direction": "east",
                                    "target_room": "room_b",
                                    "type": "normal",
                                }
                            ],
                        },
                    },
                    {
                        "file": "rooms/room_b.json",
                        "payload": {
                            "id": "room_b",
                            "coord": [1, 0],
                            "layout": self._empty_layout(),
                            "spawns": {"default": [1, 1]},
                        },
                    },
                ],
            )

            manager = RoomManager(dungeon_path)

        room = manager.get_room((0, 0))
        self.assertEqual(room.exits[0].target_entry, "west")

    def test_directional_target_entry_rejects_blocked_spawn_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            blocked_layout = self._empty_layout()
            blocked_layout[3] = ".#........"
            blocked_layout[4] = ".#........"
            dungeon_path = self._write_dungeon(
                root,
                start_room="room_a",
                rooms=[
                    {
                        "file": "rooms/room_a.json",
                        "payload": {
                            "id": "room_a",
                            "coord": [0, 0],
                            "layout": self._empty_layout(),
                            "spawns": {"default": [1, 1]},
                            "exits": [
                                {
                                    "id": "east_exit",
                                    "direction": "east",
                                    "target_room": "room_b",
                                    "target_entry": "west",
                                    "type": "normal",
                                }
                            ],
                        },
                    },
                    {
                        "file": "rooms/room_b.json",
                        "payload": {
                            "id": "room_b",
                            "coord": [1, 0],
                            "layout": blocked_layout,
                            "spawns": {"default": [2, 2]},
                        },
                    },
                ],
            )

            with self.assertRaises(MapValidationError) as ctx:
                RoomManager(dungeon_path)

        self.assertIn("target_entry", str(ctx.exception))
        self.assertIn("no valid non-wall spawn tile", str(ctx.exception))

    def test_conditional_exit_button_reference_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dungeon_path = self._write_dungeon(
                root,
                start_room="room_a",
                rooms=[
                    {
                        "file": "rooms/room_a.json",
                        "payload": {
                            "id": "room_a",
                            "coord": [0, 0],
                            "layout": self._empty_layout(),
                            "spawns": {"default": [1, 1]},
                            "exits": [
                                {
                                    "id": "south_exit",
                                    "direction": "south",
                                    "target_room": "room_b",
                                    "target_entry": "north_entry",
                                    "type": "conditional",
                                    "requires": {"button_pressed": "missing_button"},
                                }
                            ],
                        },
                    },
                    {
                        "file": "rooms/room_b.json",
                        "payload": {
                            "id": "room_b",
                            "coord": [0, 1],
                            "layout": self._empty_layout(),
                            "spawns": {"north_entry": [4, 1]},
                        },
                    },
                ],
            )

            with self.assertRaises(MapValidationError) as ctx:
                RoomManager(dungeon_path)

        self.assertIn("missing_button", str(ctx.exception))

    def _write_dungeon(self, root: Path, start_room: str, rooms: list[dict]) -> Path:
        room_files: list[str] = []
        for room in rooms:
            room_path = root / room["file"]
            room_path.parent.mkdir(parents=True, exist_ok=True)
            room_path.write_text(json.dumps(room["payload"], indent=2), encoding="utf-8")
            room_files.append(room["file"])

        dungeon = {"schema_version": 1, "start_room": start_room, "room_files": room_files}
        dungeon_path = root / "dungeon.json"
        dungeon_path.write_text(json.dumps(dungeon, indent=2), encoding="utf-8")
        return dungeon_path

    @staticmethod
    def _empty_layout() -> list[str]:
        return ["." * GRID_WIDTH for _ in range(GRID_HEIGHT)]


if __name__ == "__main__":
    unittest.main()
