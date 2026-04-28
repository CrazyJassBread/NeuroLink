from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from env_diy.constants import GRID_HEIGHT, GRID_WIDTH
from env_diy.room import MapValidationError, RoomManager


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
        self.assertEqual(len(room.transitions), 3)
        self.assertTrue(any(transition.requires_key == 1 for transition in room.transitions))

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

    def test_invalid_transition_target_rejected(self) -> None:
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
                            "transitions": [
                                {
                                    "id": "exit_1",
                                    "pos": [0, 4],
                                    "direction": "left",
                                    "target_room": "missing_room",
                                    "target_spawn": "default",
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

    def test_transition_must_be_on_matching_edge(self) -> None:
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
                            "transitions": [
                                {
                                    "id": "exit_1",
                                    "pos": [4, 4],
                                    "direction": "left",
                                    "target_room": "room_b",
                                    "target_spawn": "default",
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

            with self.assertRaises(MapValidationError) as ctx:
                RoomManager(dungeon_path)

        self.assertIn("matching room edge", str(ctx.exception))

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
