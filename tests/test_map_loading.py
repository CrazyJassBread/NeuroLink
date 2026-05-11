from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from env_diy.maps import MapValidationError, RoomManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


class MapLoadingTests(unittest.TestCase):
    def test_all_current_dungeons_load(self) -> None:
        for config_path in sorted(DUNGEON_ROOT.glob("*/*.json")):
            with self.subTest(config_path=config_path.name):
                manager = RoomManager(config_path)
                room = manager.get_room(manager.start_room)
                self.assertEqual(room.room_id, manager.start_room_id)

    def test_invalid_map_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "broken.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(MapValidationError) as ctx:
                RoomManager(path)
        self.assertIn("schema_version", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
