from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from env_diy.constants import (
    ACTION_A,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAP_PIXEL_HEIGHT,
    MONSTER_HIT_KNOCKBACK_PX,
    MONSTER_SPEED_RATIO,
    MONSTER_STUN_TICKS,
    PLAYER_SPEED_PX_PER_STEP,
    TILE_SIZE,
)
from env_diy.entities import tile_from_position_px
from env_diy.env import DungeonEnv


class DungeonEnvTests(unittest.TestCase):
    def test_noop_advances_tick_and_moves_monster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_monster_dungeon(Path(tmp_dir)))
            env.reset()
            before_step = env.step_count
            monster_before = next(iter(env.room.monsters.values())).position_px

            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)

        monster_after = next(iter(env.room.monsters.values())).position_px
        self.assertEqual(env.step_count, before_step + 1)
        self.assertNotEqual(monster_before, monster_after)
        self.assertIn("noop", info["events"])
        self.assertIn("monsters_updated", info["events"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_interact_still_moves_monster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_monster_dungeon(Path(tmp_dir)))
            env.reset()
            monster_before = next(iter(env.room.monsters.values())).position_px

            obs, reward, terminated, truncated, info = env.step(ACTION_A)

        monster_after = next(iter(env.room.monsters.values())).position_px
        self.assertNotEqual(monster_before, monster_after)
        self.assertIn("action_a", info["events"])
        self.assertIn("monsters_updated", info["events"])

    def test_move_action_uses_pixel_motion_not_tile_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_empty_dungeon(Path(tmp_dir), spawn=[1, 1]))
            env.reset()
            before = env.player.position_px

            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

        delta_x = env.player.position_px[0] - before[0]
        self.assertEqual(delta_x, PLAYER_SPEED_PX_PER_STEP)
        self.assertLess(delta_x, TILE_SIZE)
        self.assertEqual(env.player.position_px[1], before[1])

    def test_default_monster_speed_is_half_of_player(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_monster_dungeon(Path(tmp_dir)))
            env.reset()
            monster = next(iter(env.room.monsters.values()))

        self.assertEqual(monster.speed_px_per_step, PLAYER_SPEED_PX_PER_STEP * MONSTER_SPEED_RATIO)
        self.assertEqual(MONSTER_SPEED_RATIO, 0.5)

    def test_monster_speed_override_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_monster_dungeon(Path(tmp_dir), speed_override=0.75))
            env.reset()
            monster = next(iter(env.room.monsters.values()))

        self.assertEqual(monster.speed_px_per_step, 0.75)
        self.assertLess(monster.speed_px_per_step, PLAYER_SPEED_PX_PER_STEP)

    def test_pixel_to_tile_conversion_uses_entity_center(self) -> None:
        tile = tile_from_position_px((10.0, 0.0), TILE_SIZE)
        self.assertEqual(tile, (1, 0))

    def test_player_cannot_move_into_wall_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_wall_dungeon(Path(tmp_dir)))
            env.reset()
            before = env.player.position_px

            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

        self.assertEqual(env.player.position_px, before)
        self.assertIn("blocked_wall", info["events"])
        self.assertLess(reward, 0.0)

    def test_player_cannot_move_into_hud_area(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_empty_dungeon(Path(tmp_dir), spawn=[4, 7]))
            env.reset()
            before_y = env.player.position_px[1]

            obs, reward, terminated, truncated, info = env.step(ACTION_UP)
            self.assertLess(env.player.position_px[1], before_y)

            env.player.position_px = (env.player.position_px[0], MAP_PIXEL_HEIGHT - TILE_SIZE)
            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)
            self.assertEqual(env.player.position_px[1], MAP_PIXEL_HEIGHT - TILE_SIZE)

    def test_any_east_exit_tile_triggers_room_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dungeon_path = self._write_simple_exit_dungeon(Path(tmp_dir), direction="east", exit_type="normal")
            env = DungeonEnv(dungeon_path)
            env.reset()

            env.player.position_px = (144.0, 48.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertEqual(env.room.room_id, "room_b")
            self.assertIn("room_transition", info["events"])

            env.reset()
            env.player.position_px = (144.0, 64.0)
            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

        self.assertEqual(env.room.room_id, "room_b")
        self.assertIn("room_transition", info["events"])

    def test_non_exit_boundary_does_not_switch_room(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dungeon_path = self._write_simple_exit_dungeon(Path(tmp_dir), direction="east", exit_type="normal")
            env = DungeonEnv(dungeon_path)
            env.reset()
            env.player.position_px = (144.0, 16.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

        self.assertEqual(env.room.room_id, "room_a")
        self.assertIn("blocked_bounds", info["events"])

    def test_normal_exit_requires_no_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dungeon_path = self._write_simple_exit_dungeon(Path(tmp_dir), direction="west", exit_type="normal")
            env = DungeonEnv(dungeon_path)
            env.reset()
            env.player.position_px = (0.0, 48.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)
            obs, reward, terminated, truncated, info = env.step(3)

        self.assertEqual(env.room.room_id, "room_b")
        self.assertIn("room_transition", info["events"])

    def test_locked_key_exit_blocks_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dungeon_path = self._write_simple_exit_dungeon(
                Path(tmp_dir),
                direction="east",
                exit_type="locked_key",
                requires={"key_count": 1, "consume_key": True},
            )
            env = DungeonEnv(dungeon_path)
            env.reset()
            env.player.position_px = (144.0, 48.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

        self.assertEqual(env.room.room_id, "room_a")
        self.assertIn("blocked_locked", info["events"])

    def test_locked_key_exit_consumes_key_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dungeon_path = self._write_simple_exit_dungeon(
                Path(tmp_dir),
                direction="east",
                exit_type="locked_key",
                requires={"key_count": 1, "consume_key": True},
            )
            env = DungeonEnv(dungeon_path)
            env.reset()
            env.player.keys = 1
            env.player.position_px = (144.0, 48.0)

            obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

        self.assertEqual(env.room.room_id, "room_b")
        self.assertIn("used_key", info["events"])
        self.assertEqual(env.player.keys, 0)

    def test_conditional_exit_blocks_until_button_pressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dungeon_path = self._write_conditional_exit_dungeon(Path(tmp_dir))
            env = DungeonEnv(dungeon_path)
            env.reset()
            env.player.position_px = (64.0, 112.0)

            obs, reward, terminated, truncated, info = env.step(2)
            self.assertEqual(env.room.room_id, "room_a")
            self.assertIn("missing_requirement", info["events"])

            env.room.buttons["button_1"].is_pressed = True
            env.player.position_px = (64.0, 112.0)
            obs, reward, terminated, truncated, info = env.step(2)

        self.assertEqual(env.room.room_id, "room_b")
        self.assertIn("room_transition", info["events"])

    def test_trap_and_button_use_player_center_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_trigger_dungeon(Path(tmp_dir)))
            env.reset()

            for _ in range(4):
                obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
            self.assertIn("pressed_button", info["events"])

            for _ in range(8):
                obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)

        self.assertIn("trap_damage", info["events"])
        self.assertEqual(info["player_tile"], (1, 1))

    def test_game_over_sets_terminated_and_next_step_auto_resets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_spawn_trap_dungeon(Path(tmp_dir)))
            env.reset()
            env.player.health = 1

            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["game_over"])
            self.assertEqual(env.player.health, 0)

            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)

        self.assertFalse(terminated)
        self.assertTrue(info["auto_reset"])
        self.assertLess(env.player.health, env.player.max_health)
        self.assertEqual(env.player.position_px, (16.0, 16.0))

    def test_monster_collision_knockback_and_stun_still_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_monster_dungeon(Path(tmp_dir)))
            env.reset()
            monster = next(iter(env.room.monsters.values()))
            monster.position_px = env.player.position_px
            starting_x = monster.position_px[0]
            starting_health = env.player.health

            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)

        self.assertEqual(env.player.health, starting_health - monster.damage)
        self.assertEqual(monster.position_px[0] - starting_x, MONSTER_HIT_KNOCKBACK_PX)
        self.assertGreater(monster.stun_ticks_remaining, 0)
        self.assertIn("monster_hit", info["events"])

    def test_stunned_monster_resumes_movement_after_stun_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = DungeonEnv(self._write_monster_dungeon(Path(tmp_dir)))
            env.reset()
            monster = next(iter(env.room.monsters.values()))
            monster.position_px = env.player.position_px

            env.step(ACTION_NOOP)
            stunned_position = monster.position_px
            for _ in range(MONSTER_STUN_TICKS):
                env.step(ACTION_NOOP)

            self.assertEqual(monster.stun_ticks_remaining, 0)
            self.assertEqual(monster.position_px, stunned_position)

            obs, reward, terminated, truncated, info = env.step(ACTION_NOOP)

        self.assertNotEqual(monster.position_px, stunned_position)

    def test_observation_is_dict_with_pixel_fields(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        dungeon_path = project_root / "env_diy" / "map_data" / "dungeons" / "prototype" / "dungeon.json"

        env = DungeonEnv(dungeon_path)
        obs, info = env.reset()

        self.assertEqual(obs["grid"].shape, (GRID_HEIGHT, GRID_WIDTH))
        self.assertEqual(obs["player_position_px"].shape, (2,))
        self.assertEqual(obs["player_tile"].shape, (2,))
        self.assertEqual(obs["monsters_position_px"].shape[1], 2)
        self.assertEqual(obs["monsters_active_mask"].shape[0], env.max_monster_slots)

    def _write_empty_dungeon(self, root: Path, *, spawn: list[int]) -> Path:
        room = {
            "id": "room_a",
            "coord": [0, 0],
            "layout": self._empty_layout(),
            "spawns": {"default": spawn},
        }
        return self._write_dungeon(root, [room], "room_a")

    def _write_monster_dungeon(self, root: Path, speed_override: float | None = None) -> Path:
        monster_payload: dict[str, object] = {
            "id": "monster_1",
            "kind": "monster",
            "pos": [6, 1],
            "monster_type": "chaser",
        }
        if speed_override is not None:
            monster_payload["speed_px_per_step"] = speed_override
        room = {
            "id": "room_a",
            "coord": [0, 0],
            "layout": self._empty_layout(),
            "spawns": {"default": [1, 1]},
            "objects": [monster_payload],
        }
        return self._write_dungeon(root, [room], "room_a")

    def _write_wall_dungeon(self, root: Path) -> Path:
        layout = self._empty_layout()
        layout[1] = "..#......."
        room = {
            "id": "room_a",
            "coord": [0, 0],
            "layout": layout,
            "spawns": {"default": [1, 1]},
            "objects": [
                {
                    "id": "monster_1",
                    "kind": "monster",
                    "pos": [4, 1],
                    "monster_type": "chaser",
                }
            ],
        }
        return self._write_dungeon(root, [room], "room_a")

    def _write_trigger_dungeon(self, root: Path) -> Path:
        room_a = {
            "id": "room_a",
            "coord": [0, 0],
            "layout": self._empty_layout(),
            "spawns": {"default": [1, 1]},
            "objects": [
                {"id": "button_1", "kind": "button", "pos": [2, 1], "message": "pressed"},
                {"id": "trap_1", "kind": "trap", "pos": [3, 1], "damage": 1, "respawn_to": "default"},
            ],
        }
        return self._write_dungeon(root, [room_a], "room_a")

    def _write_spawn_trap_dungeon(self, root: Path) -> Path:
        room = {
            "id": "room_a",
            "coord": [0, 0],
            "layout": self._empty_layout(),
            "spawns": {"default": [1, 1]},
            "objects": [
                {"id": "trap_1", "kind": "trap", "pos": [1, 1], "damage": 1, "respawn_to": "default"}
            ],
        }
        return self._write_dungeon(root, [room], "room_a")

    def _write_simple_exit_dungeon(
        self,
        root: Path,
        *,
        direction: str,
        exit_type: str,
        requires: dict | None = None,
    ) -> Path:
        target_coord = {
            "east": [1, 0],
            "west": [-1, 0],
            "north": [0, -1],
            "south": [0, 1],
        }[direction]
        target_spawn = {
            "east": "west_entry",
            "west": "east_entry",
            "north": "south_entry",
            "south": "north_entry",
        }[direction]
        room_a = {
            "id": "room_a",
            "coord": [0, 0],
            "layout": self._empty_layout(),
            "spawns": {"default": [1, 1]},
            "exits": [
                {
                    "id": f"{direction}_exit",
                    "direction": direction,
                    "target_room": "room_b",
                    "target_entry": target_spawn,
                    "type": exit_type,
                    "requires": requires or {},
                }
            ],
        }
        room_b = {
            "id": "room_b",
            "coord": target_coord,
            "layout": self._empty_layout(),
            "spawns": {target_spawn: [1, 4]},
        }
        return self._write_dungeon(root, [room_a, room_b], "room_a")

    def _write_conditional_exit_dungeon(self, root: Path) -> Path:
        room_a = {
            "id": "room_a",
            "coord": [0, 0],
            "layout": self._empty_layout(),
            "spawns": {"default": [1, 1]},
            "objects": [
                {"id": "button_1", "kind": "button", "pos": [2, 6], "message": "pressed"}
            ],
            "exits": [
                {
                    "id": "south_exit",
                    "direction": "south",
                    "target_room": "room_b",
                    "target_entry": "north_entry",
                    "type": "conditional",
                    "requires": {"button_pressed": "button_1"},
                }
            ],
        }
        room_b = {
            "id": "room_b",
            "coord": [0, 1],
            "layout": self._empty_layout(),
            "spawns": {"north_entry": [4, 1]},
        }
        return self._write_dungeon(root, [room_a, room_b], "room_a")

    def _write_dungeon(self, root: Path, rooms: list[dict], start_room: str) -> Path:
        room_files: list[str] = []
        for index, payload in enumerate(rooms):
            room_rel = f"rooms/room_{index}.json"
            room_path = root / room_rel
            room_path.parent.mkdir(parents=True, exist_ok=True)
            room_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            room_files.append(room_rel)

        dungeon = {"schema_version": 1, "start_room": start_room, "room_files": room_files}
        dungeon_path = root / "dungeon.json"
        dungeon_path.write_text(json.dumps(dungeon, indent=2), encoding="utf-8")
        return dungeon_path

    @staticmethod
    def _empty_layout() -> list[str]:
        return ["." * GRID_WIDTH for _ in range(GRID_HEIGHT)]


if __name__ == "__main__":
    unittest.main()
