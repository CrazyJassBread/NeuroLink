from __future__ import annotations

from pathlib import Path

from .specs import BenchmarkSuiteSpec, BenchmarkTaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DUNGEON_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"


NESYLINK_V0 = BenchmarkSuiteSpec(
    suite_id="NesyLink-v0",
    tasks=(
        BenchmarkTaskSpec(
            suite_id="NesyLink-v0",
            task_id="prototype",
            map_id="prototype",
            map_path=DUNGEON_ROOT / "prototype" / "dungeon.json",
            reward_id=None,
            reward_module=None,
            difficulty="intro",
            max_episode_steps=200,
            observation_mode="dict",
            action_mode="discrete7",
            objective="Reach the prototype dungeon's built-in completion condition.",
        ),
        BenchmarkTaskSpec(
            suite_id="NesyLink-v0",
            task_id="avoid_traps",
            map_id="avoid_traps",
            map_path=DUNGEON_ROOT / "avoid_traps" / "room_001.json",
            reward_id="sparse_exit",
            reward_module=None,
            difficulty="easy",
            max_episode_steps=100,
            observation_mode="dict",
            action_mode="discrete7",
            objective="Reach the exit under sparse exit reward.",
        ),
        BenchmarkTaskSpec(
            suite_id="NesyLink-v0",
            task_id="kill_monsters",
            map_id="kill_monsters",
            map_path=DUNGEON_ROOT / "kill_monsters" / "room_001.json",
            reward_id="kill_monster",
            reward_module=None,
            difficulty="easy",
            max_episode_steps=120,
            observation_mode="dict",
            action_mode="discrete7",
            objective="Clear all monsters in the room.",
        ),
        BenchmarkTaskSpec(
            suite_id="NesyLink-v0",
            task_id="key_door",
            map_id="key_door",
            map_path=DUNGEON_ROOT / "key_door" / "room_001.json",
            reward_id="collect_key",
            reward_module=None,
            difficulty="easy",
            max_episode_steps=120,
            observation_mode="dict",
            action_mode="discrete7",
            objective="Collect the key and open the locked door.",
        ),
    ),
)
