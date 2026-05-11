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
            task_rooms=("prototype",),
            config_path=DUNGEON_ROOT / "prototype" / "dungeon.json",
            difficulty="intro",
            max_episode_steps=200,
            default_reward_mode="legacy",
            supported_reward_modes=("legacy", "event", "sparse"),
            observation_mode="dict",
            action_mode="discrete7",
            success_condition="Reach built-in legacy victory conditions in the prototype dungeon.",
            failure_condition="Agent death or truncated rollout before solving the prototype.",
        ),
        BenchmarkTaskSpec(
            suite_id="NesyLink-v0",
            task_id="avoid_traps",
            map_id="avoid_traps",
            task_rooms=("avoid_traps",),
            config_path=DUNGEON_ROOT / "avoid_traps" / "room_001.json",
            difficulty="easy",
            max_episode_steps=100,
            default_reward_mode="legacy",
            supported_reward_modes=("legacy", "event", "sparse"),
            observation_mode="dict",
            action_mode="discrete7",
            success_condition="Reach the configured exit without failing the task.",
            failure_condition="Agent death or timeout before the target exit is reached.",
        ),
        BenchmarkTaskSpec(
            suite_id="NesyLink-v0",
            task_id="kill_monsters",
            map_id="kill_monsters",
            task_rooms=("kill_monsters",),
            config_path=DUNGEON_ROOT / "kill_monsters" / "room_001.json",
            difficulty="easy",
            max_episode_steps=120,
            default_reward_mode="legacy",
            supported_reward_modes=("legacy", "event", "sparse"),
            observation_mode="dict",
            action_mode="discrete7",
            success_condition="Defeat the target monsters defined by the room objective.",
            failure_condition="Agent death or timeout before all target monsters are defeated.",
        ),
        BenchmarkTaskSpec(
            suite_id="NesyLink-v0",
            task_id="key_door",
            map_id="key_door",
            task_rooms=("key_door",),
            config_path=DUNGEON_ROOT / "key_door" / "room_001.json",
            difficulty="easy",
            max_episode_steps=120,
            default_reward_mode="legacy",
            supported_reward_modes=("legacy", "event", "sparse"),
            observation_mode="dict",
            action_mode="discrete7",
            success_condition="Collect the key and unlock the goal door.",
            failure_condition="Agent death or timeout before opening the door.",
        ),
    ),
)
