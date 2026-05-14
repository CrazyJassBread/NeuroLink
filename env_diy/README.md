# Env DIY

`env_diy` is the current canonical Gymnasium-compatible dungeon environment in
this repository.

It is intended for:

- fast local environment iteration
- RL smoke tests and baseline integration
- benchmark task experiments built on top of the environment

## Canonical Entry Point

```python
from env_diy.env import make_env

env = make_env("env_diy/map_data/dungeons/prototype/dungeon.json", api="gym")
```

`make_env(api="gym")` returns the canonical Gym wrapper with
`auto_reset_on_step=False`.

Compatibility notes:

- `env_diy.env.DungeonEnv` remains available for compatibility auto-reset behavior
- `env_diy.envs.DungeonEnv` and `env_diy.DungeonEnv` are deprecated
  compatibility aliases

## Core Facts

- Room size: `10 x 8` tiles
- HUD size: `10 x 2` tiles
- Render size: `160 x 160` RGB
- Base player speed: `1 px/tick`
- Default monster speed: `0.5 px/tick`
- Action space: `Discrete(7)`

Action ids:

```text
0 = no-op
1 = up
2 = down
3 = left
4 = right
5 = A / interact
6 = B / shield
```

## Quick Commands

Human play:

```bash
source .venv/bin/activate
python -m env_diy.main
```

Random smoke training:

```bash
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
python rl/train_single_task.py --task avoid_traps --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
```

Benchmark v0 random evaluation:

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 2 --seed 0
```

## Architecture Boundaries

`env_diy` now separates game mechanics from training tasks:

- map JSON is a pure world-construction file
- `BaseGameEnv` / `DungeonEngine` only expose generic game state and generic events
- task success, task failure, progress, and reward shaping live outside the base env

Base termination is limited to hard game-mechanic outcomes such as:

- `agent_dead`
- `world_completed`
- other unrecoverable environment-level end states

Generic events such as `exit_reached`, `key_collected`, `door_opened`,
`monster_killed`, and `trap_triggered` do not end the episode by themselves in
the Base env.

## Pure Map Schema

Room JSON files under `env_diy/map_data/` must only describe the map and
initial world state. They must not contain training-task fields such as:

- `task`
- `task_id`
- `task_type`
- `objective`
- `progress`
- `reward`
- `rewards`
- `reward_profile`
- `success_condition`
- `failure_condition`

Typical room fields are:

- `id`
- `coord`
- `layout`
- `spawns`
- `default_spawn`
- `objects`
- `exits`

Minimal example:

```json
{
  "id": "room_001",
  "coord": [0, 0],
  "layout": [
    "....E.....",
    ".##.##.##.",
    "..........",
    ".##.##.##.",
    "..........",
    "..........",
    "....P.....",
    ".........."
  ],
  "spawns": { "default": [4, 6] },
  "default_spawn": "default",
  "objects": [],
  "exits": [
    {
      "id": "north_exit",
      "direction": "north",
      "target_room": "room_001",
      "target_entry": "from_south",
      "type": "normal",
      "success_message": "CLEARED!"
    }
  ]
}
```

## Base Info Contract

`reset()` and `step()` now return a task-agnostic `info` payload. The top-level
keys are:

- `episode`
- `env`
- `agent`
- `inventory`
- `entities`
- `events`
- `terminal_reason`
- `control`
- `debug`

There is no `info["task"]` or `info["reward"]`. Reward terms and task outcome
belong to wrappers, not to Base env state.

## Reward Wrappers

Task reward is injected outside the base environment:

```python
from env_diy.env import make_env
from env_diy.rewards import RewardWrapper
from env_diy.rewards.reward_fn import KeyDoorReward
from env_diy.tasks import get_task_spec

base_env = make_env("env_diy/map_data/dungeons/key_door/room_001.json", api="gym")
task_spec = get_task_spec("key_door_room_001")
env = RewardWrapper(base_env, reward_fn=KeyDoorReward(task_spec))
```

The reward function only receives:

- `prev_obs`
- `prev_info`
- `obs`
- `info`
- `action`

It does not read map JSON, `task_type`, or `map_id`.

## Map Export Tool

`env_diy/tools/export_map.py` is a development-only helper for converting
editable map source files into the current pure room JSON schema used by
`env_diy`.

It does not change runtime game logic. Exported JSON is validated by the
current `RoomManager` before it is written.

Source files can live under `env_diy/diy_map_sources/` and may be provided as:

- ASCII `.txt`
- JSON grid arrays
- optional YAML with `metadata` + `grid`

CLI:

```bash
python env_diy/tools/export_map.py \
  --input env_diy/diy_map_sources/examples/key_door_room.yaml \
  --output /tmp/key_door_room.json \
  --room-id room_001
```

ASCII source symbols:

- `#` wall
- `.` floor
- `P` player spawn
- `K` key chest
- `D` locked door / keyed exit
- `M` monster
- `G` gold chest
- `T` trap
- `E` exit

Notes:

- The exporter targets the current fixed dungeon size of `10 x 8`.
- `K` and `G` export as existing `chest` objects with current loot payloads.
- `D` exports to a keyed `locked_key` exit.
- `E` exports to a normal exit.
- `D` and `E` must be placed on room-edge tiles so they can map to the current
  exit schema.
- YAML support is optional and only needed when exporting YAML sources. Runtime
  environment loading does not depend on YAML.

Example source files:

- `env_diy/diy_map_sources/examples/key_door_room.yaml`
- `env_diy/diy_map_sources/examples/kill_monster_room.yaml`
- `env_diy/diy_map_sources/examples/trap_room.yaml`

## Legacy Map Migration

Use `env_diy/tools/migrate_map_schema.py` to remove legacy task/reward fields
from older room JSON files:

```bash
python env_diy/tools/migrate_map_schema.py \
  --input env_diy/map_data/dungeons/key_door/room_001.legacy.json \
  --output /tmp/room_001.json \
  --report /tmp/room_001.report.json
```

The migrated room JSON is pure map schema. The optional report stores removed
legacy fields so task metadata can be re-registered in code.

## Canonical Documentation

- [Docs Hub](../docs/README.md)
- [Env Overview](../docs/guides/env-overview.md)
- [Environment API](../docs/reference/env-api.md)
- [Rewards](../docs/reference/rewards.md)
- [Tasks and Validators](../docs/reference/tasks-and-validators.md)
- [Benchmark v0](../docs/reference/benchmark-v0.md)
- [Training Guide](../docs/guides/training.md)
- [Development Guide](../docs/project/development-guide.md)

This README stays intentionally short. Detailed mechanics, schemas, and
training notes should live in the linked canonical docs above.
