# Env DIY

`env_diy` is an original Gymnasium-compatible dungeon environment with pixel-level movement, JSON map configuration, procedural rendering, and a small pygame manual runner.

It does not use commercial sprites, maps, music, names, or proprietary data. Keep new content original and programmatic.

Recommended public entrypoint:

```python
from env_diy.env import make_env

env = make_env("env_diy/map_data/dungeons/prototype/dungeon.json", api="gym")
```

`make_env(api="gym")` returns the canonical Gymnasium wrapper with `auto_reset_on_step=False`.
`env_diy.envs.DungeonEnv` and `env_diy.DungeonEnv` remain available as compatibility aliases that keep legacy auto-reset behavior.

## What This Environment Is

- Fixed-size top-down dungeon rooms with Gymnasium `reset()` / `step()` semantics.
- Player and monsters move in pixel coordinates, not tile jumps.
- Base player speed is `1 px/tick`; default monster speed is `0.5 px/tick`.
- Every action advances exactly one environment tick, including no-op, A/interact, and B/shield.
- `action_repeat` is now an environment config. Default remains `1`, so base physics are unchanged.

## Map and Screen Size

- Room map: `10 x 8` tiles.
- HUD: bottom `10 x 2` tiles.
- Tile size: `16 x 16` pixels.
- Render output: `160 x 160` RGB array.
- Dynamic entities are constrained to the `160 x 128` dungeon area; HUD rows are visual-only.

## Core Controls

Action ids are stable:

```text
0 = no-op
1 = up
2 = down
3 = left
4 = right
5 = A / interact
6 = B / shield
```

Human play maps arrow keys to held movement, `Z` to A/interact, and held `X` to B/shield. Held B has priority over held movement, matching the current discrete-action limitation.

## Config Location

- Prototype multi-room dungeon: `env_diy/map_data/dungeons/prototype/dungeon.json`
- Single-task rooms:
  - `env_diy/map_data/dungeons/avoid_traps/room_001.json`
  - `env_diy/map_data/dungeons/kill_monsters/room_001.json`
  - `env_diy/map_data/dungeons/key_door/room_001.json`

Single-task rooms can define `task_id`, `task_type`, `objective`, and task-level rewards. Completion emits `task_finished`, sets `info["finish"] == True`, and terminates the episode.

## API Notes

- `reset(seed=...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- default `reward_mode="legacy"` strictly preserves the old reward behavior
- optional reward modes: `event`, `sparse`
- `info` includes both stable fields such as `episode_id`, `step_count`, `task_progress`, `reward_terms`, `event_counts`, and legacy fields such as `finish`, `task_success`, `health`, `gold`, `keys`, `room_id`, and `message`
- native `action_repeat` must not be stacked with `rl.utils.wrappers.ActionRepeatWrapper`
- validator migration is still observable in `info` via `legacy_done`, `validator_done`, and `validator_matches_legacy`

## How to Run

```bash
source .venv/bin/activate
python -m env_diy.main
```

Use a specific config:

```bash
python -m env_diy.main --rooms env_diy/map_data/dungeons/prototype/dungeon.json
python -m env_diy.main --rooms env_diy/map_data/dungeons/avoid_traps/room_001.json
```

Minimal RL smoke:

```bash
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
python rl/train_single_task.py --task avoid_traps --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
```

Unified classic RL training entry:

```bash
python rl/train.py --method ppo --task-rooms prototype --total-timesteps 50000 --episodes 5 --seed 0
python rl/train.py --method ppo --task-rooms avoid_traps kill_monsters key_door --episodes 2
```

Benchmark v0 random evaluation:

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 2 --seed 0
```

## Related Docs

- [Env DIY Game Guide](../docs/env_diy_game_guide.md)
- [Environment API](../docs/env_api.md)
- [Reward Modes](../docs/reward.md)
- [Tasks and Validators](../docs/tasks.md)
- [Benchmark v0](../docs/benchmark.md)
- [RL Smoke Training](../rl/README.md)
- [Development Guide](../docs/DEVELOPMENT_GUIDE.md)
