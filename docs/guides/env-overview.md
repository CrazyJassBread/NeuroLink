# env_diy Environment Overview

This guide gives the practical gameplay view of `env_diy`. For the exact API,
see [`docs/reference/env-api.md`](../reference/env-api.md).

## What the Environment Simulates

`env_diy` is a tile-based Zelda-like dungeon environment with:

- pixel-level movement
- room transitions
- keys, doors, chests, traps, buttons, and monsters
- prototype dungeon play plus single-task challenge rooms

The public Gym entrypoint is:

```python
from env_diy.env import make_env

env = make_env("env_diy/map_data/dungeons/prototype/dungeon.json", api="gym")
```

## Step Flow

At a high level, each step:

1. applies the selected action
2. advances room and monster state
3. resolves events such as damage, pickups, unlocks, and transitions
4. evaluates task success or failure
5. computes reward
6. returns the canonical nested `info`

The canonical wrapper does not auto-reset after termination. Call `reset()`
before stepping again. The compatibility `DungeonEnv(...)` wrapper may auto-reset, but
it still returns the same nested `info` schema.

## Observation Space

Observation keys remain unchanged:

- `grid`
- `player_position_px`
- `player_tile`
- `health`
- `gold`
- `keys`
- `inventory_ids`
- `monsters_position_px`
- `monsters_tile`
- `monsters_active_mask`
- `monsters_hp`

## Action Space

The action space is `Discrete(7)`:

```text
0=no-op  1=up  2=down  3=left  4=right  5=A/interact  6=B/shield
```

Action semantics are unchanged. `action_repeat` repeats the same outer action
for multiple inner environment ticks and is reported through
`info["control"]`.

## Canonical Info Schema

Both `reset()` and `step()` return the same top-level `info` keys:

- `episode`
- `env`
- `agent`
- `inventory`
- `events`
- `task`
- `reward`
- `control`
- `debug`

Highlights:

- `info["episode"]` contains `id`, `step_count`, and `seed`
- `info["env"]` contains `map_id`, `room_id`, and `room_coord`
- `info["agent"]` contains `hp`, `position_px`, and `tile`
- `info["inventory"]` contains `gold`, `keys`, `items`, `tools`, and `equipped`
- `info["events"]` contains normalized `records`, `flags`, `counts`, and `details`
- `info["task"]` contains success, failure, progress, and termination metadata
- `info["reward"]` contains `mode`, `total`, and `terms`
- `info["control"]` contains `action_repeat`, `inner_steps`, and `movement_pixels`
- `info["debug"]` contains `message`, `engine_done`, `validator_done`, and
  `validator_matches_engine`

Removed compatibility fields:

- no `info["legacy"]`
- no flat aliases like `game_over`, `finish`, `task_success`, `reward_terms`,
  `player_position_px`, `agent_pos`, `event_flags`, or `event_records`

## Events

Events are surfaced in four parallel forms:

- `records`: normalized event records, each with at least a `name`
- `flags`: per-step booleans such as `action_a`, `action_a_empty`, `picked_key`
- `counts`: per-step counts by event name
- `details`: structured payloads for richer events such as room transitions or
  task completion

Example task-finished detail:

```python
{
    "type": "task_finished",
    "task_id": "avoid_traps_001",
    "task_type": "avoid_traps",
    "reward": 10.0,
}
```

## Rewards

The scalar reward returned by `step()` always matches:

```python
info["reward"]["total"]
```

Reward decomposition lives in:

```python
info["reward"]["terms"]
```

Common reward-producing events include movement, empty actions, pickups, door
unlocks, monster kills, trap damage, death, and goal completion.

## Task Outcomes

Task and episode outcomes are represented through nested task/debug state:

- success: `info["task"]["success"]`
- failure: `info["task"]["failure"]`
- reason: `info["task"]["terminated_reason"]`

Typical meanings:

- `terminated_reason == "reached_goal"` for success
- `terminated_reason == "agent_dead"` for death

Single-task rooms still define task metadata in room JSON with `task_id`,
`task_type`, `objective`, and optional task reward overrides. That metadata is
used by runtime validators and event details, but it is no longer copied into
top-level `info`.

## Reset Defaults

After `reset()`:

- `info["events"]["records"] == []`
- `info["events"]["flags"] == {}`
- `info["events"]["counts"] == {}`
- `info["events"]["details"] == []`
- `info["reward"]["total"] == 0.0`
- `info["reward"]["terms"] == {}`
- `info["debug"]["message"] is None`

## Determinism

For reproducible behavior:

```python
obs, info = env.reset(seed=0)
env.action_space.seed(0)
```

With the same seed and action sequence, observations, rewards, and nested
`info` values should match apart from episode ids.
