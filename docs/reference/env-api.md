# env_diy API

This file is the precise API reference for environment construction, Gymnasium
semantics, observation keys, and the canonical nested `info` schema.
Use `docs/guides/env-overview.md` for higher-level gameplay and mechanics notes.

## Create an Environment

Canonical entrypoint:

```python
from env_diy.env import make_env

env = make_env(
    "env_diy/map_data/dungeons/prototype/dungeon.json",
    api="gym",
    reward_mode="default",
    action_repeat=1,
)
```

Compatibility entrypoints still exist for wrapper behavior only:

```python
from env_diy import DungeonEnv
from env_diy.env import DungeonEnv
```

`make_env(api="gym")` returns the canonical Gymnasium wrapper with
`auto_reset_on_step=False`. The compatibility `DungeonEnv(...)` class keeps
`auto_reset_on_step=True` behavior, but both wrappers now return the same clean
nested `info` structure.

## reset / step

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

- `reset(seed=...)` stores the seed in the runtime and reports it through
  `info["episode"]["seed"]`.
- `step()` does not silently reset after termination in the canonical wrapper.
- The compatibility wrapper may auto-reset before a follow-up `step()`, but it still
  returns the same canonical `info` schema.

## Action Space

The action space is `Discrete(7)`:

```text
0=no-op  1=up  2=down  3=left  4=right  5=A/interact  6=B/shield
```

`action_repeat` is supported directly by the wrapper. Default is `1`.
Do not combine it with `rl.utils.wrappers.ActionRepeatWrapper`; double stacking
raises an error.

## Observation Space

Observations are a `Dict` space with:

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

## Info Schema

Returned `info` top-level keys are exactly:

- `episode`
- `env`
- `agent`
- `inventory`
- `events`
- `task`
- `reward`
- `control`
- `debug`

Canonical nested fields:

- `info["episode"]["id"]`
- `info["episode"]["step_count"]`
- `info["episode"]["seed"]`
- `info["env"]["map_id"]`
- `info["env"]["room_id"]`
- `info["env"]["room_coord"]`
- `info["agent"]["hp"]`
- `info["agent"]["position_px"]`
- `info["agent"]["tile"]`
- `info["inventory"]["gold"]`
- `info["inventory"]["keys"]`
- `info["inventory"]["items"]`
- `info["inventory"]["tools"]`
- `info["inventory"]["equipped"]`
- `info["events"]["records"]`
- `info["events"]["flags"]`
- `info["events"]["counts"]`
- `info["events"]["details"]`
- `info["task"]["success"]`
- `info["task"]["failure"]`
- `info["task"]["progress"]`
- `info["task"]["terminated_reason"]`
- `info["task"]["subgoals"]`
- `info["task"]["completed_subgoals"]`
- `info["task"]["failure_stage"]`
- `info["reward"]["mode"]`
- `info["reward"]["total"]`
- `info["reward"]["terms"]`
- `info["control"]["action_repeat"]`
- `info["control"]["inner_steps"]`
- `info["control"]["movement_pixels"]`
- `info["debug"]["message"]`
- `info["debug"]["engine_done"]`
- `info["debug"]["validator_done"]`
- `info["debug"]["validator_matches_engine"]`

Removed fields:

- No `info["legacy"]`
- No flat aliases such as `episode_id`, `step_count`, `room_id`, `health`,
  `agent_hp`, `gold`, `keys`, `message`, `event_flags`, `task_success`,
  `reward_terms`, `reward_breakdown`, `player_position_px`, or `agent_pos`

Reset defaults:

- `info["events"]["records"] == []`
- `info["events"]["flags"] == {}`
- `info["events"]["counts"] == {}`
- `info["events"]["details"] == []`
- `info["reward"]["total"] == 0.0`
- `info["reward"]["terms"] == {}`
- `info["debug"]["message"] is None`

## Determinism

For reproducible action sampling:

```python
obs, info = env.reset(seed=7)
env.action_space.seed(7)
```

Same seed plus the same action sequence should produce the same observation,
reward, and nested `info` trajectories apart from episode ids.
