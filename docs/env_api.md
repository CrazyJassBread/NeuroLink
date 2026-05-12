# env_diy API

## Create an Environment

Canonical entrypoint:

```python
from env_diy.env import make_env

env = make_env(
    "env_diy/map_data/dungeons/prototype/dungeon.json",
    api="gym",
    reward_mode="legacy",
    action_repeat=1,
)
```

Compatibility entrypoints kept during the current cleanup window:

```python
from env_diy import DungeonEnv          # deprecated legacy auto-reset compatibility
from env_diy.env import DungeonEnv      # same legacy class without the root namespace shim
```

`make_env(api="gym")` returns the canonical Gymnasium wrapper with `auto_reset_on_step=False`.
`env_diy.envs.DungeonEnv` still forwards to the same legacy class, but it is a deprecated compatibility namespace and should not be used in new code.

## reset / step

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

- `reset(seed=...)` stores the seed in the env runtime and in `info["seed"]`.
- `step()` does not silently reset after `terminated` or `truncated` in the canonical wrapper.
- Legacy `DungeonEnv(...)` keeps `auto_reset_on_step=True` for compatibility.

## Action Space

The action space is `Discrete(7)`:

```text
0=no-op  1=up  2=down  3=left  4=right  5=A/interact  6=B/shield
```

`action_repeat` is supported directly by the canonical wrapper. Default remains `1`.
Do not combine it with `rl.utils.wrappers.ActionRepeatWrapper`; double stacking raises an error.

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

Stable fields:

- `episode_id`
- `step_count`
- `task_id`
- `task_type`
- `map_id`
- `seed`
- `agent_pos`
- `agent_hp`
- `inventory`
- `events`
- `event_flags`
- `event_counts`
- `event_records`
- `task_progress`
- `success`
- `failure`
- `terminated_reason`
- `reward_terms`
- `action_repeat`
- `inner_steps`
- `legacy_done`
- `validator_done`
- `validator_matches_legacy`

Compatibility fields kept for existing scripts:

- `finish`
- `task_success`
- `health`
- `gold`
- `keys`
- `key_count`
- `room_id`
- `room_coord`
- `message`
- `reward_breakdown`

## Determinism

If you want reproducible action sampling:

```python
obs, info = env.reset(seed=7)
env.action_space.seed(7)
```

Same seed plus same action sequence should produce the same reward and position sequence.
