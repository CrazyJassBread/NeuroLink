# Env DIY

`env_diy` is the Gymnasium-compatible dungeon environment in this repo.

## Canonical API

```python
from env_diy.envs.factory import make_env

env = make_env(
    map_id="dungeon",
    reward_id="sparse_exit",
    max_steps=500,
)
```

Equivalent custom-path form:

```python
env = make_env(
    map_path="env_diy/maps/dungeon.json",
    reward_module="env_diy.rewards.collect_key",
    reward_kwargs={"step": -0.01, "keys_delta": 5.0},
)
```

## Boundaries

- map JSON only builds the world
- env runs game mechanics and Gymnasium `reset/step`
- reward modules compute reward and task-driven termination flags
- `info` exposes generic state, events, episode metadata, and `info["reward"]`
- task registry and task-driven map registration are deprecated

## Map Rules

Map JSON may contain:

- layout
- spawns
- objects
- exits
- room graph / dungeon root references

Map JSON must not contain:

- `task_id`
- `task_type`
- `reward`
- `success_condition`
- `failure_condition`
- `progress`

## Reward Rules

Builtin reward modules live under `env_diy/rewards/`.

- `BaseReward` maintains `prev_obs` / `prev_info`
- `BaseReward` extracts common reward signals
- concrete reward files usually only define `reward_name`, `reward_weights`, and `make_reward(**kwargs)`

## Gymnasium Contract

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

`info["reward"]` contains:

- `reward_name`
- `reward_signals`
- `reward_weights`
- `terminated`
- `terminated_reason`
