# NesyLink

`nesylink` is the Gymnasium-compatible dungeon environment in this repo.

## Canonical API

```python
from nesylink.env import make_env

env = make_env(
    map_id="dungeon",
    reward_id="sparse_exit",
    max_steps=500,
)
```

Equivalent custom-path form:

```python
env = make_env(
    map_path="nesylink/map_data/dungeons/prototype/dungeon.json",
    reward_module="nesylink.rewards.collect_key",
    reward_kwargs={"step": -0.01, "keys_delta": 5.0},
)
```

Human-play / debug entrypoint:

```bash
python -m nesylink.game --rooms nesylink/map_data/dungeons/prototype/dungeon.json
```

## Package Layout

- `core/`: game runtime, state, world loading, mechanics, rendering, and human input
- `rewards/`: reward calculation modules and loader
- `wrappers/`: Gymnasium and Dreamer-facing adapters
- `env.py`: public training facade exposing `make_env(...)`
- `game.py`: pygame human-play/debug runner

## Boundaries

- map JSON only builds the world
- `core` owns game mechanics and runtime state
- reward modules compute reward and task-driven termination flags
- wrappers expose agent-facing APIs without owning mechanics
- `info` exposes generic state, events, episode metadata, and `info["reward"]`
- benchmark code stays outside the core migration

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

Builtin reward modules live under `nesylink/rewards/`.

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
