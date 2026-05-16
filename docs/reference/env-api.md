# NesyLink API

## Environment Construction

Preferred entrypoint:

```python
from nesylink.env import make_env
```

Supported forms:

```python
env = make_env(map_id="dungeon", reward_id="sparse_exit", max_steps=500)
env = make_env(map_path="nesylink/map_data/dungeons/prototype/dungeon.json", reward_id="collect_key")
env = make_env(map_id="dungeon", reward_module="nesylink.rewards.exploration")
```

Parameters:

- `map_id`
- `map_path`
- `reward_id`
- `reward_module`
- `reward_kwargs`
- `max_steps`
- `render_mode`
- `action_repeat`
- `api`

`map_path` takes precedence over `map_id`.

## reset / step

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

`reset()` synchronizes the reward object by calling `reward_fn.reset(obs, info)`.

`step()`:

- advances game mechanics
- computes reward from the configured reward object
- merges base termination with reward-driven termination
- truncates on `max_steps`
- stores reward metadata in `info["reward"]`

## Info Shape

Top-level `info` keys:

- `episode`
- `env`
- `agent`
- `inventory`
- `entities`
- `events`
- `game`
- `terminal_reason`
- `control`
- `debug`
- `reward`

`info["task"]` is deprecated and no longer part of the contract.
