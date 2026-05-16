# NesyLink Rewards

## Core Files

- `nesylink/rewards/base.py`
- `nesylink/rewards/sparse_exit.py`
- `nesylink/rewards/collect_key.py`
- `nesylink/rewards/collect_gold.py`
- `nesylink/rewards/kill_monster.py`
- `nesylink/rewards/exploration.py`
- `nesylink/rewards/custom_template.py`

## BaseReward

`BaseReward` is the unified reward core.

Responsibilities:

- maintain `prev_obs` / `prev_info`
- extract stable reward signals from `obs/info/action`
- compute weighted reward via `reward_weights`
- support task-specific shaping via `extra_reward()`
- support task-specific termination via `check_termination()`

Common signals:

- `step`
- `hp_delta`
- `hp_loss`
- `gold_delta`
- `keys_delta`
- `monster_hit`
- `monster_kill`
- `door_opened`
- `chest_opened`
- `room_changed`
- `exit_reached`
- `death`
- `invalid_action`

## Reward Module Contract

Each concrete reward module must expose:

```python
def make_reward(**kwargs):
    ...
```

Typical custom reward:

```python
from nesylink.rewards.base import BaseReward


class MyReward(BaseReward):
    reward_name = "my_reward"
    reward_weights = {
        "step": -0.01,
        "gold_delta": 1.0,
        "keys_delta": 5.0,
        "exit_reached": 50.0,
        "death": -20.0,
    }


def make_reward(**kwargs):
    return MyReward(**kwargs)
```

## Weight Overrides

```python
env = make_env(
    map_id="dungeon",
    reward_id="collect_key",
    reward_kwargs={
        "step": -0.01,
        "keys_delta": 5.0,
        "door_opened": 3.0,
        "exit_reached": 20.0,
    },
)
```
