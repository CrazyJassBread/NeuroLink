# RL Training Guide

Use `map + reward` directly.

## Examples

```python
from nesylink.env import make_env

env = make_env(
    map_id="dungeon",
    reward_id="sparse_exit",
    max_steps=500,
)
```

```python
env = make_env(
    map_path="nesylink/map_data/dungeons/key_door/room_001.json",
    reward_id="collect_key",
    max_steps=200,
)
```

```python
env = make_env(
    map_path="nesylink/map_data/dungeons/prototype/dungeon.json",
    reward_module="experiments.rewards.my_custom_reward",
)
```

`info["reward"]` is the supported place to inspect reward decomposition metadata.
