# env_diy Overview

`env_diy` is now organized around three separate concerns:

1. map JSON builds the world
2. env runs game mechanics
3. reward modules define training objectives

## Create an Environment

```python
from env_diy.envs.factory import make_env

env = make_env(map_id="dungeon", reward_id="exploration", max_steps=500)
```

## Add a New Map

1. Create a new JSON file containing only world-building data.
2. Put it under `env_diy/maps/` or `env_diy/map_data/dungeons/`.
3. Call `make_env(map_id="new_map_name", reward_id="sparse_exit")`.

No task registry update is required.

## Add a New Reward

1. Create `env_diy/rewards/my_reward.py`.
2. Inherit from `BaseReward`.
3. Define `reward_name`.
4. Define `reward_weights`.
5. Export `make_reward(**kwargs)`.
