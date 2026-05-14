# Task API Deprecation

The old task registry / task wrapper flow is deprecated.

Deprecated concepts:

- task registry
- task factory
- task-driven map registration
- task reward config inside map JSON
- `info["task"]`

Use this instead:

```python
from env_diy.envs.factory import make_env

env = make_env(
    map_id="dungeon",
    reward_id="sparse_exit",
)
```

or:

```python
env = make_env(
    map_path="env_diy/maps/dungeon.json",
    reward_module="experiments.rewards.my_custom_reward",
)
```
