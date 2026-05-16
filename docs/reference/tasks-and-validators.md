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
from nesylink.env import make_env

env = make_env(
    map_id="dungeon",
    reward_id="sparse_exit",
)
```

or:

```python
env = make_env(
    map_path="nesylink/map_data/dungeons/prototype/dungeon.json",
    reward_module="experiments.rewards.my_custom_reward",
)
```
