# NeuroLink

NeuroLink contains `env_diy`, RL smoke training helpers, and DreamerV3 experiments.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```

## env_diy

```python
from env_diy.envs.factory import make_env

env = make_env(
    map_id="dungeon",
    reward_id="sparse_exit",
    max_steps=500,
)
```

## Key Architecture Rules

- map JSON is world-only
- reward modules are training-only
- env owns Gymnasium `reset/step`
- task registry API is deprecated
