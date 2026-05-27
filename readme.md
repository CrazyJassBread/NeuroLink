# NeuroLink

NeuroLink contains `nesylink`, RL smoke training helpers, and DreamerV3 experiments.

```text
Project Framework
    nesylink/     - game env and core mechanics (world, collision, rendering, rewards)
    rl/           - training and evaluation pipeline (algorithms, runner, config loading)
    world_model/  - DreamerV3 experiments and scripts
    envs/         - environment variants
    docs/     - project documentation
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```

## NesyLink

```python
from nesylink.env import make_env

env = make_env(
    map_id="dungeon",
    reward_id="sparse_exit",
    max_steps=500,
)
```
This creates a dungeon environment where the agent receives a reward only upon exiting the dungeon. You can customize the map and reward by changing the `map_id` and `reward_id` parameters.

for more details on available maps and rewards, see the `nesylink` directory.