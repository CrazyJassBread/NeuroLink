# Env DIY

`env_diy` is the current canonical Gymnasium-compatible dungeon environment in
this repository.

It is intended for:

- fast local environment iteration
- RL smoke tests and baseline integration
- benchmark task experiments built on top of the environment

## Canonical Entry Point

```python
from env_diy.env import make_env

env = make_env("env_diy/map_data/dungeons/prototype/dungeon.json", api="gym")
```

`make_env(api="gym")` returns the canonical Gym wrapper with
`auto_reset_on_step=False`.

Compatibility notes:

- `env_diy.env.DungeonEnv` remains available for legacy auto-reset behavior
- `env_diy.envs.DungeonEnv` and `env_diy.DungeonEnv` are deprecated
  compatibility aliases

## Core Facts

- Room size: `10 x 8` tiles
- HUD size: `10 x 2` tiles
- Render size: `160 x 160` RGB
- Base player speed: `1 px/tick`
- Default monster speed: `0.5 px/tick`
- Action space: `Discrete(7)`

Action ids:

```text
0 = no-op
1 = up
2 = down
3 = left
4 = right
5 = A / interact
6 = B / shield
```

## Quick Commands

Human play:

```bash
source .venv/bin/activate
python -m env_diy.main
```

Random smoke training:

```bash
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
python rl/train_single_task.py --task avoid_traps --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
```

Benchmark v0 random evaluation:

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 2 --seed 0
```

## Canonical Documentation

- [Docs Hub](../docs/README.md)
- [Env Overview](../docs/guides/env-overview.md)
- [Environment API](../docs/reference/env-api.md)
- [Rewards](../docs/reference/rewards.md)
- [Tasks and Validators](../docs/reference/tasks-and-validators.md)
- [Benchmark v0](../docs/reference/benchmark-v0.md)
- [Training Guide](../docs/guides/training.md)
- [Development Guide](../docs/project/development-guide.md)

This README stays intentionally short. Detailed mechanics, schemas, and
training notes should live in the linked canonical docs above.
