# NeuroLink

NeuroLink is a reinforcement learning workspace for Link's Awakening-style experiments. It contains the original PyBoy-based Zelda environments, plus `env_diy`, a small Gymnasium-compatible dungeon environment used for fast local iteration and RL smoke tests.

## Overview

- `env_diy/`: original pixel dungeon environment, procedural renderer, pygame runner, and JSON map data.
- `rl/`: dependency-light smoke training scripts and RL utilities for `env_diy`.
- `world_model/`: DreamerV3 integration experiments.
- `docs/`: implementation notes, game guide, training notes, skills, and worklogs.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```

## Human Play

```bash
source .venv/bin/activate
python -m env_diy.main
```

Controls are arrow keys for held movement, `Z` for A/interact, `X` for held B/shield, and `Esc` to quit.

## RL Training

Tips 💡: `env_diy` now uses 1 pixel per environment tick for base player movement. For RL exploration, use action repeat at the training-script layer:

```bash
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
python rl/train_single_task.py --task avoid_traps --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
```

`max_steps` counts outer agent decisions. Actual environment ticks are approximately `max_steps * action_repeat`, unless the episode ends early.

## Documentation

- [Docs Hub](docs/README.md)
- [Env DIY README](env_diy/README.md)
- [Env Overview](docs/guides/env-overview.md)
- [Training Guide](docs/guides/training.md)
- [Development Guide](docs/project/development-guide.md)

