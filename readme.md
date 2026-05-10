# NeuroLink

NeuroLink is a reinforcement learning workspace for Link's Awakening-style experiments. It contains the original PyBoy-based Zelda environments, plus `env_diy`, a small Gymnasium-compatible dungeon environment used for fast local iteration and RL smoke tests.

## Overview

- `env_diy/`: original pixel dungeon environment, procedural renderer, pygame runner, and JSON map data.
- `rl/`: dependency-light smoke training scripts and RL utilities for `env_diy`.
- `orgin_zelda/`: PyBoy-based historical environments and training code.
- `world_model/`: DreamerV3 integration experiments.
- `docs/`: implementation notes, game guide, training notes, skills, and worklogs.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
python -m pytest -q tests/test_env_diy_env.py tests/test_rl_smoke.py
```

## Human Play

```bash
source .venv/bin/activate
python -m env_diy.main
```

Controls are arrow keys for held movement, `Z` for A/interact, `X` for held B/shield, and `Esc` to quit.

## RL Smoke Training

`env_diy` now uses 1 pixel per environment tick for base player movement. For RL exploration, use action repeat at the training-script layer:

```bash
source .venv/bin/activate
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
python rl/train_single_task.py --task avoid_traps --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
```

`max_steps` counts outer agent decisions. Actual environment ticks are approximately `max_steps * action_repeat`, unless the episode ends early.

## Documentation

- [Env DIY README](env_diy/README.md)
- [Env DIY Game Guide](docs/env_diy_game_guide.md)
- [RL Smoke Training](rl/README.md)
- [Training Notes](docs/train.md)
- [Development Guide](docs/DEVELOPMENT_GUIDE.md)

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

Some vendored or historical test modules may require optional dependencies; see the latest worklog under `docs/worklog/` for current known issues.
