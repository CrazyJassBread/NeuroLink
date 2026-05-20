# RL Training Guide

Use the unified root entrypoint:

```bash
python train.py --config rl/config/defaults/ppo_nesylink.yaml
python train.py --config rl/config/defaults/ppo_minigrid.yaml
```

Override config values from the CLI when needed:

```bash
python train.py \
  --config rl/config/defaults/ppo_minigrid.yaml \
  --set environment.task_id=MiniGrid-DoorKey-5x5-v0 \
  --set algorithm.total_timesteps=100000
```

Config layout:

- `experiment`: run name, seed, device, output path, resume flag.
- `environment`: environment id plus environment-specific parameters.
- `algorithm`: algorithm name, total timesteps, and extra hyperparameters.
- `evaluation`: eval cadence and artifact persistence settings.

Supported environment ids today:

- `nesylink`
- `minigrid`

Supported algorithm ids today:

- `ppo`
- `dqn` (registry only, implementation pending)
- `a3c` (registry only, implementation pending)
