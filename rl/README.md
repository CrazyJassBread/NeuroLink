# RL Training

Training now uses one unified entrypoint at the repository root:

```bash
python train.py --config rl/config/defaults/ppo_nesylink.yaml
python train.py --config rl/config/defaults/ppo_minigrid.yaml
python train.py --config rl/config/defaults/ppo_minigrid.yaml --set algorithm.total_timesteps=100000
```

Structure:

- `rl/config/`: YAML experiment configs plus schema/loader code.
- `rl/envs/`: environment adapters and registry.
- `rl/algorithms/`: algorithm-specific trainers and registries.
- `rl/runners/`: top-level orchestration from config to training run.

The root `train.py` is the only supported training interface.
