# RL Training

RL training is launched from the `rl/` package entrypoint:

```bash
python rl/train.py --config rl/config/defaults/ppo_kill_monsters.yaml
```

Run commands from the repository root so relative config paths and output paths resolve consistently.

## Quick Start

Train PPO on the default NesyLink task(kill monsters):

```bash
python rl/train.py --config rl/config/defaults/ppo_kill_monsters.yaml
```

Train PPO on NesyLink using native RGB pixel observations:

```bash
python rl/train.py --config rl/config/defaults/ppo_kill_monsters_pixels.yaml
```
> you can change the yaml config to use a different map, or override it from the command line with `--set environment.map_path=...`

Train PPO on MiniGrid:

```bash
python rl/train.py --config rl/config/defaults/ppo_minigrid.yaml
```

Override config values from the command line:

```bash
python rl/train.py \
  --config rl/config/defaults/ppo_kill_monsters_pixels.yaml \
  --set algorithm.total_timesteps=100000 \
  --set experiment.seed=1
```

## Configuration

Training configs are YAML files under `rl/config/defaults/`. The top-level sections are:

- `experiment`: run name, random seed, device, output directory, and resume behavior.
- `environment`: environment id, map path, reward id/module, action repeat, episode length, parallel environments, rendering, and environment-specific parameters.
- `algorithm`: algorithm name, total timesteps, and algorithm hyperparameters.
- `evaluation`: evaluation episodes, deterministic policy flag, model saving, metrics saving, and rendering.

Any unknown keys in `environment` are collected into `environment.params`. For NesyLink, this is how pixel observations are enabled:

```yaml
environment:
  id: nesylink
  observation_mode: pixels
```

Supported NesyLink observation modes:

- `dict`: default symbolic observation dict, including grid, player state, inventory, and monster state.
- `pixels`: native RGB game screen from `env.render()`, shape `(160, 160, 3)`, dtype `uint8`, no resize.

When `observation_mode: pixels` is used, PPO automatically selects `CnnPolicy`. When the default dict observation is used, PPO selects `MultiInputPolicy` with the custom NesyLink feature extractor.

## Parallel environments (num_envs)

PPO supports vectorized sampling by setting `environment.num_envs`. When `num_envs` is greater than 1, training creates multiple environments in parallel (for NesyLink this uses subprocesses) to increase rollout throughput and improve GPU utilization.

Example configuration:

```yaml
environment:
  id: nesylink
  num_envs: 8
```

You can also override it on the command line:

```bash
python rl/train.py \
  --config rl/config/defaults/ppo_nesylink.yaml \
  --set environment.num_envs=8
```

Notes:

- Keep `algorithm.batch_size <= algorithm.n_steps * environment.num_envs`.
- If GPU utilization is still low, increase `algorithm.n_steps` and `algorithm.batch_size` together after raising `num_envs`.
- Higher `num_envs` increases CPU usage; scale up gradually to avoid CPU bottlenecks.

## Common Commands

Short smoke run without evaluation:

```bash
python rl/train.py \
  --config rl/config/defaults/ppo_nesylink_pixels.yaml \
  --set algorithm.total_timesteps=1000 \
  --set evaluation.enabled=false
```

Train on a different NesyLink map:

```bash
python rl/train.py \
  --config rl/config/defaults/ppo_kill_monsters_pixels.yaml \
  --set environment.map_path=nesylink/map_data/dungeons/kill_monsters/room_001.json
```

Change the output directory:

```bash
python rl/train.py \
  --config rl/config/defaults/ppo_kill_monsters_pixels.yaml \
  --set experiment.output_dir=rl/outputs/ppo_kill_monsters_pixels_room_001
```

Resume from an existing saved model:

```bash
python rl/train.py \
  --config rl/config/defaults/ppo_kill_monsters_pixels.yaml \
  --set experiment.resume=true
```

## Evaluation (rl/eval.py)

Use `rl/eval.py` to run evaluation rollouts for a trained PPO model and export GIFs.

Basic usage (uses the config output directory and model path):

```bash
python rl/eval.py --config rl/config/defaults/ppo_kill_monsters.yaml
```

Override evaluation settings and save metrics:

```bash
python rl/eval.py \
  --config rl/config/defaults/ppo_kill_monsters.yaml \
  --episodes 3 \
  --fps 12 \
  --deterministic \
  --save-metrics
```

Evaluate a specific model and export GIFs to a custom path:

```bash
python rl/eval.py \
  --config rl/config/defaults/ppo_kill_monsters.yaml \
  --model rl/outputs/ppo_kill_monsters/model.zip \
  --output rl/outputs/ppo_kill_monsters/eval.gif
```

## Outputs

By default, PPO writes artifacts under the configured `experiment.output_dir`:

- `model.zip`: saved Stable-Baselines3 model when `evaluation.save_model: true`.
- `eval.jsonl`: one JSON record per evaluation episode when `evaluation.save_metrics: true`.

## Project Layout

- `rl/train.py`: training CLI entrypoint.
- `rl/config/`: YAML configs, dataclass schema, and config loader.
- `rl/envs/`: environment adapters and registry.
- `rl/algorithms/`: algorithm-specific trainers and registries.
- `rl/runners/`: orchestration from parsed config to training run.

## Notes

- Run from the repository root, not from inside `rl/`.
- Pixel PPO uses the full game interface at `(160, 160, 3)`, while the separate Dreamer wrapper may still use its own configured image size.
- If startup fails with `ModuleNotFoundError`, install the project requirements for the training stack before rerunning.
