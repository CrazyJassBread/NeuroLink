# RL Smoke Training

This folder contains lightweight RL smoke scripts for validating that `env_diy` can be used by training code. The current baseline is a random policy, not a serious learning agent.

## Random Policy Rollout

Run from the repository root:

```bash
source .venv/bin/activate
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
```

The script:

- creates `env_diy.envs.DungeonEnv` directly;
- optionally wraps it with `ActionRepeatWrapper` for RL-only frame skip;
- samples actions from `env.action_space`;
- validates observations against `env.observation_space`;
- records episode reward, length, termination flags, and game-over status;
- writes JSONL summaries to `rl/outputs/random_training.jsonl` by default.

Use a custom dungeon config:

```bash
python rl/train_random.py --config env_diy/map_data/dungeons/prototype/dungeon.json
```

Use a custom output file:

```bash
python rl/train_random.py --episodes 2 --max-steps 20 --seed 0 --output rl/outputs/smoke.jsonl
```

## Action Repeat

The base `env_diy` environment moves the player at `1 px/tick`; default monster speed is `0.5 px/tick`. This keeps environment physics fine-grained, but random or early RL policies may explore too slowly if every agent decision advances only one pixel.

Use `--action-repeat` in RL scripts to repeat the same action for multiple underlying environment ticks:

```bash
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
```

`max_steps` is the outer agent step count. Actual environment ticks are approximately `max_steps * action_repeat`, unless `terminated` or `truncated` stops the repeat early. Rewards from inner ticks are summed and the final `info` includes `action_repeat`, `inner_steps`, and `repeated_reward`.

Recommended starting point:

- use `--action-repeat 4` for smoke training;
- try `--action-repeat 8` or larger `--max-steps` if exploration is still too sparse;
- prefer reward shaping, curriculum, and single-task maps for task-specific learning signals;
- keep action repeat in RL wrappers/scripts, not in base environment movement speed.

Rendering is off by default so the script can run in headless test environments. To explicitly call `env.render()` every step:

```bash
python rl/train_random.py --episodes 1 --max-steps 20 --render
```

## Tests

```bash
source .venv/bin/activate
python -m pytest -q tests/test_rl_smoke.py
```

The smoke tests cover env creation, reset/step return shapes, sampled actions, observation validation, explicit render output, and the CLI JSONL output path.

## Single-task Rooms

Use `train_single_task.py` to verify a focused challenge room can be loaded and
stepped by RL-style code. The script uses a random policy and reports finish
rate; it is a smoke test, not a training-quality baseline.

```bash
source .venv/bin/activate
python rl/train_single_task.py --task avoid_traps --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
python rl/train_single_task.py --task kill_monsters --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
python rl/train_single_task.py --task key_door --episodes 1 --max-steps 20 --action-repeat 4 --seed 0
```

Supported task directories are:

- `env_diy/map_data/dungeons/avoid_traps/room_001.json`
- `env_diy/map_data/dungeons/kill_monsters/room_001.json`
- `env_diy/map_data/dungeons/key_door/room_001.json`

Useful options:

```bash
python rl/train_single_task.py --task avoid_traps --room room_001 --episodes 5
python rl/train_single_task.py --config env_diy/map_data/dungeons/key_door/room_001.json --episodes 2
```

Episode logs include reward, length, `finish`, `terminated`, and `truncated`.
JSONL output defaults to `rl/outputs/single_task_training.jsonl`.

## PPO Training (Stable Baselines3)

Install the extra dependency first:

```bash
pip install stable-baselines3
```

Then run from the repository root:

```bash
python rl/train_ppo.py --total-timesteps 50000 --n-eval-episodes 5 --seed 0
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--total-timesteps` | 50 000 | Training timesteps |
| `--n-eval-episodes` | 5 | Greedy evaluation episodes after training |
| `--max-steps` | 500 | Max steps per evaluation episode |
| `--seed` | 0 | Random seed |
| `--action-repeat` | 1 | Repeat each policy action for multiple env ticks |
| `--config` | prototype dungeon | Dungeon config JSON path |
| `--save-path` | `rl/outputs/ppo_model` | Model save path (SB3 appends `.zip`) |
| `--output` | `rl/outputs/ppo_eval.jsonl` | Evaluation summary JSONL |
| `--render` | off | Call `env.render()` during evaluation |

Quick smoke run (1 000 timesteps):

```bash
python rl/train_ppo.py --total-timesteps 1000 --n-eval-episodes 2 --max-steps 100
```

**Notes**

- Uses `MultiInputPolicy` with a custom `DungeonFeaturesExtractor` (flattens all Dict obs keys and concatenates them) instead of SB3's default `CombinedExtractor`, which cannot handle obs keys that shadow Python built-ins such as `"keys"`.
- An `_EpisodeKeyAdapter` wrapper renames `info["episode"]` (an integer in `DungeonEnv`) to `info["dungeon_episode"]` so SB3's Monitor can safely store its own episode-stats dict under `info["episode"]`.

## Extension Notes

Keep dependency-light smoke scripts in `rl/` and shared helpers in `rl/utils/`. More complete algorithms can be added later as separate scripts, but should not replace the random policy smoke path.
