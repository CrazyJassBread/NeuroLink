# RL Smoke Training

This folder contains lightweight RL smoke scripts for validating that `env_diy` can be used by training code. The current baseline is a random policy, not a serious learning agent.

## Random Policy Rollout

Run from the repository root:

```bash
source .venv/bin/activate
python rl/train_random.py --episodes 5 --max-steps 200 --seed 0
```

The script:

- creates `env_diy.envs.DungeonEnv` directly;
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

## Extension Notes

Keep dependency-light smoke scripts in `rl/` and shared helpers in `rl/utils/`. More complete algorithms can be added later as separate scripts, but should not replace the random policy smoke path.
