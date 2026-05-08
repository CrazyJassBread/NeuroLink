# Training Env DIY with DreamerV3

This document explains how to run the bundled DreamerV3 implementation on the
custom `env_diy` dungeon environment.

The current integration is intended for training, evaluation wiring, and smoke
testing. It verifies that DreamerV3 can reset the environment, step it, collect
transitions, write replay/checkpoints, initialize the learner, and enter the
training loop. It does not guarantee convergence on the dungeon task.

## 1. Relevant Files

- `env_diy/envs/dungeon_env.py`: Gymnasium-style dungeon environment.
- `world_model/dreamerv3/embodied/envs/env_diy.py`: adapter from `DungeonEnv`
  to DreamerV3's `embodied.Env` interface.
- `world_model/dreamerv3/dreamerv3/main.py`: DreamerV3 training entry.
- `world_model/dreamerv3/dreamerv3/configs.yaml`: `env_diy` config block.
- `tests/test_dreamerv3_env_diy_smoke.py`: adapter and make-env smoke tests.

## 2. Environment Interface

DreamerV3 does not consume Gymnasium environments directly. The adapter exposes
`env_diy` as an `embodied.Env` with:

### Observations

- `image`: `uint8`, shape `(64, 64, 3)`.
  - Source: `DungeonEnv.render()`, originally `(160, 160, 3)` HWC RGB.
  - Processing: nearest-neighbor resize.
  - No manual normalization. DreamerV3 handles image scaling internally.
- `vector`: `float32`, shape `(101,)`.
  - Flattened from `DungeonEnv` observation fields:
    - `grid`
    - `player_position_px`
    - `player_tile`
    - `health`
    - `gold`
    - `keys`
    - `inventory_ids`
    - `monsters_position_px`
    - `monsters_tile`
    - `monsters_active_mask`
    - `monsters_hp`
- `reward`: `float32`.
- `is_first`: true only on reset observations.
- `is_last`: true when an episode ends or the adapter time limit is reached.
- `is_terminal`: true only for environment terminal states.

The adapter does not expose a training `discount` field because this DreamerV3
implementation learns continuation from `is_terminal`. For logging, it emits
`log/discount = 0.0` on terminal states and `1.0` otherwise.

### Actions

DreamerV3 outputs a scalar discrete action. With the default
`agent_noop_enabled=false`, the adapter exposes six training actions and shifts
them to the base Gymnasium actions `1..6`:

| ID | Meaning |
|---:|---|
| `0` | up |
| `1` | down |
| `2` | left |
| `3` | right |
| `4` | A / interact |
| `5` | B / shield |

If `agent_noop_enabled=true`, the adapter exposes the base `Discrete(7)` action
set unchanged, including no-op action `0`. No one-hot conversion is needed.

### Rewards

The adapter uses the environment reward directly. It does not clip or scale
rewards. The same value is also logged as `log/reward_raw`.

### Logging

The adapter converts selected `info` fields and events into scalar `log/*`
metrics so DreamerV3 records them without storing them in replay:

- State metrics: `log/health`, `log/gold`, `log/keys`, `log/step`,
  `log/dungeon_episode`, `log/room_x`, `log/room_y`, `log/visited_rooms`,
  `log/has_key`, `log/no_progress_steps`, and `log/noop_mapped`.
- Success metric: `log/success`, currently based on `info["victory"]`.
- Event metrics: `log/opened_chest`, `log/got_key`, `log/got_gold`,
  `log/got_item`, `log/healed`, `log/pressed_button`,
  `log/room_transition`, `log/door_unlocked`, `log/blocked_locked`,
  `log/missing_requirement`, `log/trap_damage`, `log/monster_hit`,
  `log/monster_damaged`, `log/shield_block`, `log/monster_killed`,
  `log/game_over`, and `log/victory`.

## 3. Setup

Use the project virtual environment:

```bash
source .venv/bin/activate
```

Install the base project dependencies if needed:

```bash
python -m pip install -r requirements/requirements.txt
```

DreamerV3 also needs its own runtime dependencies. On this Python 3.13 setup,
the verified JAX stack is:

```bash
python -m pip install \
  elements portal ninjax einops granular scope tqdm colored_traceback ipdb \
  jaxtyping av ruamel.yaml jax==0.4.34 jaxlib==0.4.34 \
  chex==0.1.86 optax==0.2.3
```

Then verify dependency consistency:

```bash
python -m pip check
```

Expected:

```text
No broken requirements found.
```

## 4. Smoke Tests

Run the DreamerV3 adapter smoke tests:

```bash
python -m pytest -q tests/test_dreamerv3_env_diy_smoke.py
```

Run the broader environment and RL smoke tests:

```bash
python -m pytest -q \
  tests/test_dreamerv3_env_diy_smoke.py \
  tests/test_rl_smoke.py \
  tests/test_env_diy_env.py \
  tests/test_env_diy_renderer.py
```

## 5. Minimal DreamerV3 Smoke Run

This command uses the `debug` config, CPU execution, one environment, a tiny
batch, and a short run. It is intended to verify integration, not performance.

```bash
python world_model/dreamerv3/dreamerv3/main.py \
  --configs env_diy debug \
  --task env_diy_default \
  --logdir runs/dreamerv3_env_diy_smoke \
  --run.steps 20 \
  --run.envs 1 \
  --batch_size 2 \
  --batch_length 8 \
  --report_length 8 \
  --run.train_ratio 8 \
  --jax.platform cpu \
  --jax.prealloc False
```

Expected behavior:

- DreamerV3 prints `vector` and `image` observation spaces.
- Action space is `action: int32, low=0, high=7`.
- JAX initializes on CPU.
- Parameters initialize and train/report functions compile.
- A checkpoint directory is written under the selected `--logdir`.
- The training loop starts without observation/action format errors.

`runs/` is ignored by Git and can be deleted after smoke testing.

## 6. Longer Training

For a longer local run, start from:

```bash
python world_model/dreamerv3/dreamerv3/main.py \
  --configs env_diy \
  --task env_diy_default \
  --logdir runs/dreamerv3_env_diy_train \
  --jax.platform cpu \
  --jax.prealloc False
```

On a compatible GPU setup, omit the CPU overrides and configure JAX for the
available device.

Useful overrides:

- `--env.env_diy.image False`: vector-only training for faster smoke runs.
- `--env.env_diy.length 400`: adapter-side time limit.
- `--env.env_diy.move_speed_px 4`: default movement scale, applied as 1px collision sub-steps.
- `--env.env_diy.agent_noop_enabled False`: expose six shifted training actions so no-op is unavailable to the agent.
- `--env.env_diy.stuck_penalty_enabled False`: default disabled; enable only after baseline behavior is understood.
- `--run.envs 1`: simplest single-worker debugging.
- `--logger.outputs jsonl`: JSONL-only logs if Scope output is not needed.

Recommended initial exploration settings are `move_speed_px=4`,
`length=400`, no action repeat/repeat wrapper for `env_diy`, no stuck penalty,
and the existing movement reward design. The bundled `env_diy` config follows
these defaults.

## 7. Known Limitations

- The dungeon task has sparse rewards and long-range dependencies. Early
  training can collect many low-signal trajectories.
- `env_diy` currently uses a discrete action API, so movement plus shield cannot
  be represented in one action.
- `vector` includes symbolic and numeric state. This is useful for debugging and
  smoke training, but it changes the learning problem compared with pure pixels.
- The current adapter time limit produces `is_last=True` and
  `is_terminal=False`, matching a non-terminal truncation.
- This integration has been smoke-tested. It has not been tuned for final
  DreamerV3 performance or sample efficiency.
