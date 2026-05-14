# Env DIY Reward Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy task-driven reward wrapper flow with direct `map + reward` environment construction while preserving existing game mechanics and Gymnasium step/reset semantics.

**Architecture:** Keep `DungeonEngine` and room loading as the source of game mechanics, add a small map loader plus reward loader, move reward evaluation into the Gym wrapper, and make concrete reward modules thin subclasses of a shared `BaseReward`. Deprecate task registry and wrapper-era reward modules instead of using them on the main path.

**Tech Stack:** Python, Gymnasium, importlib, pytest, existing env_diy engine/room loader.

---

### Task 1: Add failing API and reward tests

**Files:**
- Modify: `tests/test_env_diy_architecture_boundaries.py`
- Modify: `tests/test_env_diy_reward_modules.py`
- Modify: `tests/test_rl_reward_module_config.py`
- Create: `tests/test_env_diy_base_reward.py`

- [ ] Add failing tests for `map_id`, `map_path`, `reward_id`, `reward_module`, reward reset, reward info payload, reward signals, and env termination/truncation rules.
- [ ] Run focused pytest commands to confirm failures are caused by missing new API behavior rather than test mistakes.

### Task 2: Add map loader and new env factory

**Files:**
- Create: `env_diy/maps/loader.py`
- Create: `env_diy/envs/factory.py`
- Modify: `env_diy/env.py`

- [ ] Implement `load_map(map_id=None, map_path=None)` with `map_path` priority and builtin search paths.
- [ ] Implement `make_env(map_id=..., map_path=..., reward_id=..., reward_module=..., reward_kwargs=..., max_steps=...)`.
- [ ] Keep `env_diy.env.make_env` as the compatibility public entrypoint delegating to the new factory.

### Task 3: Replace wrapper-era reward core with BaseReward

**Files:**
- Create: `env_diy/rewards/base.py`
- Create: `env_diy/rewards/sparse_exit.py`
- Create: `env_diy/rewards/collect_key.py`
- Create: `env_diy/rewards/collect_gold.py`
- Create: `env_diy/rewards/kill_monster.py`
- Create: `env_diy/rewards/exploration.py`
- Create: `env_diy/rewards/custom_template.py`
- Modify: `env_diy/rewards/loader.py`
- Modify: `env_diy/rewards/__init__.py`

- [ ] Implement `BaseReward` with `prev_obs`, `prev_info`, `extract_signals()`, `compute_reward()`, `extra_reward()`, and reward termination metadata.
- [ ] Implement thin builtin rewards with default weights and task-specific termination flags.
- [ ] Update reward loading to support builtin reward ids plus importlib-loaded modules exposing `make_reward(**kwargs)`.

### Task 4: Inline reward computation into Gym wrapper

**Files:**
- Modify: `env_diy/core/info.py`
- Modify: `env_diy/wrappers/gym_env.py`

- [ ] Extend base info with stable `game` flags needed by reward extraction.
- [ ] Reset the reward object during `env.reset()`.
- [ ] Compute reward during `env.step()`, merge reward-driven termination with engine termination, attach `info["reward"]`, and apply `max_steps` truncation.

### Task 5: Migrate training, benchmark, and Dreamer integration

**Files:**
- Modify: `rl/utils/env_factory.py`
- Modify: `rl/config/training.py`
- Modify: `rl/train_single_task.py`
- Modify: `rl/baselines/ppo.py`
- Modify: `env_diy/benchmark/specs.py`
- Modify: `env_diy/benchmark/suites.py`
- Modify: `env_diy/benchmark/registry.py`
- Modify: `env_diy/benchmark/eval.py`
- Modify: `env_diy/benchmark/metrics.py`
- Modify: `world_model/dreamerv3/embodied/envs/env_diy.py`

- [ ] Replace task-registry-derived env construction with direct `map_id/map_path + reward_id/reward_module`.
- [ ] Remove task progress/success bookkeeping assumptions from benchmark output where required.
- [ ] Update Dreamer adapter to read nested info fields and current event names.

### Task 6: Deprecate task-era modules and update docs

**Files:**
- Modify: `env_diy/tasks/__init__.py`
- Modify: `env_diy/tasks/registry.py`
- Modify: `env_diy/tasks/task_spec.py`
- Modify: `env_diy/tasks/reward_fns.py`
- Modify: `env_diy/tasks/wrappers.py`
- Modify: `env_diy/maps/tasks.py`
- Modify: `env_diy/README.md`
- Modify: `docs/reference/env-api.md`
- Modify: `docs/reference/rewards.md`
- Modify: `docs/reference/tasks-and-validators.md`
- Modify: `docs/guides/env-overview.md`
- Modify: `docs/guides/rl-training.md`
- Modify: `docs/guides/dreamer-training.md`
- Modify: `rl/README.md`
- Modify: `readme.md`

- [ ] Mark task modules deprecated or legacy-only so new code paths no longer depend on them.
- [ ] Rewrite docs around map-only JSON, reward modules, `BaseReward`, and new `make_env(...)` parameters.

### Task 7: Verify with focused and integration tests

**Files:**
- Test: `tests/test_env_diy_architecture_boundaries.py`
- Test: `tests/test_env_diy_base_reward.py`
- Test: `tests/test_env_diy_reward_modules.py`
- Test: `tests/test_rl_reward_module_config.py`

- [ ] Run the focused pytest subset for the refactor.
- [ ] Fix failures until the updated suite is green.
- [ ] Summarize modified, added, deprecated, and remaining compatibility surfaces with concrete verification evidence.
