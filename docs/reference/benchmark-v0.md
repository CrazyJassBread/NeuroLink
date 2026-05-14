# env_diy Benchmark v0

This file defines the current benchmark v0 smoke suite only.
For broader benchmark process and maturity policy, use
`docs/project/development-guide.md`.

## Suite

The current suite is `NesyLink-v0`.

Tasks included:

- `prototype`
- `avoid_traps`
- `kill_monsters`
- `key_door`

This benchmark layer organizes existing environments only. It does not change
base movement or core game mechanics.

## Registry API

```python
from env_diy.benchmark import (
    list_suites,
    get_suite,
    list_tasks,
    get_task_spec,
    make_benchmark_env,
)
```

Examples:

```python
list_suites()
list_tasks("NesyLink-v0")
spec = get_task_spec("NesyLink-v0", "avoid_traps")
env = make_benchmark_env("NesyLink-v0", "avoid_traps", seed=0)
```

For task maps, `make_benchmark_env(...)` creates the env directly from
`map_path + reward_id/reward_module`. The `prototype` task uses the base
reward configuration.

## Random Eval

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 10 --seed 0
```

Save JSON:

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 10 --seed 0 --json-output /tmp/nesylink-v0.json
```

## Output Metrics

Per-task metrics:

- `episodes`
- `completion_rate`
- `death_rate`
- `truncation_rate`
- `mean_return`
- `mean_episode_length`
- `mean_reward_terms`
- `terminated_reason_counts`

Suite aggregate metrics:

- `mean_completion_rate`
- `mean_return`
- `mean_episode_length`

## Benchmark Task Metadata

Benchmark specs currently track fields such as:

- `suite_id`
- `task_id`
- `map_id`
- `map_path`
- `reward_id`
- `reward_module`
- `difficulty`
- `max_episode_steps`
- `observation_mode`
- `action_mode`
- `objective`

Task metadata lives in benchmark suite specs. It does not
come from map JSON or base `info`.
