# env_diy Benchmark v0

This file defines the current benchmark v0 smoke suite only.
For broader project process and benchmark maturity policy, use
`docs/project/development-guide.md`.

## Suite

The initial benchmark suite is `NesyLink-v0`.

Tasks included:

- `prototype`
- `avoid_traps`
- `kill_monsters`
- `key_door`

This benchmark layer organizes existing environments only. It does not change default movement, reward, or game mechanics.

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

`make_benchmark_env(...)` delegates to `env_diy.env.make_env(api="gym")`.

## Random Eval

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 10 --seed 0
```

Optional reward override:

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 10 --seed 0 --reward-mode event
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 10 --seed 0 --reward-mode sparse
```

Save JSON:

```bash
python -m env_diy.benchmark.eval --suite NesyLink-v0 --policy random --episodes 10 --seed 0 --json-output /tmp/nesylink-v0.json
```

## Output Metrics

Per-task metrics:

- `episodes`
- `success_rate`
- `failure_rate`
- `truncation_rate`
- `mean_return`
- `mean_episode_length`
- `mean_task_progress`
- `mean_reward_terms`
- `terminated_reason_counts`

Suite aggregate metrics:

- `mean_success_rate`
- `mean_return`
- `mean_episode_length`

## Reward Modes

- default benchmark reward mode is `legacy`
- `event` and `sparse` are optional evaluation modes
- benchmark v0 does not change the environment default reward behavior
