# env_diy Reward

This file is the canonical reference for reward implementation and reward-mode
behavior.

## Canonical Path

There is exactly one real reward implementation:

[`env_diy/rewards/reward_fn.py`](../env_diy/rewards/reward_fn.py)

## Entry Point

```python
from env_diy.rewards import RewardConfig, compute_reward

reward, reward_terms = compute_reward(
    prev_state,
    next_state,
    engine_result,
    task_spec=task_config,
    config=RewardConfig(reward_mode="default"),
)
```

## Reward Modes

- `default`
  - default mode
  - preserves existing reward behavior used by current training code
- `event`
  - returns named terms such as `picked_key`, `opened_door`, `picked_coin`,
    `killed_monster`, `hit_trap`, `agent_dead`, `reached_goal`, `step_penalty`
- `sparse`
  - rewards only terminal success with the configured finish reward

## Reward Reporting

`reward_terms` is always returned as a dict. The wrapper stores:

- the scalar reward in `info["reward"]["total"]`
- the decomposition in `info["reward"]["terms"]`
- the selected mode in `info["reward"]["mode"]`

The scalar step reward should equal:

```python
sum(reward_terms.values())
```

There are no flat compatibility aliases such as `info["reward_terms"]`,
`info["reward_breakdown"]`, or `info["legacy"]["reward_terms"]`.

## Extending Reward

To add a new event reward:

1. Make sure the engine emits a stable event name.
2. Update the centralized reward rule tables in `env_diy/rewards/reward_fn.py`.
3. Add or extend a unit test in `tests/test_reward_fn.py`.
