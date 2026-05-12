# env_diy Reward

This file is the canonical reference for reward implementation and reward-mode
behavior. Keep exact reward semantics here instead of duplicating them in
README files.

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
    config=RewardConfig(reward_mode="legacy"),
)
```

## Reward Modes

- `legacy`
  - default mode
  - strictly preserves the prior reward behavior used by existing training scripts
- `event`
  - returns named terms such as `picked_key`, `opened_door`, `picked_coin`, `killed_monster`, `hit_trap`, `agent_dead`, `reached_goal`, `step_penalty`
- `sparse`
  - rewards only terminal success with the configured finish reward

## reward_terms

`reward_terms` is always returned as a dict. The total reward should equal:

```python
sum(reward_terms.values())
```

This is also copied into `info["reward_terms"]`. `info["reward_breakdown"]` is kept as a compatibility alias.
The default wrapper mode is always `reward_mode="legacy"` unless you opt into another mode explicitly.

## Extending Reward

To add a new event reward:

1. Make sure the engine emits a stable event name.
2. Update `env_diy/rewards/reward_fn.py`.
3. Add or extend a unit test in `tests/test_reward_fn.py`.
