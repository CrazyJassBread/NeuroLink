# env_diy Tasks

This file documents task metadata, runtime validator behavior, and task
extension points. Keep exact validator/output details here.

## Canonical Task Modules

- `env_diy/tasks/task_spec.py`
- `env_diy/tasks/validators.py`
- `env_diy/tasks/registry.py`

The map parser still loads task metadata from room JSON through `env_diy/maps/tasks.py`. The `tasks/` package is the runtime validation layer.

## Current Single Tasks

- `avoid_traps`
- `kill_monsters`
- `key_door`

`collect_coin` is registered as an extension point, but there is no built-in single-task map for it in the current repo.

## Validator Output

Each validator returns:

- `success`
- `failure`
- `task_progress`
- `terminated_reason`
- `subgoal_status`
- `legacy_done`
- `validator_done`
- `validator_matches_legacy`

The wrapper keeps both legacy and validator termination signals so migration can be checked safely.
At the moment, these comparison fields intentionally remain in `info` even though the canonical Gym wrapper can already use validator termination when parity holds.

## Minimal Example

```python
from env_diy.tasks.task_spec import TaskSpec
from env_diy.tasks.validators import validate_task

task_spec = TaskSpec.from_task_config(env.task_config)
result = validate_task(
    task_spec,
    env.engine.runtime,
    info["events"],
    info["event_details"],
    legacy_done=terminated,
)
```

## Adding a New Single-Task Map

1. Add `task_id`, `task_type`, `objective`, and optional `reward` to the room JSON.
2. Extend `env_diy/maps/tasks.py` if the schema needs a new objective field.
3. Register a validator in `env_diy/tasks/validators.py`.
4. Add coverage in `tests/test_task_completion.py`.

## Adding a Composite Task

Composite tasks should follow the same path:

1. represent the task in map metadata
2. expose a stable objective description via `TaskSpec`
3. compute progress and terminal reasons in a validator
4. keep legacy parity checks until the validator is trusted for termination
