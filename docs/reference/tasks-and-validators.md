# env_diy Tasks

This file documents task metadata, validator behavior, and task extension
points.

## Canonical Task Modules

- `env_diy/tasks/task_spec.py`
- `env_diy/tasks/validators.py`
- `env_diy/tasks/registry.py`

Room JSON task metadata is parsed through `env_diy/maps/tasks.py`. The
`tasks/` package is the runtime validation layer.

## Current Single Tasks

- `avoid_traps`
- `kill_monsters`
- `key_door`

`collect_coin` is reserved as an extension point, but there is no built-in map
for it in the current repo.

## Validator Output

Each validator returns:

- `success`
- `failure`
- `task_progress`
- `terminated_reason`
- `subgoal_status`
- `engine_done`
- `validator_done`
- `validator_matches_engine`

The wrapper exposes validator state through:

- `info["task"]["success"]`
- `info["task"]["failure"]`
- `info["task"]["progress"]`
- `info["task"]["terminated_reason"]`
- `info["task"]["subgoals"]`
- `info["debug"]["engine_done"]`
- `info["debug"]["validator_done"]`
- `info["debug"]["validator_matches_engine"]`

There is no `info["legacy"]` namespace anymore.

## Minimal Example

```python
from env_diy.tasks.task_spec import TaskSpec
from env_diy.tasks.validators import validate_task

task_spec = TaskSpec.from_task_config(env.task_config)
result = validate_task(
    task_spec,
    env.engine.runtime,
    [record["name"] for record in info["events"]["records"]],
    info["events"]["details"],
    engine_done=terminated,
)
```

## Adding a New Single-Task Map

1. Add `task_id`, `task_type`, `objective`, and optional `reward` to the room JSON.
2. Extend `env_diy/maps/tasks.py` if the schema needs a new objective field.
3. Register a validator in `env_diy/tasks/validators.py`.
4. Add coverage in `tests/test_task_completion.py`.

## Adding a Composite Task

Composite tasks follow the same path:

1. represent the task in map metadata
2. expose a stable objective description via `TaskSpec`
3. compute progress and terminal reasons in a validator
4. return the resulting state through the nested `task` and `debug` info fields
