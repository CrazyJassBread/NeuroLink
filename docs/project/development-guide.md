# NesyLink Development Guide

## 1. Purpose

NesyLink is an original reinforcement learning environment project centered on
`nesylink`, plus a benchmark layer built on top of it.

The project is developed as both:

1. a configurable, testable, extensible Gymnasium environment; and
2. a staged benchmark that becomes more reproducible over time.

This guide is the single canonical development guide for the repository.

## 2. Originality and Naming

The project must remain original.

Do not include:

- copyrighted characters
- copyrighted maps
- copyrighted room layouts
- copyrighted sprites
- copyrighted audio or music
- copyrighted item names
- proprietary commercial game logic copied verbatim

Prefer generic or original names such as:

- hero
- slime
- bat
- chest
- trap
- dungeon
- key
- door
- coin
- potion

## 3. Current Project Reality

The repository already contains more than a raw environment prototype.
It currently includes:

- a canonical Gymnasium wrapper under `nesylink.env.make_env(...)`
- deterministic reset and action sampling support
- pure map JSON loading through `RoomManager`
- task-agnostic base `info` and generic event emission
- direct map-plus-reward environment construction
- a lightweight benchmark layer under `nesylink/benchmark/`
- random-policy smoke scripts and PPO integration under `rl/`

When making decisions, optimize for the current repository reality first.
Do not force the codebase to match an aspirational folder layout unless there
is a clear implementation reason and an approved migration plan.

## 4. Benchmark Maturity Model

This project now uses staged benchmark goals instead of treating every
long-term benchmark feature as an immediate minimum requirement.

### 4.1 Current stage: benchmark v0 smoke suite

This is the current baseline and should remain healthy at all times.

Required properties:

- Gymnasium API passes
- random-policy smoke rollouts pass
- seeded determinism tests pass
- at least 3 official benchmark tasks exist
- benchmark task metadata is registered
- benchmark evaluation can run and write machine-readable output
- action space and observation space are documented

### 4.2 Next stage: benchmark v0.1 MVP

This is the next practical benchmark milestone.

Required additions:

- benchmark metadata is exposed consistently in benchmark outputs without
  polluting base env info
- evaluation writes a stable run artifact layout
- benchmark version is included in benchmark outputs
- README and benchmark docs clearly describe the benchmark entrypoints
- benchmark metrics and output schema are documented

### 4.3 Later stage: benchmark v0.2 reproducible suite

This is a larger benchmark milestone, not the current minimum bar.

Expected additions:

- fixed train, validation, test, and OOD splits
- split validation tests
- scripted expert or oracle baseline where practical
- PPO benchmark baseline result
- richer benchmark metrics beyond smoke aggregates
- reproducibility instructions for official reporting

### 4.4 Long-term stage: benchmark v1.0

This is the freeze point for official benchmark comparability.

Expected additions:

- frozen benchmark task suites
- frozen seed splits
- frozen scoring scripts
- published baseline results
- complete reproducibility instructions
- stable benchmark release/version policy

## 5. Canonical Repository Structure

Work with the current repository structure unless a migration is explicitly
approved.

```text
nesylink/
  __init__.py
  env.py              # public environment facade, including make_env(...)
  game.py             # pygame human-play / debug entrypoint
  core/               # runtime, state, world loading, mechanics, rendering, input
    world/            # room/map schema, parsing, validation, RoomManager
    mechanics/        # engine, movement, interactions, combat, progress
    rendering/        # frame renderer and procedural sprites
    input/            # human-play input helpers
  rewards/            # BaseReward, builtin rewards, reward loader
  wrappers/           # Gymnasium and Dreamer-facing adapters
  benchmark/          # current benchmark registry, metrics, evaluation
  tools/              # development-only utilities such as export_map.py

rl/
  utils/              # smoke training helpers
  baselines/          # classic RL integrations
  outputs/            # local RL outputs

docs/
  README.md
  guides/
  reference/
  project/
  worklog/
```

Prefer imports from:

```python
from nesylink.env import make_env
```

or from concrete subpackages such as:

```python
from nesylink.core.world import ...
from nesylink.rewards import ...
```

## 6. Required Workflow for Codex

Before making changes, Codex must:

- read this file
- read `README.md`
- inspect project structure
- read relevant source files
- read current tests
- read configuration and map examples when relevant
- read benchmark files if the change affects benchmark behavior
- produce a plan
- wait for human confirmation before editing

Do not edit code or docs before presenting the plan and receiving approval.

### 6.1 Plan format

Use this structure:

```markdown
## Plan

### Files to modify
- ...

### Why these changes are needed
- ...

### Expected behavior changes
- ...

### Benchmark impact
- Does this change affect action semantics?
- Does this change affect observation space?
- Does this change affect reward values?
- Does this change affect reward-driven termination conditions?
- Does this change affect seed determinism?
- Does this change affect official metrics?
- Does this change require a benchmark version bump?

### Compatibility impact
- ...

### Risks
- ...

### Test plan
- ...

### Evaluation plan
- ...
```

## 7. Coding Standards

Prefer:

- readability over cleverness
- explicitness over implicitness
- simple abstractions
- modular design
- small incremental changes

Avoid:

- giant files
- giant classes
- giant functions
- unnecessary metaprogramming
- hidden side effects
- behavior-changing refactors disguised as cleanup

Use type hints whenever practical.

Use clear names such as `player_health`, `current_room`, or `is_locked`.
Short names are acceptable only for standard terms such as `obs`, `env`, or
`rng`.

Prefer structured dataclasses and enums for:

- actions
- entity types
- item types
- event types
- observation modes
- benchmark splits

## 8. Gymnasium API Policy

The canonical environment must comply with Gymnasium.

Required interface:

```python
reset(seed=None, options=None)
step(action)
render()
close()
```

Required return values:

```python
obs, info = reset()
obs, reward, terminated, truncated, info = step(action)
```

Required properties:

```python
action_space
observation_space
```

For new code, the canonical entrypoint is:

```python
from nesylink.env import make_env
env = make_env(config_path, api="gym")
```

The canonical Gym wrapper should remain non-auto-reset by default.
Legacy convenience wrappers may keep auto-reset behavior only for explicit
compatibility.

## 9. Action and Tick Semantics

Keep the discrete action API stable unless a behavior change is explicitly
approved and versioned.

| ID | Action |
|---:|---|
| 0 | no-op |
| 1 | up |
| 2 | down |
| 3 | left |
| 4 | right |
| 5 | A / interact |
| 6 | B / shield |

Rules for `nesylink`:

- `env.step(action)` advances exactly one environment tick
- no-op, A/interact, and B/shield still advance monster AI, timers, collision,
  reward logic, and info generation
- keep the environment action space as `Discrete(7)` unless a broader migration
  is explicitly approved
- A and B dispatch through player equipment state
- default A is `interact`
- default B is `shield`
- equipment/tool state belongs in player state, `info`, and HUD
- do not add equipment/tool state to observations without an explicit
  observation-space migration

Human-play policy:

- held direction keys repeat movement every frame
- held B repeats shield and takes priority over movement
- if multiple direction keys are held and B is not held, the most recently
  pressed direction should win unless a different rule is documented and tested

Default speed policy:

- player speed unit: pixels per environment step
- default player movement: `1 px/tick`
- default monster speed: `player_speed * 0.5`

## 10. Observation Policy

Observation format must remain stable within a released benchmark version.

For `nesylink`:

- the observation covers only the `8 x 10` dungeon area
- the HUD region must never be treated as walkable map space
- pixel-level movement should be exposed explicitly where applicable

Preferred current structured observation keys:

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

Observation-space changes are benchmark-breaking unless versioned.

## 11. Reward Policy

Task reward logic must remain external to the base environment.

The canonical implementation lives under `nesylink/rewards/`.

Reward policy requirements:

- the base environment must not infer task type from map JSON, `map_id`, or
  base `info`
- reward calculation may depend on previous state, current state, events,
  task specification, and termination status
- reward terms should live on the wrapper, not on base `info`
- task success or task failure termination may be decided by a wrapper

Possible reward components include:

- step penalty
- blocked movement penalty
- damage penalty
- trap penalty
- item reward
- key reward
- door reward
- task completion reward
- death penalty

## 12. Info Schema Policy

The environment should expose clear structured info for RL and debugging.

Current stable top-level info fields should include:

- `episode`
- `env`
- `agent`
- `inventory`
- `entities`
- `events`
- `terminal_reason`
- `control`
- `debug`

Current stable nested fields should include, when applicable:

- `episode.id`
- `episode.step_count`
- `episode.seed`
- `env.map_id`
- `env.room_id`
- `entities.monsters_remaining`
- `events.records`
- `events.counts`
- `terminal_reason`
- `control.action_repeat`
- `control.inner_steps`

Task metadata belongs in wrapper state and benchmark outputs, not in base env
info.

## 13. Game Over and Episode End Policy

When player health reaches `0`:

- the current `step()` must return `terminated=True`
- the current `step()` must return `truncated=False`
- `info["terminal_reason"]` should identify the terminal cause
- the lethal step must return the terminal observation, not a hidden reset
  observation

If the caller invokes `step()` again before `reset()`:

- canonical Gym wrappers should not silently reset
- compatibility wrappers may preserve compatibility auto-reset if documented
- tests must verify the chosen behavior
- benchmark evaluation must not rely on hidden auto-reset behavior

Episode truncation should be used for:

- max episode steps
- time limit
- evaluation budget

Episode termination should be used for:

- player death
- world completion
- unrecoverable environment-level end states

## 14. Map and Geometry Policy

JSON is currently the active map format, but it is not the only possible future
format.

Before changing map format, present:

- rationale
- example files
- migration plan
- parser design
- validation strategy
- test plan
- benchmark compatibility impact

Current `nesylink` geometry policy:

- dungeon width: 10 tiles
- dungeon height: 8 tiles
- HUD height: 2 tiles
- total render canvas: 10 x 10 tiles
- tile size: `16 x 16`
- map pixel width: `160`
- map pixel height: `128`
- HUD pixel height: `32`
- render frame shape: `160 x 160 x 3`

Valid map coordinates:

- columns: `0..9`
- rows: `0..7`

Rows `8..9` are HUD-only and illegal in map config.

## 15. Room, Exit, and Door Policy

Room transitions must be configuration-driven.

General rules:

- only configured exits may change rooms
- colliding with a map boundary from a non-exit tile must not change rooms
- locked exits should report a clear blocked event when requirements are not met

Current fixed exit shape policy for `nesylink`:

- north exit tiles: `(4, 0)` and `(5, 0)`
- south exit tiles: `(4, 7)` and `(5, 7)`
- west exit tiles: `(0, 3)` and `(0, 4)`
- east exit tiles: `(9, 3)` and `(9, 4)`

Directional target entry policy:

- `north`: `(4, 1)`, `(5, 1)`
- `south`: `(4, 6)`, `(5, 6)`
- `west`: `(1, 3)`, `(1, 4)`
- `east`: `(8, 3)`, `(8, 4)`

Supported aliases:

- `<direction>`
- `from_<direction>`
- `<direction>_entry`

If all directional entry candidates are walls, validation must fail.

Exit state policy:

- distinguish `normal`, `locked_key`, and `conditional` exits in config and
  render output
- keep static config separate from runtime opened/unlocked state
- consume keys only on the first unlock when configured to do so
- later traversals through an open locked door must not consume keys again
- expose blocked reasons such as `blocked_locked` or `missing_requirement`

## 16. Dynamic Entity and Collision Policy

Dynamic entities use pixel/world coordinates. Static map content remains
tile-based.

Rules:

- player and monsters use `16 x 16` AABBs by default
- wall and bounds collision should use explicit AABB-vs-tile checks
- dynamic entities must remain inside the dungeon area only
- valid dynamic `x` range is `[0, 159]`
- valid dynamic `y` range is `[0, 127]`
- the HUD area is visual only and must never be entered by dynamic entities
- tile-triggered mechanics should use the center tile derived from pixel
  position

Monster contact damage policy:

- a valid overlap deals damage once and attempts monster knockback
- shield blocks prevent contact damage but still follow the same knockback/stun
  path
- shield blocks should emit `shield_block` plus structured details when
  available
- shield must not become an offensive kill mechanic unless explicitly added
- stun duration must be tick-based, not wall-clock based
- the primary protection against repeated contact damage should come from
  knockback plus stun, not from a long invincibility timer

## 17. Entity and Mechanic Policy

Player state should support:

- position
- health
- max health
- inventory
- gold
- key count
- equipped A tool
- equipped B tool

Mechanic expectations:

- movement respects walls, bounds, locked doors, and blocking entities
- invalid moves do not crash
- monsters continue updating even if the player does not move
- A and B are slot triggers, not hard-coded tool meanings
- default equipment is `A=sword`, `B=shield`
- A should prefer chest/NPC interaction before using the equipped A item
- sword is responsible for monster damage and kills
- shield is responsible for contact blocking, knockback, and stun only
- action poses may persist for several ticks for rendering, but must not apply repeated damage from one press
- traps emit event info and may damage, teleport, or disappear
- chests require interact, open once unless configured otherwise, and emit
  reward-relevant events
- buttons and switches may unlock exits, open doors, spawn items, or reveal
  keys

## 18. Validation Policy

All configs should validate explicitly.

Validation should catch:

- unknown entity type
- invalid coordinates
- duplicate IDs
- impossible exits
- malformed schema
- illegal HUD coordinates
- missing required fields
- invalid geometry or world-state config

Raise explicit exceptions instead of leaking generic parsing errors.

Preferred examples:

```python
InvalidDungeonConfigError
InvalidBenchmarkSplitError
InvalidEvaluationConfigError
UnsolvableGeneratedMapError
```

## 19. Task and Benchmark Policy

A benchmark task is more than a map.

Treat it as the combination of:

```text
environment mechanics
+ map or map generator
+ reward specification
+ observation mode
+ action mode
+ seed
+ maximum episode length
+ evaluation metrics
```

Changing any of the following may break comparability and may require a version
bump:

- physics
- action semantics
- observation space
- reward values
- success or failure conditions
- map layout
- generator behavior
- seed policy
- maximum episode length
- metrics or scoring

### 19.1 Current official benchmark scope

The current repository benchmark lives under `nesylink/benchmark/`.

Its current role is:

- a smoke benchmark registry
- fixed named tasks
- lightweight random-policy evaluation
- machine-readable benchmark outputs

It is not yet a fully frozen multi-split benchmark protocol.

### 19.2 Task specification policy

Task logic should be described separately from map geometry whenever practical.

A task specification should eventually capture:

- `task_id`
- `suite_id`
- `difficulty`
- `max_episode_steps`
- success conditions
- failure conditions
- subgoal definitions where applicable
- reward-function compatibility
- metric requirements

Current task logic is already code-side and should stay separate from base map
loading and base info generation.

### 19.3 Suite policy by phase

Benchmark v0 smoke:

- one or more suites are acceptable
- a small set of stable named tasks is enough
- tasks should be resettable, evaluable, and covered by tests

Benchmark v0.2 reproducible and later:

- fixed suite definitions
- documented split policy
- documented metric coverage
- frozen official task lists for release reporting

## 20. Seed and Split Policy

Determinism is required now. Fixed benchmark splits are a later phase
requirement.

Current requirement:

- seeded reset must be deterministic
- same seed plus same action sequence must be reproducible
- no hidden randomness without explicit seeding

Later benchmark requirement:

- fixed `train`, `validation`, `test`, and `ood_test` splits
- split validation tests
- no overlap between official splits

Do not present split support as already implemented unless the repository
actually contains it.

## 21. Metrics and Evaluation Policy

Evaluation requirements are also phased.

### 21.1 Current benchmark v0 smoke metrics

At minimum, benchmark evaluation should support:

- `success_rate`
- `failure_rate`
- `truncation_rate`
- `mean_return`
- `mean_episode_length`
- `mean_task_progress`

### 21.2 Benchmark v0.1 MVP additions

Add a stable run artifact layout such as:

```text
benchmark/outputs/<run_id>/
  config.json or config.yaml
  episodes.jsonl
  summary.json
```

### 21.3 Benchmark v0.2 and later additions

Later benchmark metrics may include:

- `normalized_score`
- split-specific scores
- subgoal metrics
- failure-stage metrics
- exploration and safety metrics

Do not require `normalized_score` until both random and expert baselines exist.

## 22. Baseline Policy

Differentiate between local training integrations and official benchmark
baselines.

Current repository reality:

- random-policy smoke scripts under `rl/` are required and must stay healthy
- PPO integration exists for training workflows
- DreamerV3 integration exists for experimentation

Official benchmark baseline requirements by phase:

- benchmark v0 smoke: random-policy result path exists
- benchmark v0.2 reproducible: random result plus at least one documented
  benchmark baseline result
- benchmark v1.0: official baseline set is frozen for reporting

Do not call a training integration an official benchmark baseline unless its
evaluation protocol and output are documented.

## 23. RL Smoke Script Policy

Lightweight RL smoke code should live under `rl/`.

Rules:

- prefer `nesylink.env.make_env(api="gym", ...)`
- keep `nesylink.env.DungeonEnv` only for compatibility auto-reset behavior
- keep smoke scripts dependency-light
- standard library, NumPy, and Gymnasium are enough for random rollout
  validation
- do not require rendering by default
- keep reusable helpers in `rl/utils/`
- keep smoke outputs under `rl/outputs/`
- do not remove the random-policy smoke path

## 24. Testing Requirements

Maintain or add tests for:

### 24.1 Core environment

- environment creation
- reset
- step output
- observation-space compatibility
- action-space compatibility
- render in headless mode
- close

### 24.2 Movement and transitions

- bounds
- walls
- exits
- invalid moves
- room transitions
- HUD boundary exclusion

### 24.3 Entities and mechanics

- traps
- chest open logic
- button trigger effects
- key and door logic
- monster movement
- monster contact damage
- shield block
- monster stun

### 24.4 Determinism and reward

- seeded reset determinism
- sampled-action determinism where applicable
- reward term correctness
- scalar reward equals reward term sum

### 24.5 Benchmark smoke

- benchmark registry coverage
- benchmark evaluation runs
- benchmark output schema for the current stage

Add stricter split, metrics, and solvability tests only when those benchmark
features are actually implemented.

## 25. Documentation Requirements

`README.md` should explain:

- install
- run example
- random agent example
- action space
- observation space
- config format
- tests

Current supporting docs should remain aligned with code:

- `docs/README.md`
- `docs/guides/env-overview.md`
- `docs/guides/training.md`
- `docs/reference/env-api.md`
- `docs/reference/rewards.md`
- `docs/reference/tasks-and-validators.md`
- `docs/reference/benchmark-v0.md`
- `nesylink/README.md`
- `rl/README.md`

When benchmark maturity changes, update the docs in the same change.

## 26. Versioning and Compatibility Policy

Use semantic versioning for benchmark-facing releases.

Patch changes:

- bug fixes
- documentation fixes
- test improvements
- non-behavior-changing refactors

Minor changes:

- adding new optional tasks
- adding optional metrics
- adding optional observation modes
- adding new baseline scripts without replacing official scoring

Major changes:

- action semantics
- observation space
- reward definitions
- official task success or failure conditions
- official seed split policy
- map geometry
- benchmark scoring
- official evaluation protocol

## 27. Refactor Rules

Avoid massive rewrites.

Prefer small incremental changes.

Each change should:

- preserve functionality unless behavior change is intentional
- keep tests passing
- maintain Gymnasium compatibility
- document benchmark impact when relevant
- avoid unversioned changes to action, observation, reward, or task semantics

## 28. Dependency Policy

Do not add dependencies without justification.

For each new dependency, explain:

- why it is needed
- alternatives considered
- package size and complexity
- effect on CI
- effect on installation friction
- whether it is required for core env, benchmark evaluation, rendering, or
  optional baselines

Core environment dependencies should remain minimal.

## 29. Performance Considerations

Prioritize correctness first.

Avoid:

- unnecessary deep copies
- avoidable `O(N^2)` scans
- excessive object recreation
- expensive rendering inside headless training loops unless requested
- excessive per-step logging in hot paths

Benchmark evaluation should report enough metadata for reproducibility without
making training loops unnecessarily slow.

## 30. Worklog Format

If the repository uses a worklog, update it after a session that changes code,
docs, or benchmark policy.

Use this format:

```markdown
## YYYY-MM-DD

### Summary
- ...

### Files Changed
- ...

### Behavior Changes
- ...

### Benchmark Impact
- ...

### Tests
- Command:
- Result:

### Known Issues / Next Steps
- ...
```

## 31. Safety and Legal Requirements

Keep all new assets, maps, tasks, and naming original.

All sprites should be procedural, simple, and original.
All maps should be original, generated, or manually designed without copying
commercial layouts.
All task names and entity names should be generic or original.
