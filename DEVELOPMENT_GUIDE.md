# DEVELOPMENT_GUIDE.md

# RL Dungeon Environment Development Guide

## 1. Project Overview

This project is an original Python-based Reinforcement Learning environment compatible with Gymnasium.

The environment is inspired by classic top-down 2D dungeon exploration mechanics from retro handheld-era games, but it must remain original.

The project should NOT copy or reproduce:

- copyrighted characters
- copyrighted maps
- copyrighted room layouts
- copyrighted sprites
- copyrighted audio/music
- copyrighted item names
- proprietary game logic from commercial games

The goal is to build a configurable, testable, extensible RL dungeon environment.

---

## 2. Primary Development Goals

The project should prioritize:

### 2.1 Gymnasium compatibility

The environment must comply with Gymnasium API.

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
### 2.2 RL-friendly design

The environment should be easy for RL algorithms to use.

Requirements:

- deterministic reset when seed is provided
- stable observation format
- stable action semantics
- centralized reward logic
- no hidden randomness without seeding

### 2.3 Extensibility

The system should allow future expansion for:

- more monster types
- more item types
- new interaction mechanics
- procedural generation
- partial observability
- multiple dungeons
- curriculum learning

## 3. Required Workflow for Codex

Before making any changes:

- Read this file.
- Read README.md.
- Inspect project structure.
- Read relevant source files.
- Read current tests.
- Read configuration/map examples.
- Produce a plan.
- Wait for human confirmation.

Do NOT edit code before presenting a plan.

## 4. Plan Format

The plan must include:
```markdown
## Plan

### Files to modify
- ...

### Why these changes are needed
- ...

### Expected behavior changes
- ...

### Risks
- ...

### Test plan
- ...
```

Only execute after approval.

## 5. Coding Standards
### 5.1 General Principles

Prefer:

- readability over cleverness
- explicitness over implicitness
- simple abstractions
- modular design

Avoid:

- giant files
- giant classes
- giant functions
- unnecessary metaprogramming
- hidden side effects

### 5.2 Naming

Use clear names.

Examples:

Good:
```python
player_health
current_room
is_locked
```
Bad:
```python
hp
cr
f
```

### 5.3 Type Hints

Use type hints whenever practical.

Example:
```python
def move_player(action: int) -> None:
    pass
```

### 5.4 Dataclasses / Enums

Prefer structured types:
```python
@dataclass
class Monster:
    ...
```

Use Enum for:

- actions
- entity types
- item types
- event types

## 6. Map / Dungeon Configuration Policy

The project currently may use JSON, but JSON is not mandatory.

Before changing the map system, evaluate alternatives.

Possible options:

- JSON
- YAML
- TOML
- ASCII layout + metadata
- Tiled Map Editor export
- Python DSL
- hybrid approaches

Preferred criteria:

- easy to edit
- easy to validate
- easy to diff in Git
- easy to test
- deterministic loading
- minimal dependencies
- scalable

Do NOT migrate formats without first presenting:

- rationale
- example files
- migration plan
- parser design
- validation strategy
- test plan

## 7. Recommended Map Format

Preferred short-term approach:

ASCII layout + metadata.

Example:
```json
{
  "room_id": "room_001",
  "layout": [
    "########",
    "#P..C..#",
    "#..T...#",
    "#..B..E#",
    "########"
  ],
  "legend": {
    "#": "wall",
    "P": "spawn",
    "C": "chest",
    "T": "trap",
    "B": "button",
    "E": "exit"
  }
}
```
Benefits:

- human-readable
- easy to debug
- easy to diff

Current `env_diy` geometry policy:

- dungeon area width: 10 tiles
- dungeon area height: 8 tiles
- HUD height: 2 tiles
- total render canvas: 10 x 10 tiles
- tile size: 16 x 16 pixels
- map pixel width: 160
- map pixel height: 128
- HUD pixel height: 32

## 8. Rendering Policy

`env_diy` rendering must remain original and procedural. Do not copy or imitate commercial game sprites, maps, or UI assets.

Current renderer expectations:

- `render()` returns a `160 x 160 x 3` RGB array and must work in headless tests.
- Drawing helpers should live outside `DungeonEnv`; environment logic should not own sprite/icon details.
- Use lightweight primitives or small code-defined pixel icons instead of external image assets.
- Keep player, monsters, chests, exits/doors, keys, coins/gold, heal items, traps, and buttons visually distinguishable inside a `16 x 16` tile.
- Normal exits, locked-key doors, and conditional doors should remain visually distinct, and two-tile exits should read as one connected doorway.
- HUD rendering should stay compact and show room id, HP, gold, and collected items without reintroducing a red health bar.

Recommended render smoke test:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_env_diy_renderer.py
```

Exit policy for `env_diy`:

- exits are fixed two-tile regions centered on the room edge
- north exit tiles: `(4, 0)` and `(5, 0)`
- south exit tiles: `(4, 7)` and `(5, 7)`
- west exit tiles: `(0, 3)` and `(0, 4)`
- east exit tiles: `(9, 3)` and `(9, 4)`
- room connectivity must still come from map config; only the exit shape/placement rule is centralized

## 8. Input and Tick Semantics

For `env_diy`, keep these rules stable unless a task explicitly changes them and also updates tests/docs:

- `env.step(action)` always advances exactly one environment tick.
- `0 = no-op`, `1 = up`, `2 = down`, `3 = left`, `4 = right`, `5 = interact`, `6 = B/reserved`.
- `no-op`, `interact`, and `B` still advance monster AI, stun timers, collision checks, reward logic, and info generation.
- Human play may translate held keyboard state into per-frame actions, but that logic belongs in the interactive runner or input helper, not in the Gymnasium API itself.
- In the pygame runner, held direction keys should repeat movement every frame.
- If multiple direction keys are held, the most recently pressed direction should win unless a different policy is intentionally documented and tested.
- Default monster speed should remain `player_speed * 0.5` unless a specific monster overrides its own speed in config.

## 9. Dynamic Entity Collision Policy

Dynamic entities use pixel/world coordinates. Static map content remains tile-based.

- Player and monsters use pixel-level positions and `16 x 16` AABBs by default.
- Wall/bounds collision should resolve through explicit AABB-vs-tile checks.
- Dynamic entities must remain inside the dungeon area only:
  - `x in [0, 159]`
  - `y in [0, 127]`
- The HUD area (`y in [128, 159]`) is visual only and must never be entered by dynamic entities.
- Tile-based triggers such as traps, buttons, and exits should continue to use the entity center tile derived from pixel position.

Monster contact damage policy:

- A valid monster overlap deals damage once and attempts to knock the monster away from the player.
- The preferred knockback distance is one full tile (`16px`), but the environment may fall back to shorter legal distances such as `12px`, `8px`, `4px`, or `0px`.
- After a valid hit, the monster enters a tick-based stun window and must not move, chase, or apply contact damage while stunned.
- Stun duration must be based on environment ticks, not wall-clock time.
- The environment should expose these outcomes clearly in `info`, while keeping compatibility with existing string event lists.
- The primary protection against repeated damage should come from monster knockback plus monster stun, not from a long player invincibility timer.
- Knockback must still obey walls, map bounds, and the HUD boundary.

Exit and door policy:

- Distinguish `normal`, `locked_key`, and `conditional` exits in both config and render output.
- `normal` exits have no requirements.
- `locked_key` exits should use explicit requirement fields such as `key_count` and optional `consume_key`.
- `conditional` exits should use explicit requirement fields such as `button_pressed` or `item`.
- When requirements are not satisfied, the player must remain in the current room and `info` should expose a blocked reason such as `blocked_locked` or `missing_requirement`.

Map configuration coordinates must only target the dungeon area.

- valid columns: `0..9`
- valid rows: `0..7`
- rows `8..9` are reserved for HUD and are illegal in map config
## 8. Environment Mechanics
### 8.1 Player

The player should have:

- position
- health
- max health
- inventory
- gold
- key count

### 8.2 Movement

Movement must respect:

- walls
- room bounds
- locked doors
- blocking entities

For `env_diy`:

- static map layout remains tile-based
- dynamic entities use pixel/world positions
- default player movement is measured in pixels per environment step
- no-op still advances the environment tick
- monsters must continue updating even when the player does not move

Invalid moves should not crash.

### 8.3 Monsters

Monsters should support:

- id
- type
- position
- health
- damage

Optional future:

- patrol
- random walk
- chase behavior

Default speed policy:

- player speed unit: pixels per environment step
- default monster speed: `player_speed * 0.8`
### 8.4 Traps

When stepped on:

- deal damage
- optional teleport to room entry
- optional disappear after trigger

Must produce event info.

### 8.5 Chests

Rules:

- require interact action
- may contain:
- keys
- items
- healing
- gold
- can only open once unless repeatable
### 8.6 Buttons / Switches

Can trigger:

- spawn item
- unlock exit
- open door
- activate bridge
- reveal key

Can be one-time or repeatable.

### 8.7 Keys / Doors

Keys can:

- unlock exits
- unlock doors
- trigger progression
### 8.8 Room Transition

Room transitions must be configuration-driven.

- only configured exits may change rooms
- exits should declare direction and target spawn
- colliding with a map boundary from a non-exit tile must not change rooms
- locked exits should report a clear blocked event when requirements are not met

## 9. Action Space Policy

Unless explicitly changed:

ID	Action
0	no-op
1	up
2	down
3	left
4	right
5	button A (use for attack or interact)
6	button B (use for shield or reserved for future)

Do not change without updating tests/docs.

## 10. Observation Space Policy

Observation format must remain stable.

Possible formats:

Grid format
- Box(...)
Dict format
- Dict(...)

Including:

- player position
- health
- inventory
- visible entities
- room grid

For `env_diy`, the observation should cover only the `8 x 10` dungeon area.
The bottom HUD region must never be included as walkable map space.

If dynamic entities use pixel-level movement, the observation should expose that explicitly.

Preferred current format for `env_diy`:

- `Dict`
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

Must document clearly.

## 11. Reward Policy

Reward logic should be centralized.

Possible components:

- step penalty
- damage penalty
- trap penalty
- item reward

## 12. Game Over Policy

When player health reaches `0`:

- the current `step()` must return `terminated=True`
- the current `step()` must return `truncated=False`
- `info` must contain a clear game-over flag such as `game_over=True`
- the lethal step should return the terminal observation, not a hidden reset observation

If the caller invokes `step()` again before calling `reset()`:

- automatic reset is allowed
- that behavior must be documented
- tests must verify the chosen behavior
- chest reward
- key reward
- room completion reward
- dungeon completion reward
- death penalty

Avoid hardcoding across files.

## 12. Validation Policy

All configs must validate.

Validation should catch:

- unknown entity type
- invalid coordinates
- duplicate IDs
- impossible exits
- malformed schema

Raise explicit exceptions.

Bad:
```python
KeyError
```
Good:
```python
InvalidDungeonConfigError
```

## 13. Testing Requirements

Add tests for:

Core Environment
 - creation
 - reset
 - step output
Movement
 - bounds
- walls
- exits
Traps
- damage
- teleport
Chests
- open logic
- rewards
Buttons
- trigger effects
Keys / Doors
- unlock logic
Config Loader
- valid config
- invalid config
Determinism
- seeded reset

## 14. Documentation Requirements

README should explain:

- install
- run example
- random agent example
- action space
- observation space
- config format
- tests
## 15. File Structure Recommendation

Example:

envdiy/
│
├── env/
│   ├── dungeon_env.py
│   ├── mechanics/
│   ├── entities/
│   ├── config/
│
├── maps/
│
├── tests/
│
├── README.md
├── DEVELOPMENT_GUIDE.md
## 16. Refactor Rules

Avoid massive rewrites.

Prefer:

small incremental changes.

Each change should:

- preserve functionality
- keep tests passing
## 17. Worklog Format

After each session:
```markdown
## YYYY-MM-DD

### Summary
- ...

### Files Changed
- ...

### Behavior Changes
- ...

### Tests
- Command:
- Result:

### Known Issues / Next Steps
- ...
```

## 18. Dependency Policy

Do not add dependencies without justification.

For each new dependency explain:

- why needed
- alternatives considered
- package size / complexity

## 19. Performance Considerations

Prioritize correctness first.

But avoid:

- unnecessary deep copies
- O(N²) scans if avoidable
- excessive object recreation
## 20. Safety / Legal

The project must remain original.

Do not include:

- copyrighted names
- copyrighted maps
- copyrighted room layouts
- copyrighted art/audio/assets

Use generic names like:

- hero
- slime
- bat
- chest
- trap
- dungeon
