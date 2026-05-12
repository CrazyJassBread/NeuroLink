# Env DIY Game Guide

This guide documents the current `env_diy` Gymnasium dungeon environment for human players, RL researchers, and future developers. It is based on the current code in `env_diy/`, the prototype map config, and the test suite.

## 1. Overview

`env_diy` is an original top-down dungeon environment with a Gymnasium-compatible API. It uses JSON room files, pixel-level movement, tile-level map configuration, simple monster AI, procedural rendering, and a compact HUD.

The environment is useful for:

- human play through the pygame runner;
- RL smoke tests and baseline rollouts;
- future experiments with navigation, sparse rewards, room transitions, keys, doors, traps, monsters, and inventory-like state.

It does not use commercial sprites, maps, audio, or proprietary game data.

## 2. Game Screen and Map Layout

Fixed geometry:

| Value | Current Setting |
|---|---:|
| Tile size | `16 x 16` pixels |
| Map area | `10 x 8` tiles |
| HUD area | `10 x 2` tiles |
| Full screen | `10 x 10` tiles |
| Render size | `160 x 160` pixels |
| Map pixel bounds | `160 x 128` pixels |

Coordinate conventions:

- Tile coordinates are `(col, row)`, with valid dungeon rows `0..7` and columns `0..9`.
- Pixel positions are top-left `(x, y)` positions for dynamic entities.
- Player and monsters use `16 x 16` AABBs by default.
- Tile checks for traps, buttons, exits, and observations use the entity center tile, not the top-left tile.
- Rows `8..9` are HUD-only. Map objects cannot be placed there.

## 3. Core Gameplay

The player explores connected rooms, opens chests, presses buttons, collects gold/keys/items, uses keys to open locked doors, avoids traps, and survives monsters.

Current prototype progression examples:

- The start room has a conditional south exit requiring a button.
- The south room contains a key chest and a trap.
- The start room has a locked east door requiring a key.
- Door and room transition behavior is driven by room JSON config.

Prototype dungeon episodes can still end through legacy victory checks. Single-task rooms define explicit objectives; completing one returns `terminated=True` and reports `info["finish"] = True`.

## 4. Rooms, Exits, and Doors

Rooms are loaded from a dungeon index JSON plus one JSON file per room. Each room has fixed dimensions of `10 x 8` tiles.

Exit tiles are fixed by direction:

| Direction | Exit Tiles |
|---|---|
| north | `(4, 0)`, `(5, 0)` |
| south | `(4, 7)`, `(5, 7)` |
| west | `(0, 3)`, `(0, 4)` |
| east | `(9, 3)`, `(9, 4)` |

Exit types:

| Type | Behavior |
|---|---|
| `normal` | No requirement. Moving flush into the configured boundary exit transitions rooms. |
| `locked_key` | Requires `key_count` while locked. If `consume_key=true`, keys are consumed once on first unlock. |
| `conditional` | Requires config conditions such as `button_pressed`, `item`, or `all_monsters_defeated`. |

Room transitions:

- The player must be on one of the two exit tiles and flush with that room edge.
- Successful transition gives `+0.1` reward.
- Non-exit boundary movement is blocked and gives `-0.02`.
- Failed locked/conditional exits also give `-0.02`.

Directional entry spawn rules:

| Target Entry Direction | Spawn Candidates |
|---|---|
| `north` | `(4, 1)`, then `(5, 1)` |
| `south` | `(4, 6)`, then `(5, 6)` |
| `west` | `(1, 3)`, then `(1, 4)` |
| `east` | `(8, 3)`, then `(8, 4)` |

Aliases such as `west`, `from_west`, and `west_entry` are directional entries. Missing `target_entry` defaults to the opposite of the exit direction. If both directional spawn candidates are walls, map validation fails. Non-directional `target_entry` values are treated as named room spawns.

Locked door runtime state:

- Static `ExitConfig` is not mutated.
- Each `RoomState` has runtime `ExitRuntimeState`.
- Locked-key doors start locked on reset.
- Once opened, `unlocked=True` and `opened=True` persist until reset.
- Reusing an opened locked door does not consume another key.
- Render switches from locked-door art to open/unlocked art.

## 5. Player State

Current `PlayerState` fields include:

| Field | Default | Meaning |
|---|---:|---|
| `position_px` | room default spawn | Top-left pixel position. |
| `size_px` | `16` | AABB size. |
| `speed_px_per_step` | `2.0` | Legacy per-pixel actor speed field; RL movement now uses `DungeonEnv.move_speed_px`. |
| `health` / `max_health` | `5` / `5` | Health and game-over threshold. |
| `gold` | `0` | Gold count. |
| `keys` | `0` | Key count for locked doors. |
| `items` | `["sword", "shield"]` | Inventory item names encoded into observation IDs. |
| `tools` | `["interact", "shield"]` | Equip-capable tool list. |
| `equipped` | `{"A": "interact", "B": "shield"}` | Current A/B slot tools. |

> Note: The A slot defaults to `interact`; after chest/NPC interaction checks, it can attack an adjacent or overlapping monster for 1 HP.

Game over occurs when `health <= 0`.

## 6. Entities

Supported room object kinds:

| Entity | Current Behavior |
|---|---|
| Player | Pixel-moving controlled actor. |
| Monster | Dynamic enemy with chaser, ambusher, or patroller behavior. |
| Chest | Opens with A/interact when adjacent; grants configured loot once. |
| NPC | A/interact when adjacent sets message and event. |
| Trap | Deals damage when the player center tile overlaps it; may respawn player and deactivate if single-use. |
| Button | Pressed when the player center tile overlaps it; can satisfy conditional exits. |
| Door/Exit | Fixed two-tile room transition regions on room edges. |

Loot kinds:

| Loot Kind | Effect |
|---|---|
| `key` | Adds keys. |
| `gold` or unspecified | Adds gold. |
| `heal` | Restores health up to max. |
| `item` | Adds `item_id` to `player.items` if missing. |

## 7. Monster Mechanics

Monster defaults:

- Size: `16 x 16`.
- Default speed: `0.5 px/tick`, equal to `player_speed * 0.5`.
- Default damage: `1`.
- Default hp: at least `1`; monsters can be damaged by A/interact fallback attacks, shield contact, and damaging collisions.
- Stun duration: `60` environment ticks.
- Preferred knockback: `16px`, with fallback distances `12px`, `8px`, `4px`, or `0px`.

Monster types:

| Type | Movement Policy |
|---|---|
| `chaser` | Moves toward player position every tick. |
| `ambusher` | Activates when player is within `ambush_range` tiles, then chases. |
| `patroller` | Moves around generated patrol points based on spawn and `patrol_span`. |

Update and contact behavior:

- Monsters update even on `no-op`, A/interact, and B/shield actions.
- Stunned monsters decrement stun ticks and do not move.
- Monster overlap without shield deals damage, applies monster knockback/stun, adds `monster_hit`, and gives `-0.4`.
- Shield overlap prevents damage, applies the same monster knockback/stun, damages the monster by 1 HP, and adds `shield_block` or `monster_killed`.
- Stunned monsters do not apply contact damage.

## 8. Items and Interactions

A button:

- Action ID `5`.
- Dispatches the currently equipped A-slot tool.
- Default A tool is `interact`.
- Interact opens adjacent unopened chests or talks to adjacent NPCs first.
- If no chest or NPC can be used, A attacks an adjacent or overlapping monster for 1 HP.
- If nothing is adjacent, event `action_a_empty` is emitted and reward is `-0.01`.

B button:

- Action ID `6`.
- Dispatches the currently equipped B-slot tool.
- Default B tool is `shield`.
- Shield is active only for the current environment tick.
- Shield contact damages monsters and can kill them.

Buttons and traps:

- Buttons trigger by standing on their tile, no A press needed.
- Traps trigger by standing on their tile, no A press needed.

## 9. Reward Design

Rewards are now centralized in `env_diy/rewards/reward_fn.py`.

| Event | Reward | Notes |
|---|---:|---|
| Successful movement | `-0.01` | Movement actions run up to `move_speed_px` one-pixel collision sub-steps; default `4px`. |
| Blocked wall/bounds movement | `-0.02` | Includes non-exit boundary attempts. |
| A/interact no effect | `-0.01` | `action_a_empty`. |
| Unknown equipped A/B tool no effect | `-0.01` | Future extension path. |
| B/shield activation | `0.0` | Shield only affects later contact in same tick. |
| Room transition | `+0.1` | Added after movement reward, so a moving transition usually nets `+0.09`. |
| Failed locked/conditional door | `-0.02` | `blocked_locked` or `missing_requirement`. |
| Press button | `+0.1` | First press only. |
| Trap damage | `-0.5` by default, task `damage` reward in single-task rooms | May terminate if health reaches `0`. |
| Monster hit | `-0.4` by default, task `damage` reward in single-task rooms | Damage plus knockback/stun. |
| Shield block | `0.0` | Prevents monster contact damage and damages the monster. |
| Monster kill | `+0.3` by default, task `monster_kill` reward in single-task rooms | Removes the monster and adds gold. |
| Open chest with key loot | `+0.4` | Also emits `opened_chest`. |
| Open chest with heal loot | `+0.2` if actual healing, else `+0.05` | Also emits `opened_chest`. |
| Open chest with item loot | `+0.3` | Adds configured item if not already present. |
| Open chest with gold/default loot | `+0.2` | Adds gold. |
| Talk to NPC | `0.0` | Sets message. |
| Monster update | `0.0` | Adds `monsters_updated` if room has monsters. |
| Task finish | task `finish` reward, default `+10.0` | Emits `task_finished`, `victory`, and terminates. |
| Game over | no additional explicit reward | Existing trap/monster penalty applies first. |

Single-task room reward config supports `step`, `damage`, `key`, `door_unlock`, `monster_kill`, and `finish`. Missing keys use defaults.

## 10. Action Space

The current action space is `gymnasium.spaces.Discrete(7)`.

| Action ID | Meaning |
|---:|---|
| `0` | no-op / wait |
| `1` | move up |
| `2` | move down |
| `3` | move left |
| `4` | move right |
| `5` | A / equipped A tool, default `interact` |
| `6` | B / equipped B tool, default `shield` |

Action semantics:

- Every action advances exactly one environment tick.
- `no-op` does not move the player, but monsters still update.
- Movement is pixel-level, not tile jumps. `DungeonEnv` defaults to `move_speed_px=1.0`, so a movement action advances the player by 1 pixel per environment tick.
- Default monster speed is `0.5 px/tick`, derived from `player_speed * 0.5`.
- The canonical Gym wrapper also supports `action_repeat` directly. Default remains `1`, so the base environment still advances one tick per `step()`.
- A/interact and B/shield also allow monsters and contact checks to run.
- The current `Discrete(7)` API cannot express simultaneous movement plus shield.
- Human play maps held X to repeated B/shield with priority over held movement.

## 11. Observation Space

The observation space is `gymnasium.spaces.Dict`.

| Field | Space | Shape / dtype | Meaning |
|---|---|---|---|
| `grid` | `Box(0, 8)` | `(8, 10)`, `uint8` | Semantic room grid. |
| `player_position_px` | `Box` | `(2,)`, `float32` | Player top-left pixel position. |
| `player_tile` | `Box` | `(2,)`, `int32` | Player center tile. |
| `health` | `Box(0, 99)` | `(1,)`, `int32` | Current health. |
| `gold` | `Box(0, 9999)` | `(1,)`, `int32` | Gold count. |
| `keys` | `Box(0, 99)` | `(1,)`, `int32` | Key count. |
| `inventory_ids` | `Box` | `(2,)`, `int32` | First two `items` encoded by `ITEM_NAME_TO_ID`. |
| `monsters_position_px` | `Box(-1, max(screen))` | `(max_monster_slots, 2)`, `float32` | Monster top-left positions, padded with `-1`. |
| `monsters_tile` | `Box(-1, max(grid))` | `(max_monster_slots, 2)`, `int32` | Monster center tiles, padded with `-1`. |
| `monsters_active_mask` | `Box(0, 1)` | `(max_monster_slots,)`, `uint8` | Active monster slots. |

Grid tile codes:

| Code | Meaning |
|---:|---|
| `0` | empty |
| `1` | wall |
| `2` | player |
| `3` | monster |
| `4` | unopened chest |
| `5` | exit |
| `6` | active trap |
| `7` | button |
| `8` | NPC |

Observation notes:

- The observation includes the current room only, not the whole dungeon graph.
- The observation includes pixel-level player and monster positions.
- HUD content, current room id, equipment, and event history are not in observation; they are in `info` or render/HUD.
- `env.observation_space.contains(obs)` is expected to be true for `reset()` and `step()` outputs.

## 12. `reset(seed=None, options=None)`

Signature:

```python
obs, info = env.reset(seed=None, options=None)
```

Current behavior:

- Calls Gymnasium `super().reset(seed=seed)`.
- Rebuilds room runtime cache, clearing opened chests, pressed buttons, active trap changes, monster state, and locked-door runtime state.
- Starts in the dungeon index `start_room`.
- Places player at the start room default spawn.
- Recreates `PlayerState` with default health, gold, keys, items, tools, and equipped slots.
- Clears pending reset and resets `step_count` to `0`.
- Increments `episode`.
- Returns `events=["reset"]` and empty `event_details`.

Example:

```python
from pathlib import Path
from env_diy.env import DungeonEnv

env = DungeonEnv(Path("env_diy/map_data/dungeons/prototype/dungeon.json"))
obs, info = env.reset(seed=0)
assert env.observation_space.contains(obs)
```

## 13. `step(action)`

Signature:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

Current update order:

1. Validate the action with `action_space.contains`.
2. Auto-reset first if a previous episode ended and `auto_reset_on_step=True`.
3. Increment `step_count`.
4. Apply player action: movement, A tool, B tool, or no-op.
5. If a movement action occurred, check room transition.
6. Apply tile effects: button and trap.
7. Update monsters.
8. Resolve monster contact or shield block.
9. If health is `<= 0`, set `terminated=True`, mark pending reset, and add `game_over`.
10. Return a fresh observation and info dictionary.

`truncated` is currently always `False`; there is no built-in time limit wrapper inside `DungeonEnv`.

## 14. `info` Dictionary

Current `info` fields:

| Field | Meaning |
|---|---|
| `room_id` | Current room id. |
| `room_coord` | Current room coordinate tuple. |
| `health` | Player health. |
| `gold` | Player gold. |
| `keys` | Player keys. |
| `items` | Copy of `player.items`. |
| `tools` | Copy of equip-capable tools. |
| `equipped` | Copy of A/B equipped tool mapping. |
| `message` | Last HUD/status message. |
| `events` | List of string event names for this step. |
| `event_details` | List of structured dictionaries for selected events. |
| `episode` | Episode counter. |
| `step` | Current step count. |
| `player_position_px` | Player pixel position tuple. |
| `player_tile` | Player center tile tuple. |
| `auto_reset` | Present only when a step auto-reset occurred first. |
| `game_over` | Present and true on terminal game-over step. |
| `agent_pos` | Alias for current player pixel position, intended for RL diagnostics. |
| `has_key` | True when the player has at least one key. |
| `key_count` | Current key count. |
| `picked_key` | True on steps that collect a key. |
| `unlocked_door` | True on steps that unlock a locked door. |
| `entered_new_room` | True on room transition steps. |
| `task_success` | True on victory/task-success terminal steps. |
| `finish` | True on the step a configured single-task objective finishes. |
| `task_id` / `task_type` | Present for configured single-task rooms. |
| `no_progress_steps` | Consecutive steps without movement or configured progress events. |

Important: `info["events"]` is a list of strings, not a list of event objects. Structured fields are in `info["event_details"]`.

Example event strings:

- `reset`
- `noop`
- `move_up`, `move_down`, `move_left`, `move_right`
- `blocked_bounds`, `blocked_wall`
- `action_a`, `action_a_empty`
- `action_b`, `shield`, `shield_block`
- `opened_chest`, `got_key`, `got_gold`, `got_item`, `healed`
- `pressed_button`, `trap_damage`
- `blocked_locked`, `missing_requirement`
- `used_key`, `door_unlocked`, `room_transition`
- `monsters_updated`, `monster_hit`, `monster_damaged`, `monster_killed`
- `task_finished`, `victory`
- `game_over`

Structured detail examples:

```python
{
    "type": "room_transition",
    "from_room": "room_0_0",
    "to_room": "room_1_0",
    "exit_id": "east_exit",
    "exit_direction": "east",
    "target_entry": "from_west",
    "spawn_px": [16.0, 48.0],
}
```

```python
{
    "type": "task_finished",
    "task_id": "avoid_traps_001",
    "task_type": "avoid_traps",
    "objective_type": "reach_exit_without_trap_damage",
    "reward": 10.0,
}
```

```python
{
    "type": "door_unlocked",
    "room_id": "room_0_0",
    "direction": "east",
    "key_consumed": True,
}
```

```python
{
    "type": "shield_block",
    "monster_id": "monster_1",
    "damage_prevented": 1,
    "monster_knockback_px": 16,
    "knockback_applied_px": 16.0,
    "monster_stun_ticks": 60,
}
```

`event_details` is intentionally smaller than `events`; many events only appear as strings.

## 15. Termination and Game Over

Game-over behavior:

- If player health reaches `0`, the same step returns `terminated=True`.
- `truncated=False`.
- `info["game_over"] = True`.
- `pending_reset=True` is set internally.
- The terminal observation is returned from the game-over state.

Step after termination:

- With the default `auto_reset_on_step=True`, the next `step()` calls `reset()` internally first, then executes the new action.
- That follow-up info contains `auto_reset=True`.
- If `auto_reset_on_step=False`, stepping after termination raises `RuntimeError` until the caller calls `reset()`.

## 16. Rendering and Human Play

`render()` returns a NumPy RGB array.

| Property | Value |
|---|---:|
| Shape | `(160, 160, 3)` |
| dtype | `uint8` |
| Render modes metadata | `["rgb_array"]` |
| FPS metadata | `60` |

Render content:

- map background, floors, walls;
- connected two-tile exits/doors;
- procedural pixel icons for player, monsters, chests, loot hints, NPCs, traps, and buttons;
- HUD with room id, HP, gold, items, and A/B equipped tools.

Human play input:

| Key | Action |
|---|---|
| Arrow keys | Held movement; most recent direction wins. |
| `Z` | A / interact, edge-triggered. |
| `X` | B / shield, held/repeating and prioritized over movement. |
| `Esc` | Quit runner. |

## 17. Map Configuration Overview

Dungeon index files use schema version `1`:

```json
{
  "schema_version": 1,
  "start_room": "room_0_0",
  "room_files": ["rooms/room_0_0.json"]
}
```

Single-task challenge rooms can also be self-contained JSON files under
`env_diy/map_data/dungeons/{avoid_traps,kill_monsters,key_door}/room_001.json`.
They include normal room fields plus task metadata:

```json
{
  "task_id": "avoid_traps_001",
  "task_type": "avoid_traps",
  "room_id": "room_001",
  "objective": {"type": "reach_exit_without_trap_damage", "target_exit": "north_exit"},
  "reward": {"finish": 10.0, "step": -0.01, "damage": -1.0}
}
```

Supported task types are `avoid_traps`, `kill_monsters`, and `key_door`.
Supported objective types are `reach_exit`, `reach_exit_without_trap_damage`,
`kill_monsters`, and `key_door`.

Room files include:

- `id`
- `coord`
- `layout`
- `spawns`
- optional `default_spawn`
- optional `objects`
- optional `exits`

Validation checks include:

- layout is exactly `8` rows by `10` columns;
- layout tiles are only `.` or `#`;
- spawn/object positions are in bounds and not walls;
- object kinds are supported;
- room ids and coordinates are unique;
- exits use supported directions and types;
- normal exits do not declare requirements;
- conditional exits declare at least one supported condition;
- exit target rooms and target entries are valid;
- directional entry spawn candidates are not both blocked.

Minimal exit example:

```json
{
  "id": "east_exit",
  "direction": "east",
  "target_room": "room_1_0",
  "target_entry": "from_west",
  "type": "normal",
  "success_message": "EAST ROOM"
}
```

## 18. RL Training Notes

Current random smoke script:

```bash
source .venv/bin/activate
python rl/train_random.py --episodes 5 --max-steps 400 --action-repeat 4 --seed 0
```

Unified classic RL training entry:

```bash
python rl/train.py --method ppo --task-rooms prototype --total-timesteps 50000 --episodes 5 --seed 0
python rl/train.py --method ppo --task-rooms avoid_traps kill_monsters key_door --episodes 2
```

Environment creation:

```python
from pathlib import Path
from env_diy.env import DungeonEnv

env = DungeonEnv(Path("env_diy/map_data/dungeons/prototype/dungeon.json"))
```

RL considerations:

- `no-op` advances ticks and monsters can move.
- A/interact and B/shield also advance ticks.
- Movement is pixel-level (`move_speed_px=1.0` by default) while map layout and exits are tile-based.
- In `rl/` scripts, `max_steps` counts outer agent decisions. Actual environment ticks are approximately `max_steps * action_repeat`, unless termination or truncation stops early.
- `rl/train.py` selects classic RL methods via `--method`; PPO is implemented under `rl/baselines/ppo.py`, while DQN and A3C are registered placeholders until their baselines are added.
- The DreamerV3 adapter disables training no-op by default by exposing six shifted actions: training IDs `0..5` map to base environment actions `1..6`. Human play and direct Gymnasium use can still choose no-op.
- Optional stuck penalty is disabled by default. If enabled, consecutive no-progress steps beyond the configured threshold receive the configured small penalty.
- Current action space is discrete and cannot express movement plus shield simultaneously.
- Observation is current-room focused, not a full dungeon state.
- Reward shaping is simple and local; single-task rooms add explicit finish rewards and finish-rate diagnostics.
- `truncated` is always false unless an external wrapper adds time limits.
- Use `observation_space.contains(obs)` in smoke tests; current code expects it to pass.

## 19. Minimal Examples

### Create Environment

```python
from env_diy.env import make_env

env = make_env("env_diy/map_data/dungeons/prototype/dungeon.json", api="gym")
obs, info = env.reset(seed=0)
frame = env.render()
env.close()
```

### Random Rollout

```python
from env_diy.env import make_env

env = make_env("env_diy/map_data/dungeons/prototype/dungeon.json", api="gym")
obs, info = env.reset(seed=0)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Read Events From Info

```python
obs, reward, terminated, truncated, info = env.step(6)  # B / shield

if "shield_block" in info["events"]:
    details = [event for event in info["event_details"] if event["type"] == "shield_block"]
    print(details)
```

### Use RL Smoke Pipeline

```bash
source .venv/bin/activate
python rl/train_random.py --episodes 2 --max-steps 20 --seed 0
```

Single-task smoke:

```bash
python rl/train_single_task.py --task avoid_traps --episodes 1 --max-steps 20 --seed 0
python rl/train_single_task.py --task kill_monsters --episodes 1 --max-steps 20 --seed 0
python rl/train_single_task.py --task key_door --episodes 1 --max-steps 20 --seed 0
```
