# env_diy Dungeon Environment

`env_diy` is an original Gymnasium-compatible dungeon environment with pixel-level movement, JSON room configuration, procedural rendering, and a small pygame manual runner.

It does not use commercial game sprites, maps, names, music, or proprietary data. Keep new content original and programmatic.

## Package Layout

- `env_diy/envs/`: `DungeonEnv` and observation encoding.
- `env_diy/core/`: fixed constants, geometry, action ids, and colors.
- `env_diy/entities/`: player/object state, coordinate helpers, monster state, and monster AI updates.
- `env_diy/maps/`: map schema, validation, room state, and `RoomManager`.
- `env_diy/rendering/`: RGB renderer and procedural pixel icons.
- `env_diy/input/`: human keyboard input helper.
- `env_diy/app/`: pygame manual runner.
- `env_diy/map_data/`: example dungeon JSON files.

Prefer new imports such as:

```python
from env_diy.envs import DungeonEnv
```

Legacy imports such as `from env_diy.env import DungeonEnv` are kept as compatibility wrappers.

## Quick Start

```bash
source .venv/bin/activate
python -m env_diy.main
```

Use a custom dungeon index:

```bash
python -m env_diy.main --rooms env_diy/map_data/dungeons/prototype/dungeon.json
```

Controls:

- Arrow keys: held movement.
- `Z`: A / interact.
- `X`: B / shield. Held X repeats B and takes priority over movement.
- `Esc`: quit.

## Using the Environment for RL Training

Minimal Gymnasium smoke test:

```python
from pathlib import Path

from env_diy.envs import DungeonEnv

env = DungeonEnv(Path("env_diy/map_data/dungeons/prototype/dungeon.json"), render_mode="rgb_array")
obs, info = env.reset(seed=0)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

The observation is a Gymnasium `Dict` containing:

- `grid`: symbolic `8 x 10` dungeon grid.
- `player_position_px`, `player_tile`.
- `health`, `gold`, `keys`, `inventory_ids`.
- `monsters_position_px`, `monsters_tile`, `monsters_active_mask`.

Action ids are stable:

```text
0 = no-op
1 = up
2 = down
3 = left
4 = right
5 = A / interact
6 = B / shield
```

Every action advances one environment tick, including no-op, A, and B.
Movement actions use `move_speed_px=4` by default. The environment applies this
as up to four 1px collision-checked sub-steps, so walls, blocking objects, and
room-edge door checks keep their original behavior.

Default equipment:

- A slot: `interact`
- B slot: `shield`

The player also has lightweight `tools` and `equipped` state for future equipment work. These values are exposed in `info` and the HUD but are not part of the observation space.

Training diagnostics in `info` include `agent_pos`, `has_key`, `key_count`,
`picked_key`, `unlocked_door`, `entered_new_room`, `task_success`, and
`no_progress_steps`. Optional stuck penalty settings are available on
`DungeonEnv(...)` but are disabled by default.

The `Discrete(7)` action space cannot express move+shield at the same time. The pygame human runner uses the current v1 policy: held B means stand and guard, so X has priority over held movement.

### Stable-Baselines3 Example

This example requires `stable-baselines3` to be installed in the active environment.

```python
from pathlib import Path

from stable_baselines3 import PPO

from env_diy.envs import DungeonEnv


def make_env() -> DungeonEnv:
    return DungeonEnv(Path("env_diy/map_data/dungeons/prototype/dungeon.json"))


env = make_env()
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("runs/env_diy_ppo")
env.close()
```

If you use vectorized environments, construct each worker with its own `DungeonEnv(...)` instance.

## Adding or Modifying Rooms

`env_diy` uses a dungeon index JSON plus one JSON file per room.

Example index:

```json
{
  "schema_version": 1,
  "start_room": "room_0_0",
  "room_files": [
    "rooms/room_0_0.json",
    "rooms/room_1_0.json"
  ]
}
```

Room files live under the same folder as the index unless you choose another relative path. The prototype dungeon is in:

```text
env_diy/map_data/dungeons/prototype/
```

### Room Fields

```json
{
  "id": "room_0_0",
  "coord": [0, 0],
  "layout": [
    "..........",
    ".....#....",
    ".....#....",
    "...##.....",
    "..........",
    "......#...",
    "..........",
    ".........."
  ],
  "spawns": {
    "default": [1, 1],
    "from_west": [8, 4]
  },
  "default_spawn": "default",
  "objects": [],
  "exits": []
}
```

Rules:

- `layout` must be exactly `8` rows by `10` columns.
- `.` is floor and `#` is wall.
- Coordinates are `[col, row]`.
- Valid dungeon rows are `0..7`; rows `8..9` are HUD and cannot contain map entities.
- Object and spawn positions cannot overlap walls.
- Room ids and coordinates must be unique.

### Objects

Chest with gold:

```json
{
  "id": "chest_gold_1",
  "kind": "chest",
  "pos": [4, 2],
  "loot": {
    "kind": "gold",
    "amount": 5
  }
}
```

Chest with key:

```json
{
  "id": "chest_key_1",
  "kind": "chest",
  "pos": [8, 5],
  "loot": {
    "kind": "key",
    "amount": 1
  }
}
```

Heal chest:

```json
{
  "id": "chest_heal_1",
  "kind": "chest",
  "pos": [7, 1],
  "loot": {
    "kind": "heal",
    "amount": 1
  }
}
```

NPC:

```json
{
  "id": "npc_1",
  "kind": "npc",
  "pos": [7, 6],
  "text": "Find the south key."
}
```

Trap:

```json
{
  "id": "trap_1",
  "kind": "trap",
  "pos": [1, 5],
  "damage": 1,
  "respawn_to": "default"
}
```

Button:

```json
{
  "id": "button_1",
  "kind": "button",
  "pos": [2, 6],
  "message": "south gate ready"
}
```

Monster:

```json
{
  "id": "monster_1",
  "kind": "monster",
  "pos": [7, 4],
  "monster_type": "chaser",
  "hp": 2,
  "damage": 1
}
```

Supported object kinds are defined in `env_diy/maps/rooms.py`.

### Exits and Doors

Exit geometry is fixed by direction:

- north: `[(4, 0), (5, 0)]`
- south: `[(4, 7), (5, 7)]`
- west: `[(0, 3), (0, 4)]`
- east: `[(9, 3), (9, 4)]`

Directional entry spawn rules:

- `target_entry` may be a named spawn or a directional entry.
- Directional aliases are `north`, `south`, `west`, `east`, `from_north`, `from_south`, `from_west`, `from_east`, and `<direction>_entry`.
- If `target_entry` is omitted, it defaults to the opposite of the exit direction.
- Directional entries place the player one tile inside the target doorway:
  - north: `(4, 1)`, fallback `(5, 1)`
  - south: `(4, 6)`, fallback `(5, 6)`
  - west: `(1, 3)`, fallback `(1, 4)`
  - east: `(8, 3)`, fallback `(8, 4)`
- If both directional candidates are walls, dungeon validation fails.

Normal exit:

```json
{
  "id": "west_exit",
  "direction": "west",
  "target_room": "room_-1_0",
  "target_entry": "from_east",
  "type": "normal",
  "success_message": "WEST ROOM"
}
```

Locked key door:

```json
{
  "id": "east_exit",
  "direction": "east",
  "target_room": "room_1_0",
  "target_entry": "from_west",
  "type": "locked_key",
  "requires": {
    "key_count": 1,
    "consume_key": true
  },
  "blocked_message": "NEED KEY",
  "success_message": "EAST ROOM"
}
```

Locked-key doors have runtime state. They render as locked until the first successful unlock, then render as open/unlocked until reset. If `consume_key=true`, keys are consumed only on the first unlock; later passes through that already-open door do not consume keys again.

Conditional door requiring a button:

```json
{
  "id": "south_exit",
  "direction": "south",
  "target_room": "room_0_1",
  "target_entry": "from_north",
  "type": "conditional",
  "requires": {
    "button_pressed": "button_1"
  },
  "blocked_message": "PRESS BUTTON",
  "success_message": "SOUTH ROOM"
}
```

Conditional door requiring an item:

```json
{
  "id": "north_exit",
  "direction": "north",
  "target_room": "room_0_-1",
  "target_entry": "from_south",
  "type": "conditional",
  "requires": {
    "item": "lantern"
  }
}
```

When adding a room, also add its JSON path to the dungeon index `room_files`.

## Adding New Monsters

### Add an Existing Monster Type by Config

Current monster types are handled in `env_diy/entities/monsters.py`:

- `chaser`: moves toward the player.
- `ambusher`: activates when the player is within range.
- `patroller`: follows patrol points derived from spawn and `patrol_span`.

Example patroller:

```json
{
  "id": "monster_patrol_1",
  "kind": "monster",
  "pos": [6, 6],
  "monster_type": "patroller",
  "patrol_span": 32,
  "hp": 3,
  "damage": 1
}
```

Optional fields:

- `hp`
- `damage`
- `speed_px_per_step`
- `ambush_range`
- `patrol_span`
- `size_px`

### Add a New Monster Type in Code

For a new type such as `guard`, update:

1. `env_diy/entities/monsters.py`
   - Extend `build_monster_from_dict()` if the type needs custom initialization.
   - Extend `update_monster()` with the new behavior branch.
2. `env_diy/rendering/renderer.py` and/or `env_diy/rendering/sprites.py`
   - Add distinct rendering so the new monster is readable.
3. `tests/`
   - Add behavior tests for movement/combat if logic changes.
   - Add render tests if the monster has a distinct visual.
4. Documentation and worklog.

Keep monster movement deterministic and based on the environment tick, not wall-clock time.

## Changing A/B Button Functions

Current mappings:

- Gym action `5` is A / interact.
- Gym action `6` is B / shield.
- Keyboard `Z` maps to A.
- Keyboard `X` maps to B.

Input mapping for manual play lives in:

```text
env_diy/input/human.py
```

Environment action dispatch lives in:

```text
env_diy/envs/dungeon_env.py
```

Player labels are stored on `PlayerState` in:

```text
env_diy/entities/state.py
```

`PlayerState` owns the equipment state:

```python
tools = ["interact", "shield"]
equipped = {"A": "interact", "B": "shield"}
```

The HUD displays room id, HP, gold, items, `A:interact`, and `B:shield`.

### Change Manual Key Bindings

Edit `BUTTON_KEY_TO_ACTION`:

```python
BUTTON_KEY_TO_ACTION = {
    pygame.K_z: ACTION_A,
    pygame.K_x: ACTION_B,
}
```

This only changes keyboard input for the pygame runner. It does not change Gym action ids.

### Change A or B Behavior

In `DungeonEnv.step()`, actions dispatch through the equipped A/B tool:

```python
elif action == ACTION_A:
    events.append("action_a")
    action_reward, shield_active = self._handle_equipped_action(EquipmentSlot.A, events)
    reward += action_reward
elif action == ACTION_B:
    events.append("action_b")
    action_reward, shield_active = self._handle_equipped_action(EquipmentSlot.B, events)
    reward += action_reward
```

Recommended approach:

1. Add a new `ToolType`.
2. Keep `ACTION_A = 5` and `ACTION_B = 6` unless you are intentionally changing the public action space.
3. Equip it through `PlayerState.equip_tool(...)` or initialization/config code.
4. Add handling in `_handle_equipped_action(...)`.
5. Add tests showing the new event, reward, and state changes.
6. Update README/worklog.

Current shield semantics:

- B activates shield for the current environment tick.
- A shield block prevents monster contact damage for that tick.
- The monster is knocked back up to `16px` and stunned for the normal stun duration.
- Shield does not kill monsters and does not grant attack reward.
- `info["events"]` includes `shield_block`; `info["event_details"]` includes monster id, prevented damage, knockback, and stun ticks.

Future sword/attack behavior should be added as a new tool rather than changing the A action id.

## Validation and Tests

Run the full local suite:

```bash
source .venv/bin/activate
python -m pytest -q
```

Run focused env_diy lint:

```bash
source .venv/bin/activate
python -m ruff check env_diy tests
```

Manual import/render smoke:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path

from env_diy.envs import DungeonEnv

env = DungeonEnv(Path("env_diy/map_data/dungeons/prototype/dungeon.json"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)
frame = env.render()
print(frame.shape, env.hud_lines(), info["events"])
env.close()
PY
```

Expected frame shape is `(160, 160, 3)`.
