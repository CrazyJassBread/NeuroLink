# NeuroLink
![project preview](asserts/img/image.png)

NeuroLink is a reinforcement learning project built on top of PyBoy for The Legend of Zelda: Link's Awakening (GB).
The repository provides:
- Gymnasium-compatible custom environments.
- Training and evaluation scripts based on Stable-Baselines3 (PPO).
- Utility scripts for ROM state management and debugging.
- A standalone pygame prototype in env_diy.

## Project Structure

<details>
<summary>Show project structure</summary>

- asserts/
  - game_state/: ROM and saved states used during training/testing.
  - img/: images and references.
- env_diy/
  - app/: pygame interactive runner.
  - core/: constants and shared fixed geometry/action values.
  - entities/: entity state, coordinate helpers, monster state and monster AI updates.
  - envs/: Gymnasium `DungeonEnv` and observation encoding.
  - input/: human-play keyboard state helper.
  - maps/: map schema, validation, room templates, and `RoomManager`.
  - rendering/: RGB frame renderer and procedural pixel sprites.
  - utils/: small compatibility utilities.
  - map_data/: structured dungeon JSON examples.
- orgin_zelda/
  - envs/
    - base_env.py: shared ZeldaEnv base class.
    - env51_01.py, env51_02.py, env58.py, env58_02.py: room/task-specific environments.
    - config.py, emulator.py, observation.py, reward.py: modular environment components.
  - rl/
    - train.py: training entry (currently set to Room58_Task_Env by default).
    - test.py: evaluation/playback entry for trained PPO models.
    - PPO/model.py: custom model-related code.
  - utils/
    - play.py: manual play with optional state load/save hotkeys.
    - save_state.py: lightweight state capture script.
    - test_env.py: Gym environment sanity check using check_env.
    - count_monsters.py, game_area.py: observation/debug helpers.

</details>

## Setup

```bash
git clone https://github.com/CrazyJassBread/Link-s-awakening-RL.git
cd Link-s-awakening-RL
python -m venv .venv
```

Activate virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements/requirements.txt
```

## env_diy
Zelda-like prototype environment with a custom map format and Gymnasium API.

you can find more detail informations in [env_diy/README.md](env_diy/README.md).

### Standalone Prototype

```bash
python -m env_diy.main
```

Optional room file override:

```bash
python -m env_diy.main --rooms env_diy/map_data/dungeons/prototype/dungeon.json
```

Controls:
- Hold arrow keys: move continuously a few pixels every frame / attempt room exit
- If multiple directions are held, the most recently pressed direction wins
- Z: button A (`interact`, edge-triggered on keydown)
- X: button B (`shield`, held; while held it takes priority over movement)
- Esc: quit

### DIY Map Format

`env_diy` uses one structured dungeon JSON index file plus one JSON file per room under `env_diy/map_data/`.

Current fixed geometry:

- Dungeon area: `10 x 8` tiles
- HUD area: bottom `10 x 2` tiles
- Full render canvas: `10 x 10` tiles
- Tile size: `16 x 16` pixels
- Final render size: `160 x 160` pixels
- Dynamic entities use pixel/world positions inside the `160 x 128` dungeon area

Recommended structured room format:

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
  "objects": [
    {"id": "chest_1", "kind": "chest", "pos": [4, 2]},
    {"id": "npc_1", "kind": "npc", "pos": [7, 6], "text": "Example"},
    {"id": "monster_1", "kind": "monster", "pos": [7, 4], "monster_type": "chaser", "hp": 2, "damage": 1}
  ],
  "exits": [
    {
      "id": "east_exit",
      "direction": "east",
      "target_room": "room_1_0",
      "target_entry": "from_west",
      "type": "locked_key",
      "requires": {
        "key_count": 1,
        "consume_key": true
      }
    }
  ]
}
```

Validation currently checks:

- layout size and supported tiles for a fixed `10 x 8` dungeon area
- spawn/object coordinates
- all entity rows must remain within `0..7`; rows `8..9` are HUD and are illegal in config
- duplicate room IDs and room coordinates
- exit direction must be one of `north/south/west/east`
- exit target room and target entry references; missing `target_entry` defaults to the opposite direction
- conditional exits can only reference supported requirement fields
- positions overlapping wall tiles

Fixed two-tile exit regions:

- north: `[(4, 0), (5, 0)]`
- south: `[(4, 7), (5, 7)]`
- west: `[(0, 3), (0, 4)]`
- east: `[(9, 3), (9, 4)]`

Directional `target_entry` values place the player just inside the target doorway instead of at the room's default spawn. Supported aliases are the bare direction, `from_<direction>`, and `<direction>_entry`.

- entering from `north`: `(4, 1)`, then `(5, 1)` if the first tile is blocked
- entering from `south`: `(4, 6)`, then `(5, 6)`
- entering from `west`: `(1, 3)`, then `(1, 4)`
- entering from `east`: `(8, 3)`, then `(8, 4)`

If those directional candidates are walls, map validation fails. Non-directional entries still refer to named entries in `spawns`.

Exit types:

- `normal`: no requirement, entering either exit tile and moving into the boundary transitions immediately
- `locked_key`: checks `requires.key_count` while locked; if `requires.consume_key=true`, keys are consumed only on first unlock. The door stores runtime `unlocked/opened` state until reset and renders as open after unlocking.
- `conditional`: currently supports `requires.button_pressed` and `requires.item`

### DIY Environment Semantics

`env_diy` now exposes a Gymnasium-style environment at `env_diy.envs.DungeonEnv`.
The legacy import `env_diy.env.DungeonEnv` remains available as a compatibility wrapper.

- Static layout uses tile coordinates; player and monsters use top-left pixel coordinates.
- Player speed is `2.0 px/step`.
- Default monster speed is `1.0 px/step`, which is `player_speed * 0.5`.
- `0 = no-op`, `1 = up`, `2 = down`, `3 = left`, `4 = right`, `5 = A/interact`, `6 = B/shield`.
- Every action advances exactly one environment tick.
- `no-op` does not move the player, but monsters and environment logic still update.
- A defaults to the equipped `interact` tool. B defaults to the equipped `shield` tool.
- The player has lightweight `tools` and `equipped` state for A/B slots; this appears in `info` and HUD, not the observation space.
- `interact` and `shield` do not move the player, but monsters and environment logic still update.
- The Discrete action space cannot express move+shield simultaneously. Human play resolves held X as "stand and guard" priority over held movement.
- Observation is a Gymnasium `Dict`.
- `grid` remains a coarse `8 x 10` symbolic view of the dungeon area.
- `player_position_px`, `player_tile`, `health`, `gold`, `keys`, `inventory_ids`, `monsters_position_px`, `monsters_tile`, and `monsters_active_mask` expose the fine-grained dynamic state.
- HUD is visual only and is not part of the observation.
- Room switching only occurs through configured `exits`.
- Walking into a boundary from a non-exit tile returns a blocked event and keeps the player in place.
- Exits are always two tiles wide or high, centered on the room edge.
- Locked exits can require keys through `type=locked_key` and `requires.key_count`.
- Conditional exits can require a pressed button or held item through `type=conditional`.
- Trap, button, and exit checks use the player center tile derived from pixel position.
- Chest and NPC interaction use center-tile adjacency.
- Monster contact damage is prevented primarily by monster knockback and monster stun, not by a long player invincibility window.
- A damaging monster collision tries to knock the monster back by one tile (`16px`), falling back to shorter legal distances when necessary.
- A shield block prevents that contact damage for the current tick, applies the same monster knockback/stun path, emits `shield_block`, and does not kill or reward as an attack.
- After a valid monster collision, the monster enters a tick-based stun window and cannot move, chase, or deal contact damage until the stun expires.
- `info["events"]` remains a list of string events; `info["event_details"]` now reports structured collision fields such as `monster_id`, `damage`, `monster_knockback_px`, `knockback_applied_px`, and `monster_stun_ticks`.
- Failed door checks report blocked reasons in `info["events"]`, such as `blocked_locked` or `missing_requirement`.
- When health reaches `0`, the current `step()` returns `terminated=True`, `truncated=False`, and `info["game_over"] == True`.
- After termination, the environment sets an internal pending reset flag.
- If the caller invokes `step()` again without calling `reset()`, the environment automatically resets first, then executes the new action, and reports `info["auto_reset"] == True`.

### DIY Rendering

`env_diy.render()` returns an original `160 x 160` RGB array built from procedural pixel icons. It does not load or copy commercial sprites, maps, tiles, music, or other external game assets.

Current visual conventions:

- Player: small humanoid icon with head, body, outline, and shield highlight.
- Monsters: outlined blob icons with eyes; chaser, ambusher, and patroller types use distinct silhouettes/colors.
- Chests: closed and opened treasure chest shapes; chest loot may show a small key, coin/gold, or heal icon.
- Exits and doors: two-tile connected spans. Normal exits render as bright open passages, locked-key exits render as doors with a lock until unlocked, opened locked doors render as open/unlocked, and conditional exits render with a symbolic gate mark.
- Buttons: raised and pressed states use different height/color treatment.
- Traps: spike icons with warning color.
- Walls/floor: simple procedural tile treatment that keeps the map readable without changing fixed geometry.

The bottom HUD remains visual-only and shows room id, HP, gold, collected items, and equipped A/B tools. It uses compact pixel text and does not restore the old red health bar.

Headless render smoke test:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_env_diy_renderer.py
```

The HUD displays:

- current room id
- health
- gold
- collected items
- A slot tool
- B slot tool

## RL Smoke Training

The `rl/` folder contains a dependency-light random policy smoke pipeline for `env_diy`. It is intended to verify environment compatibility with RL-style code, not to train a strong agent.

```bash
source .venv/bin/activate
python rl/train_random.py --episodes 5 --max-steps 200 --seed 0
```

By default the script constructs `env_diy.envs.DungeonEnv` directly with `env_diy/map_data/dungeons/prototype/dungeon.json`, samples actions from `env.action_space`, validates observations, prints compact episode logs, and writes JSONL summaries to `rl/outputs/random_training.jsonl`.

Useful options:

```bash
python rl/train_random.py --config env_diy/map_data/dungeons/prototype/dungeon.json
python rl/train_random.py --episodes 2 --max-steps 20 --seed 0 --output rl/outputs/smoke.jsonl
```

Rendering is disabled by default for headless runs. Use `--render` only when you explicitly want `env.render()` called during rollout.

RL smoke tests:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_rl_smoke.py
```

See [rl/README.md](rl/README.md) for the current random baseline and extension notes.


## Tips For the Original Environment
### ROM and State Files (Important)

Most scripts in this repository currently use relative paths like game_state/...
while checked-in assets are stored under asserts/game_state/.

You can choose one of the following:

1. Copy asserts/game_state to game_state in project root.
2. Or edit path constants in entry scripts (for example rl/train.py, rl/test.py, utils/*.py).

Expected files include:
- Link's awakening.gb
- Room58_task1.state, Room58_task2.state, Room51_task1.state, etc.

### Quick Start

#### Train

```bash
python -m rl.train
```

Before training, check and adjust constants in rl/train.py:
- Zelda_Env import target.
- save_state and game_file paths.
- TOTAL_STEPS, tensorboard_log, and model save path.

TensorBoard:

```bash
tensorboard --logdir ./log
```

#### Test

```bash
python -m rl.test
```

Before testing, check rl/test.py:
- MODEL_PATH
- SAVE_STATE
- GAME_FILE

#### Environment Sanity Check

```bash
python utils/test_env.py
```

This runs Stable-Baselines3 check_env on the selected environment.

#### Manual Play and Save States

Recommended script:

```bash
python utils/play.py
```

Example with state load/save:

```bash
python utils/play.py --state game_state/Room58_task1.state --save-state game_state/manual.state
```

Default hotkeys:
- x: save state
- q: quit

Legacy quick script:

```bash
python utils/save_state.py
```

### Create Your Own Task

1. Save or prepare a target emulator state (utils/play.py or utils/save_state.py).
2. Add a new environment file in orgin_env/ (for example env54.py) inheriting ZeldaEnv.
3. Implement check_goal and calculate_reward.
4. Export the class in orgin_env/__init__.py.
5. Switch imports/config in rl/train.py or rl/test.py to your new environment.
6. Validate with python utils/test_env.py.

### Notes and Common Pitfalls

- The folder name is orgin_zelda (current repo naming).
- Path constants are not fully unified yet (asserts/game_state vs game_state).
- Some utility scripts may require small path edits for your local setup.
- If model loading fails in rl/test.py, verify MODEL_PATH points to an existing zip model.

## References

- Game memory info: https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)
- Game guide: https://www.zeldadungeon.net/
- PyBoy emulator: https://github.com/Baekalfen/PyBoy
- Related work: https://github.com/PWhiddy/PokemonRedExperiments
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3

## License

This project is for research and educational purposes only.
