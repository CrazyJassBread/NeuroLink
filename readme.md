# NeuroLink
![project preview](asserts/img/image.png)

NeuroLink is a reinforcement learning project built on top of PyBoy for The Legend of Zelda: Link's Awakening (GB).
The repository provides:
- Gymnasium-compatible custom environments.
- Training and evaluation scripts based on Stable-Baselines3 (PPO).
- Utility scripts for ROM state management and debugging.
- A standalone pygame prototype in env_diy.

## Project Structure

- asserts/
  - game_state/: ROM and saved states used during training/testing.
  - img/: images and references.
- orgin_env/
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
- env_diy/
  - standalone dual-resolution Zelda-like pygame prototype.

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
pip install -r requirements.txt
```

## ROM and State Files (Important)

Most scripts in this repository currently use relative paths like game_state/...
while checked-in assets are stored under asserts/game_state/.

You can choose one of the following:

1. Copy asserts/game_state to game_state in project root.
2. Or edit path constants in entry scripts (for example rl/train.py, rl/test.py, utils/*.py).

Expected files include:
- Link's awakening.gb
- Room58_task1.state, Room58_task2.state, Room51_task1.state, etc.

## Quick Start

### Train

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

### Test

```bash
python -m rl.test
```

Before testing, check rl/test.py:
- MODEL_PATH
- SAVE_STATE
- GAME_FILE

### Environment Sanity Check

```bash
python utils/test_env.py
```

This runs Stable-Baselines3 check_env on the selected environment.

### Manual Play and Save States

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

### Standalone Prototype (env_diy)

```bash
python -m env_diy.main
```

Optional room file override:

```bash
python -m env_diy.main --rooms env_diy/room_data/rooms.json
```

Controls:
- Arrow keys: move
- Z: button A (interact)
- X: button B (reserved)
- Esc: quit

## Create Your Own Task

1. Save or prepare a target emulator state (utils/play.py or utils/save_state.py).
2. Add a new environment file in orgin_env/ (for example env54.py) inheriting ZeldaEnv.
3. Implement check_goal and calculate_reward.
4. Export the class in orgin_env/__init__.py.
5. Switch imports/config in rl/train.py or rl/test.py to your new environment.
6. Validate with python utils/test_env.py.

## Notes and Common Pitfalls

- The folder name is orgin_env (current repo naming).
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