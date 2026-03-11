# NeuroLink
![alt text](img/image.png)

A reinforcement learning project built on the Game Boy emulator PyBoy to train agents to solve rooms in "The Legend of Zelda: Link’s Awakening". The repo includes custom Gym-style environments, training/testing scripts and state management utilities.

## Project Structure
- game_state/
  - Saved emulator states and room checkpoints used for training/testing.
  - you may need to find the rom file by yourself 😀
- envs/
  - base_env.py: Base environment implementing Gym API and common helpers.
  - room-specific envs with custom reward functions.
  - test_env: Use `python -m envs.test_env` to check the specific env you make
- rl/
  - train.py: Train an RL agent.
  - test.py: Evaluate a trained agent.
- utils/
  - save_state.py: save the game states that you may need.

## Set up

```bash
git clone https://github.com/CrazyJassBread/Link-s-awakening-RL.git
cd Link-s-awakening-RL.git
# make the virtual environment
python -m venv .venv
source .venv/bin/activate # Linux Mac
# use ".\.venv\Scripts\Activate.ps1" for windows powershell
pip install -r requirements.txt
```

## Data and ROM
Place the game ROM and states under game_state/:
- game_state/Link's awakening.gb
- game_state/Room58.state (for example)

## Usage

### Train the RL model you need
```bash
python -m rl.train
```
- Configure hyperparameters and environment selection inside RL/train.py.

you can use the tensorboard to check the training process
```bash
tensorboard --logdir=./log
```

### Test
```bash
python -m rl.test
```
- Loads the trained model and runs evaluation in the selected room environment.

### Manual Play & utils:
```bash
python utils/save_state.py
```
keyboard control:
- 'q' to quit.
- 'x' to save the state
- 'up' 'down' 'left' 'right' to control the Link
- 'a' 's' to use the item a, item b

### Create your own task
you can make your own env using the `utils/save_state.py`. After saving your target room, you can write a new file (eg:env_54.py) and place it in the folder `RL/envs/ `

## References
- Game memory info: https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)
- Game guide: https://www.zeldadungeon.net/
- PyBoy emulator: https://github.com/Baekalfen/PyBoy
- Related work: https://github.com/PWhiddy/PokemonRedExperiments
- stable baselines3: https://github.com/DLR-RM/stable-baselines3

## License
This project is only for research and educational purposes. 