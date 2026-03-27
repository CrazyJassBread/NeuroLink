# envs

this folder mainly contains the environment wrappers for the zelda game.

### base_env.py
This file contains the base environment class `ZeldaEnv` which defines the common functionalities for all the specific room environments. It handles the initialization of the game, resetting the environment, and stepping through the environment based on actions taken by the agent.

### emulator.py
This file contains the `Emulator` class which is responsible for interfacing with the game emulator. It provides methods to load the game, save and load states, and perform actions in the game.

### config.py
This file contains configuration settings for the environments, such as action mappings, reward structures, and other parameters that can be adjusted for different tasks.

### observation.py
This file contains functions and classes related to processing the observations from the game. 

- [ ] TODO: It may include methods for extracting relevant information from the game state, such as the player's position, inventory, and other game-specific features.

### reward.py

This file contains functions and classes related to calculating the rewards for the agent based on its actions and the game state.