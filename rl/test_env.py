from pathlib import Path
import sys

# Use this file to check the obs and info dict structure
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_diy.env import make_env

# env = make_env("env_diy/map_data/dungeons/prototype/dungeon.json", api="gym")
env = make_env("env_diy/map_data/dungeons/avoid_traps/room_001.json", api = "gym")
obs, info = env.reset(seed=0)

for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if i % 50 == 0:
        print(f"Observation: {obs}")
        print("-" * 40)
        print(f"Observation space: {env.observation_space}")
        print("-" * 40)
        print(f"Info: {info}")
        print("-" * 40)
    if terminated or truncated:
        obs, info = env.reset()

env.close();