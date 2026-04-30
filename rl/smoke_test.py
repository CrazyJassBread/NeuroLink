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