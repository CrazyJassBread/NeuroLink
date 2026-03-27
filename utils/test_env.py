from stable_baselines3.common.env_checker import check_env
import os
import sys
import matplotlib.pyplot as plt

# Ensure local project modules resolve before similarly named site-packages.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import Room58_Task_Env

if __name__ == "__main__":
    # check the environment
    base_dir = os.path.dirname(os.path.dirname(__file__))
    game_path = os.path.join(base_dir, "game_state", "Link's awakening.gb")
    save_path = os.path.join(base_dir, "game_state", "Room58_task1.state")

    env = Room58_Task_Env(game_file=game_path, save_file=save_path, render_mode="human")

    check_env(env, warn=True)

    obs, info = env.reset()
    print("reset ok, obs shape:", obs.shape)


    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {step}: reward={reward}, done={terminated}, truncated={truncated}")

        if terminated or truncated:
            obs, info = env.reset()
            print("episode reset")

    env.close()