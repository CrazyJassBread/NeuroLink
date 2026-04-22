from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from orgin_env import ObservationConfig, Room58_Task_Env as Zelda_Env

TOTAL_STEPS = 30000
USE_TIME_LIMIT = False

save_state = "game_state/Room58_task1.state"
game_file = "game_state/Link's awakening.gb"

obs_cfg = ObservationConfig(
    mode="bucketed",
    output_shape=(8, 10),
    grayscale=True,
    normalize=False,
)

env = Zelda_Env(game_file=game_file, save_file=save_state, observation_config=obs_cfg)
if USE_TIME_LIMIT:
    env = TimeLimit(env, max_episode_steps=1000)
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./log/Room58/ppo_tensorboard/",
)

model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
model.save("./rl_model/test/ppo58_task")
env.close()