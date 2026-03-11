import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import torch

from envs import Room58_Task_Env as Zelda_Env
from .PPO.model import CustomResNet, CustomACPolicy, CustomPPO

TOTAL_STEPS = 30000

save_state = "game_state/Room58_task1.state"
game_file = "game_state/Link's awakening.gb"

env_kwargs = {
    "game_file": game_file,
    "save_file": save_state, 
}

policy_kwargs = {
    "features_extractor_class": CustomResNet,
    "features_extractor_kwargs": {"features_dim": 1024},
    "activation_fn": nn.ReLU,
    "net_arch": [],
    "optimizer_class": torch.optim.Adam,
    "optimizer_kwargs": {"eps": 1e-5}
}

env = Zelda_Env(game_file=game_file, save_file=save_state)
env.disable_render = True
env = Monitor(env)


model = CustomPPO(
    CustomACPolicy,
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=3,
    gamma=0.95,
    gae_lambda=0.65,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    normalize_advantage=False,
    tensorboard_log="./log/Room58/ppo_tensorboard/"
)


model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
model.save("./rl_model/test/ppo58_task")
env.close()