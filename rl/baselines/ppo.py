from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.config import TrainingConfig, TrainingTarget, resolve_task_rooms
from rl.utils import EpisodeResult, make_env, make_task_env, write_episode_results_jsonl


class EpisodeKeyAdapter(gym.Wrapper):
    """Rename info['episode'] so SB3's Monitor can reserve that key."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "episode" in info:
            info["dungeon_episode"] = info.pop("episode")
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if "episode" in info:
            info["dungeon_episode"] = info.pop("episode")
        return obs, info


class DungeonFeaturesExtractor(BaseFeaturesExtractor):
    """Flatten-and-concat extractor for the dungeon Dict obs space."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self._obs_keys = sorted(observation_space.spaces.keys())
        self._flatten = nn.ModuleList(nn.Flatten() for _ in self._obs_keys)
        total_flat = sum(int(np.prod(observation_space.spaces[k].shape)) for k in self._obs_keys)
        self._net = nn.Sequential(nn.Linear(total_flat, features_dim), nn.ReLU())

    def forward(self, observations: dict) -> torch.Tensor:
        parts = [flat(observations[k].float()) for k, flat in zip(self._obs_keys, self._flatten)]
        return self._net(torch.cat(parts, dim=1))


DEFAULT_OUTPUT_PATH = Path("rl") / "outputs" / "ppo_eval.jsonl"
DEFAULT_SAVE_PATH = Path("rl") / "outputs" / "ppo_model"


def train(config: TrainingConfig) -> list[EpisodeResult]:
    _validate_training_config(config)
    targets = resolve_task_rooms(config.task_rooms, config_path=config.config_path)

    all_results: list[EpisodeResult] = []
    for target_index, target in enumerate(targets):
        target_seed = config.seed + target_index
        target_output_dir = Path(config.output_dir) / config.method / target.name
        results = run_ppo_training(
            total_timesteps=config.total_timesteps,
            n_eval_episodes=config.episodes,
            max_steps=config.max_steps,
            seed=target_seed,
            config=target.config_path,
            task_id=target.task_id,
            render=config.render,
            action_repeat=config.action_repeat,
            output=target_output_dir / "eval.jsonl",
            save_path=target_output_dir / "model",
            device=config.device,
            skip_train=config.skip_train,
        )
        all_results.extend(results)
    return all_results


def run_ppo_training(
    *,
    total_timesteps: int,
    n_eval_episodes: int,
    max_steps: int,
    seed: int,
    config: Path | None = None,
    task_id: str | None = None,
    render: bool = False,
    action_repeat: int = 1,
    output: Path = DEFAULT_OUTPUT_PATH,
    save_path: Path = DEFAULT_SAVE_PATH,
    device: str = "auto",
    skip_train: bool = False,
) -> list[EpisodeResult]:
    if total_timesteps < 1 and not skip_train:
        raise ValueError("--total-timesteps must be >= 1 unless --skip-train is set")
    if n_eval_episodes < 1:
        raise ValueError("--episodes/--n-eval-episodes must be >= 1")
    if max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if action_repeat < 1:
        raise ValueError("--action-repeat must be >= 1")

    save_path = Path(save_path)
    if skip_train:
        model_path = save_path.with_suffix(".zip")
        if not model_path.exists():
            raise FileNotFoundError(f"--skip-train requested but model was not found: {model_path}")
        model = PPO.load(str(save_path), device=device)
    else:
        raw_train_env = _make_training_env(
            config=config,
            task_id=task_id,
            render_mode=None,
            seed=seed,
            action_repeat=action_repeat,
        )
        train_env = Monitor(EpisodeKeyAdapter(raw_train_env))
        policy_kwargs = {"features_extractor_class": DungeonFeaturesExtractor}
        model = PPO(
            "MultiInputPolicy",
            train_env,
            verbose=1,
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=device,
        )
        print(f"Training PPO for {total_timesteps} timesteps on {config or 'default dungeon'}...")
        try:
            model.learn(total_timesteps=total_timesteps)
        finally:
            train_env.close()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"Model saved to {save_path}.zip")

    results = _evaluate_ppo(
        model=model,
        config=config,
        task_id=task_id,
        render=render,
        action_repeat=action_repeat,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        max_steps=max_steps,
    )
    write_episode_results_jsonl(output, results)
    print(f"Wrote {len(results)} eval episode summaries to {output}")
    return results


def _evaluate_ppo(
    *,
    model: PPO,
    config: Path | None,
    task_id: str | None,
    render: bool,
    action_repeat: int,
    seed: int,
    n_eval_episodes: int,
    max_steps: int,
) -> list[EpisodeResult]:
    eval_env = _make_training_env(
        config=config,
        task_id=task_id,
        render_mode="rgb_array" if render else None,
        seed=seed + 1,
        action_repeat=action_repeat,
    )
    results: list[EpisodeResult] = []
    try:
        for episode in range(n_eval_episodes):
            obs, info = eval_env.reset(seed=seed + 1 + episode)
            total_reward = 0.0
            terminated = False
            truncated = False
            game_over = False
            length = 0

            for step_index in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                total_reward += float(reward)
                length = step_index + 1
                game_over = game_over or info.get("terminal_reason") == "agent_dead"

                if render:
                    eval_env.render()
                if terminated or truncated:
                    break

            result = EpisodeResult(
                episode=episode,
                total_reward=total_reward,
                length=length,
                terminated=terminated,
                truncated=truncated,
                game_over=game_over,
            )
            results.append(result)
            print(
                "eval episode={episode} reward={reward:.3f} length={length} "
                "terminated={terminated} truncated={truncated} game_over={game_over}".format(
                    episode=result.episode,
                    reward=result.total_reward,
                    length=result.length,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    game_over=result.game_over,
                )
            )
    finally:
        eval_env.close()
    return results


def _validate_training_config(config: TrainingConfig) -> None:
    if config.method != "ppo":
        raise ValueError(f"PPO runner received unsupported method '{config.method}'")
    if config.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if config.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if config.action_repeat < 1:
        raise ValueError("--action-repeat must be >= 1")


def _make_training_env(
    *,
    config: Path | None,
    task_id: str | None,
    render_mode: str | None,
    seed: int,
    action_repeat: int,
):
    if task_id is not None:
        return make_task_env(
            task_id,
            render_mode=render_mode,
            seed=seed,
            action_repeat=action_repeat,
        )
    return make_env(
        config_path=config,
        render_mode=render_mode,
        seed=seed,
        action_repeat=action_repeat,
    )
