from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rl.config.schema import TrainingConfig
from rl.envs.registry import get_env_builder
from rl.utils.logging import EpisodeResult, write_episode_results_jsonl

from .policy import DungeonFeaturesExtractor


class EpisodeKeyAdapter(gym.Wrapper):
    """Rename info['episode'] so SB3's Monitor can use that key safely."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "episode" in info:
            info["training_episode"] = info.pop("episode")
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if "episode" in info:
            info["training_episode"] = info.pop("episode")
        return obs, info


def train(config: TrainingConfig) -> list[EpisodeResult]:
    trainer = PPOTrainer(config)
    return trainer.run()


class PPOTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._env_builder = get_env_builder(config.environment.id)

    def run(self) -> list[EpisodeResult]:
        save_path = self.config.experiment.output_dir / "model"
        if self.config.experiment.resume:
            model_path = save_path.with_suffix(".zip")
            if not model_path.exists():
                raise FileNotFoundError(f"resume requested but model was not found: {model_path}")
            model = PPO.load(str(save_path), device=self.config.experiment.device)
        else:
            train_env = Monitor(EpisodeKeyAdapter(self._make_env(seed=self.config.experiment.seed, render=False)))
            try:
                policy, policy_kwargs = _policy_for_env(train_env)
                algo_kwargs = dict(self.config.algorithm.params)
                model = PPO(
                    policy,
                    train_env,
                    verbose=1,
                    seed=self.config.experiment.seed,
                    device=self.config.experiment.device,
                    policy_kwargs=policy_kwargs,
                    **algo_kwargs,
                )
                print(
                    f"Training PPO for {self.config.algorithm.total_timesteps} timesteps on "
                    f"{self.config.environment.id}..."
                )
                model.learn(total_timesteps=self.config.algorithm.total_timesteps)
            finally:
                train_env.close()

            if self.config.evaluation.save_model:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(save_path))
                print(f"Model saved to {save_path}.zip")

        results = self._evaluate(model)
        if self.config.evaluation.save_metrics:
            metrics_path = self.config.experiment.output_dir / "eval.jsonl"
            write_episode_results_jsonl(metrics_path, results)
            print(f"Wrote {len(results)} eval episode summaries to {metrics_path}")
        return results

    def _make_env(self, *, seed: int, render: bool) -> gym.Env:
        return self._env_builder(self.config.environment, seed=seed, render=render)

    def _evaluate(self, model: PPO) -> list[EpisodeResult]:
        if not self.config.evaluation.enabled:
            return []
        episodes = self.config.evaluation.episodes
        if episodes < 1:
            raise ValueError("evaluation.episodes must be >= 1 when evaluation is enabled")
        eval_env = self._make_env(seed=self.config.experiment.seed + 1, render=self.config.evaluation.render)
        results: list[EpisodeResult] = []
        try:
            for episode in range(episodes):
                obs, _info = eval_env.reset(seed=self.config.experiment.seed + 1 + episode)
                total_reward = 0.0
                terminated = False
                truncated = False
                game_over = False
                length = 0

                max_steps = self.config.environment.max_episode_steps or 10_000
                for step_index in range(max_steps):
                    action, _ = model.predict(obs, deterministic=self.config.evaluation.deterministic)
                    obs, reward, terminated, truncated, info = eval_env.step(int(action))
                    total_reward += float(reward)
                    length = step_index + 1
                    game_over = game_over or info.get("terminal_reason") == "agent_dead"

                    if self.config.evaluation.render:
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


def _policy_for_env(env: gym.Env) -> tuple[str, dict]:
    if isinstance(env.observation_space, spaces.Dict):
        return "MultiInputPolicy", {"features_extractor_class": DungeonFeaturesExtractor}
    if isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 3:
        return "CnnPolicy", {}
    return "MlpPolicy", {}
