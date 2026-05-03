from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

if __package__:
    from rl.utils import EpisodeResult, make_env, write_episode_results_jsonl
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import EpisodeResult, make_env, write_episode_results_jsonl

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _EpisodeKeyAdapter(gym.Wrapper):
    """Rename info['episode'] (an int in DungeonEnv) to info['dungeon_episode'].

    SB3's Monitor and on-policy algorithms reserve info['episode'] for their
    own episode-stats dict.  Without this rename, SB3 reads the env's integer
    and crashes when it calls len() on it.
    """

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
    """Flatten-and-concat extractor for the dungeon Dict obs space.

    Avoids nn.ModuleDict (used by SB3's default CombinedExtractor), which
    rejects obs keys that shadow built-in dict methods such as 'keys'.
    Uses nn.ModuleList with a stable sorted-key ordering instead.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self._obs_keys = sorted(observation_space.spaces.keys())
        self._flatten = nn.ModuleList(nn.Flatten() for _ in self._obs_keys)
        total_flat = sum(
            int(np.prod(observation_space.spaces[k].shape)) for k in self._obs_keys
        )
        self._net = nn.Sequential(nn.Linear(total_flat, features_dim), nn.ReLU())

    def forward(self, observations: dict) -> torch.Tensor:
        parts = [
            flat(observations[k].float())
            for k, flat in zip(self._obs_keys, self._flatten)
        ]
        return self._net(torch.cat(parts, dim=1))


DEFAULT_OUTPUT_PATH = Path("rl") / "outputs" / "ppo_eval.jsonl"
DEFAULT_SAVE_PATH = Path("rl") / "outputs" / "ppo_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on env_diy using Stable Baselines3.")
    parser.add_argument(
        "--total-timesteps", type=int, default=50_000,
        help="Total environment timesteps for training.",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=5,
        help="Number of greedy-evaluation episodes run after training.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per evaluation episode.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--render", action="store_true", help="Call env.render() during evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Dungeon config path. Defaults to env_diy/map_data/dungeons/prototype/dungeon.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL output path for evaluation episode summaries.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Model save path (no .zip extension; SB3 adds it automatically).",
    )
    return parser.parse_args()


def run_ppo_training(
    *,
    total_timesteps: int,
    n_eval_episodes: int,
    max_steps: int,
    seed: int,
    config: Path | None = None,
    render: bool = False,
    output: Path = DEFAULT_OUTPUT_PATH,
    save_path: Path = DEFAULT_SAVE_PATH,
) -> list[EpisodeResult]:
    if total_timesteps < 1:
        raise ValueError("--total-timesteps must be >= 1")
    if n_eval_episodes < 1:
        raise ValueError("--n-eval-episodes must be >= 1")
    if max_steps < 1:
        raise ValueError("--max-steps must be >= 1")

    # Training env wrapped in Monitor so SB3 can collect episode stats.
    raw_train_env = make_env(config_path=config, render_mode=None, seed=seed)
    train_env = Monitor(_EpisodeKeyAdapter(raw_train_env))

    policy_kwargs = {"features_extractor_class": DungeonFeaturesExtractor}
    model = PPO("MultiInputPolicy", train_env, verbose=1, seed=seed, policy_kwargs=policy_kwargs)
    print(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    train_env.close()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}.zip")

    # Greedy evaluation on a separate env instance.
    eval_env = make_env(config_path=config, render_mode="rgb_array" if render else None, seed=seed + 1)
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
                game_over = game_over or bool(info.get("game_over", False))

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

    write_episode_results_jsonl(output, results)
    print(f"Wrote {len(results)} eval episode summaries to {output}")
    return results


def main() -> None:
    args = parse_args()
    run_ppo_training(
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        config=args.config,
        render=args.render,
        output=args.output,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
