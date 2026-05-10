from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.baselines.ppo import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SAVE_PATH,
    DungeonFeaturesExtractor,
    EpisodeKeyAdapter,
    run_ppo_training,
)


_EpisodeKeyAdapter = EpisodeKeyAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for PPO training. Prefer `python rl/train.py --method ppo`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index. Use -1 for CPU, omit for auto.")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def _device_from_gpu(gpu: int | None) -> str:
    if gpu is None:
        return "auto"
    if gpu < 0:
        return "cpu"
    return f"cuda:{gpu}"


def main() -> None:
    args = parse_args()
    run_ppo_training(
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        config=args.config,
        render=args.render,
        action_repeat=args.action_repeat,
        output=args.output,
        save_path=args.save_path,
        device=_device_from_gpu(args.gpu),
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
