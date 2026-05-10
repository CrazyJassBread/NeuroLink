"""Compatibility entry for named dungeon PPO training.

Prefer the unified entry:

    python -m rl.train --method ppo --task-rooms combat_training
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.baselines.ppo import train as train_ppo
from rl.config import TASK_ROOM_CONFIGS, TrainingConfig


DUNGEONS = {
    name: path
    for name, path in TASK_ROOM_CONFIGS.items()
    if name in {"prototype", "combat_training", "evasion_training", "chest_training"}
}
CURRICULUM_ORDER = ("combat_training", "evasion_training", "chest_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for PPO training on named dungeons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dungeon", choices=sorted(DUNGEONS))
    group.add_argument("--curriculum", action="store_true", help=f"Train sequentially on: {' -> '.join(CURRICULUM_ORDER)}")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index. Use -1 for CPU, omit for auto.")
    parser.add_argument("--skip-train", action="store_true", help="Load existing models and run evaluation only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_rooms = CURRICULUM_ORDER if args.curriculum else (args.dungeon,)
    config = TrainingConfig(
        method="ppo",
        episodes=args.n_eval_episodes,
        max_steps=args.max_steps,
        total_timesteps=args.total_timesteps,
        gpu=args.gpu,
        seed=args.seed,
        action_repeat=args.action_repeat,
        task_rooms=tuple(task_rooms),
        render=args.render,
        skip_train=args.skip_train,
    )
    train_ppo(config)


if __name__ == "__main__":
    main()
