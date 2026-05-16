from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.baselines import get_baseline_runner
from rl.config import (
    DEFAULT_TRAINING_CONFIG,
    SUPPORTED_METHODS,
    TASK_ROOM_CONFIGS,
    TrainingConfig,
    apply_overrides,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training entry for classic RL algorithms on nesylink.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default=DEFAULT_TRAINING_CONFIG.method)
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.episodes,
        help="Number of evaluation episodes after training.",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_TRAINING_CONFIG.max_steps)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TRAINING_CONFIG.total_timesteps)
    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.gpu,
        help="GPU index for algorithms that support torch devices. Use -1 for CPU, omit for auto.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG.seed)
    parser.add_argument("--action-repeat", type=int, default=DEFAULT_TRAINING_CONFIG.action_repeat)
    parser.add_argument(
        "--task-rooms",
        nargs="+",
        default=list(DEFAULT_TRAINING_CONFIG.task_rooms),
        choices=sorted(TASK_ROOM_CONFIGS),
        help="One or more named task rooms/dungeons. Multiple values run as a sequential curriculum.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG.config_path,
        help="Explicit nesylink dungeon/room JSON path. Overrides --task-rooms.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.output_dir)
    parser.add_argument("--render", action="store_true", default=DEFAULT_TRAINING_CONFIG.render)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=DEFAULT_TRAINING_CONFIG.skip_train,
        help="Load an existing model from the resolved output path and run evaluation only.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    return apply_overrides(
        method=args.method,
        episodes=args.episodes,
        max_steps=args.max_steps,
        total_timesteps=args.total_timesteps,
        gpu=args.gpu,
        seed=args.seed,
        action_repeat=args.action_repeat,
        task_rooms=tuple(args.task_rooms),
        config_path=args.config_path,
        output_dir=args.output_dir,
        render=args.render,
        skip_train=args.skip_train,
    )


def main() -> None:
    config = build_config(parse_args())
    runner = get_baseline_runner(config.method)
    runner(config)


if __name__ == "__main__":
    main()
