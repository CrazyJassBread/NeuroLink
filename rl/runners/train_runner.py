from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl.algorithms.registry import get_algorithm_trainer
from rl.config.loader import load_training_config
from rl.config.schema import TrainingConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training entry for RL experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path under rl/config or any file path.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config value with dotted key syntax, e.g. algorithm.total_timesteps=100000",
    )
    return parser.parse_args(argv)


def run_training(config: TrainingConfig) -> object:
    trainer = get_algorithm_trainer(config.algorithm.name)
    return trainer(config)


def main(argv: Sequence[str] | None = None) -> object:
    args = parse_args(argv)
    config = load_training_config(args.config, overrides=args.overrides)
    return run_training(config)
