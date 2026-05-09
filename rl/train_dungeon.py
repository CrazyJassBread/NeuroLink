"""Training entry point for single-task and curriculum dungeon training.

Usage examples
--------------
# Train on a single dungeon:
python -m rl.train_dungeon --dungeon combat_training --total-timesteps 100000

# Train all three single-task dungeons in sequence (curriculum):
python -m rl.train_dungeon --curriculum --total-timesteps 100000

# Evaluate a saved model:
python -m rl.train_dungeon --dungeon evasion_training --skip-train --n-eval-episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from rl.train_ppo import run_ppo_training
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_ppo import run_ppo_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DUNGEONS: dict[str, Path] = {
    "prototype":        PROJECT_ROOT / "env_diy/map_data/dungeons/prototype/dungeon.json",
    "combat_training":  PROJECT_ROOT / "env_diy/map_data/dungeons/combat_training/dungeon.json",
    "evasion_training": PROJECT_ROOT / "env_diy/map_data/dungeons/evasion_training/dungeon.json",
    "chest_training":   PROJECT_ROOT / "env_diy/map_data/dungeons/chest_training/dungeon.json",
}

CURRICULUM_ORDER = ["combat_training", "evasion_training", "chest_training"]


def output_dir(dungeon_name: str) -> Path:
    return PROJECT_ROOT / "rl" / "outputs" / dungeon_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a named dungeon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dungeon", choices=list(DUNGEONS.keys()))
    group.add_argument("--curriculum", action="store_true",
                       help=f"Train sequentially on: {' -> '.join(CURRICULUM_ORDER)}")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, run evaluation only.")
    return parser.parse_args()


def train_one(dungeon_name: str, args: argparse.Namespace) -> None:
    config = DUNGEONS[dungeon_name]
    out_dir = output_dir(dungeon_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Dungeon : {dungeon_name}")
    print(f"  Config  : {config}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}\n")
    run_ppo_training(
        total_timesteps=0 if args.skip_train else args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        config=config,
        render=args.render,
        output=out_dir / "eval.jsonl",
        save_path=out_dir / "model",
    )


def main() -> None:
    args = parse_args()
    if args.curriculum:
        print(f"Curriculum mode: {' -> '.join(CURRICULUM_ORDER)}")
        for name in CURRICULUM_ORDER:
            train_one(name, args)
        print("\nCurriculum complete.")
    else:
        train_one(args.dungeon, args)


if __name__ == "__main__":
    main()
