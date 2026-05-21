from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from rl.config.loader import load_training_config
from rl.envs.registry import get_env_builder
from rl.utils.logging import EpisodeResult, write_episode_results_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL model and export GIF rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path under rl/config or any file path.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config value with dotted key syntax, e.g. environment.max_episode_steps=200",
    )
    parser.add_argument("--model", type=Path, default=None, help="Path to model.zip or model base path.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes.")
    parser.add_argument("--output", type=Path, default=None, help="GIF output path or directory.")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second for the GIF.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed override.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., cpu, cuda).")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode.")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic policy actions.")
    parser.add_argument("--stochastic", action="store_true", help="Force stochastic policy actions.")
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save episode summaries to output_dir/eval_gif.jsonl.",
    )
    return parser.parse_args(argv)


def resolve_model_path(output_dir: Path, model_arg: Path | None) -> Path:
    if model_arg is None:
        return output_dir / "model"
    return model_arg


def resolve_gif_path(output_root: Path, output_arg: Path | None, episode: int, total: int) -> Path:
    if output_arg is None:
        if total == 1:
            return output_root / "eval.gif"
        return output_root / f"eval_episode_{episode:03d}.gif"

    if total == 1:
        return output_arg

    if output_arg.suffix:
        return output_arg.with_name(f"{output_arg.stem}_{episode:03d}{output_arg.suffix}")

    return output_arg / f"eval_episode_{episode:03d}.gif"


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if not frames:
        raise ValueError("no frames captured to write")

    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    duration_ms = int(1000 / fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def run_episode(
    env,
    model: PPO,
    *,
    seed: int,
    deterministic: bool,
    max_steps: int,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    obs, _info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    frame = env.render()
    if frame is None:
        raise RuntimeError("render() returned None; enable render_mode=rgb_array")
    frames.append(frame)

    total_reward = 0.0
    terminated = False
    truncated = False
    game_over = False
    length = 0

    for step_index in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        length = step_index + 1
        game_over = game_over or info.get("terminal_reason") == "agent_dead"
        frame = env.render()
        if frame is None:
            raise RuntimeError("render() returned None; enable render_mode=rgb_array")
        frames.append(frame)
        if terminated or truncated:
            break

    result = EpisodeResult(
        episode=0,
        total_reward=total_reward,
        length=length,
        terminated=terminated,
        truncated=truncated,
        game_over=game_over,
    )
    return result, frames


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.deterministic and args.stochastic:
        raise ValueError("Choose only one of --deterministic or --stochastic.")

    config = load_training_config(args.config, overrides=args.overrides)
    output_dir = config.experiment.output_dir
    model_path = resolve_model_path(output_dir, args.model)
    if model_path.suffix != ".zip" and not model_path.exists():
        candidate = model_path.with_suffix(".zip")
        if candidate.exists():
            model_path = candidate

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    episodes = args.episodes if args.episodes is not None else config.evaluation.episodes
    seed = args.seed if args.seed is not None else config.experiment.seed
    deterministic = (
        True
        if args.deterministic
        else False
        if args.stochastic
        else config.evaluation.deterministic
    )
    device = args.device or config.experiment.device
    max_steps = args.max_steps or config.environment.max_episode_steps or 10_000

    env_builder = get_env_builder(config.environment.id)
    env = env_builder(config.environment, seed=seed, render=True)
    try:
        model = PPO.load(str(model_path), device=device)

        results: list[EpisodeResult] = []
        for episode in range(episodes):
            episode_seed = seed + episode
            result, frames = run_episode(
                env,
                model,
                seed=episode_seed,
                deterministic=deterministic,
                max_steps=max_steps,
            )
            result = EpisodeResult(
                episode=episode,
                total_reward=result.total_reward,
                length=result.length,
                terminated=result.terminated,
                truncated=result.truncated,
                game_over=result.game_over,
            )
            results.append(result)

            gif_path = resolve_gif_path(output_dir, args.output, episode, episodes)
            save_gif(frames, gif_path, args.fps)
            print(
                "eval episode={episode} reward={reward:.3f} length={length} terminated={terminated} "
                "truncated={truncated} game_over={game_over} gif={gif}".format(
                    episode=episode,
                    reward=result.total_reward,
                    length=result.length,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    game_over=result.game_over,
                    gif=gif_path,
                )
            )

        if args.save_metrics:
            metrics_path = output_dir / "eval_gif.jsonl"
            write_episode_results_jsonl(metrics_path, results)
            print(f"Wrote {len(results)} eval episode summaries to {metrics_path}")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
