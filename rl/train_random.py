from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from rl.utils import EpisodeResult, make_env, observation_is_valid, write_episode_results_jsonl
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import EpisodeResult, make_env, observation_is_valid, write_episode_results_jsonl


DEFAULT_OUTPUT_PATH = Path("rl") / "outputs" / "random_training.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random-policy smoke training loop for env_diy.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--render", action="store_true", help="Call env.render() every step.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Repeat each sampled agent action for this many environment ticks.",
    )
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
        help="JSONL output path for episode summaries.",
    )
    return parser.parse_args()


def run_random_training(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    config: Path | None = None,
    render: bool = False,
    action_repeat: int = 1,
    output: Path = DEFAULT_OUTPUT_PATH,
) -> list[EpisodeResult]:
    if episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if action_repeat < 1:
        raise ValueError("--action-repeat must be >= 1")

    env = make_env(
        config_path=config,
        render_mode="rgb_array" if render else None,
        seed=seed,
        action_repeat=action_repeat,
    )
    results: list[EpisodeResult] = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            if not observation_is_valid(env, obs):
                raise RuntimeError(f"reset() returned an observation outside observation_space at episode {episode}")

            total_reward = 0.0
            terminated = False
            truncated = False
            game_over = False
            length = 0

            for step_index in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if not observation_is_valid(env, obs):
                    raise RuntimeError(
                        f"step() returned an observation outside observation_space "
                        f"at episode {episode}, step {step_index}"
                    )
                total_reward += float(reward)
                length = step_index + 1
                game_over = game_over or info.get("terminal_reason") == "agent_dead"

                if render:
                    env.render()
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
                "episode={episode} reward={reward:.3f} length={length} "
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
        env.close()

    write_episode_results_jsonl(output, results)
    print(f"wrote {len(results)} episode summaries to {output}")
    return results


def main() -> None:
    args = parse_args()
    run_random_training(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        config=args.config,
        render=args.render,
        action_repeat=args.action_repeat,
        output=args.output,
    )


if __name__ == "__main__":
    main()
