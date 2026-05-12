from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__:
    from rl.utils import make_env, observation_is_valid
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import make_env, observation_is_valid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_TASK_ROOT = PROJECT_ROOT / "env_diy" / "map_data" / "dungeons"
DEFAULT_OUTPUT_PATH = Path("rl") / "outputs" / "single_task_training.jsonl"
SUPPORTED_TASKS = {"avoid_traps", "kill_monsters", "key_door"}


@dataclass(frozen=True)
class SingleTaskEpisodeResult:
    episode: int
    total_reward: float
    length: int
    terminated: bool
    truncated: bool
    finish: bool
    task_id: str
    task_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random-policy smoke loop for an env_diy single-task room.")
    parser.add_argument("--task", choices=sorted(SUPPORTED_TASKS), default="avoid_traps", help="Single-task type.")
    parser.add_argument("--room", default="room_001", help="Room id/file stem inside the task directory.")
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
    parser.add_argument("--config", type=Path, default=None, help="Explicit single-task room JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL output path for episode summaries.",
    )
    return parser.parse_args()


def resolve_single_task_config(*, task: str, room: str, config: Path | None) -> Path:
    if config is not None:
        path = Path(config)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if task not in SUPPORTED_TASKS:
        allowed = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"unsupported task '{task}', allowed: {allowed}")
    room_file = room if room.endswith(".json") else f"{room}.json"
    return SINGLE_TASK_ROOT / task / room_file


def run_single_task_training(
    *,
    task: str,
    room: str,
    episodes: int,
    max_steps: int,
    seed: int,
    config: Path | None = None,
    render: bool = False,
    action_repeat: int = 1,
    output: Path = DEFAULT_OUTPUT_PATH,
) -> list[SingleTaskEpisodeResult]:
    if episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if action_repeat < 1:
        raise ValueError("--action-repeat must be >= 1")

    config_path = resolve_single_task_config(task=task, room=room, config=config)
    env = make_env(
        config_path=config_path,
        render_mode="rgb_array" if render else None,
        seed=seed,
        action_repeat=action_repeat,
    )
    results: list[SingleTaskEpisodeResult] = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            if not observation_is_valid(env, obs):
                raise RuntimeError(f"reset() returned an observation outside observation_space at episode {episode}")

            total_reward = 0.0
            terminated = False
            truncated = False
            finish = False
            task_id = env.task_config.task_id if env.task_config is not None else ""
            task_type = env.task_config.task_type if env.task_config is not None else task
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
                task_info = info.get("task", {})
                finish = finish or bool(task_info.get("success", False))

                if render:
                    env.render()
                if terminated or truncated:
                    break

            result = SingleTaskEpisodeResult(
                episode=episode,
                total_reward=total_reward,
                length=length,
                terminated=terminated,
                truncated=truncated,
                finish=finish,
                task_id=task_id,
                task_type=task_type,
            )
            results.append(result)
            print(
                "episode={episode} reward={reward:.3f} length={length} finish={finish} "
                "terminated={terminated} truncated={truncated}".format(
                    episode=result.episode,
                    reward=result.total_reward,
                    length=result.length,
                    finish=result.finish,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            )
    finally:
        env.close()

    _write_results_jsonl(output, results)
    finish_rate = sum(1 for result in results if result.finish) / len(results)
    mean_reward = sum(result.total_reward for result in results) / len(results)
    print(f"finish_rate={finish_rate:.2f} mean_reward={mean_reward:.3f}")
    print(f"wrote {len(results)} episode summaries to {output}")
    return results


def _write_results_jsonl(path: str | Path, results: list[SingleTaskEpisodeResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    run_single_task_training(
        task=args.task,
        room=args.room,
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
