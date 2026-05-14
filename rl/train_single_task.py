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
TASK_REWARD_IDS = {
    "avoid_traps": "sparse_exit",
    "kill_monsters": "kill_monster",
    "key_door": "collect_key",
}


@dataclass(frozen=True)
class SingleTaskEpisodeResult:
    episode: int
    total_reward: float
    length: int
    terminated: bool
    truncated: bool
    terminated_reason: str | None
    reward_name: str


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
        "--reward-module",
        default=None,
        help="Reward module import path. Overrides the default built-in reward id for --task.",
    )
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
    reward_module: str | None = None,
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
    resolved_reward_id = None if reward_module is not None else TASK_REWARD_IDS.get(task)
    env = make_env(
        config_path=config_path,
        reward_id=resolved_reward_id,
        reward_module=reward_module,
        render_mode="rgb_array" if render else None,
        seed=seed,
        action_repeat=action_repeat,
        max_steps=max_steps,
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
            terminated_reason = None
            reward_name = "base"
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
                terminated_reason = info.get("terminal_reason")
                reward_name = str(info.get("reward", {}).get("reward_name", reward_name))

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
                terminated_reason=terminated_reason,
                reward_name=reward_name,
            )
            results.append(result)
            print(
                "episode={episode} reward={reward:.3f} length={length} reward_name={reward_name} "
                "terminated={terminated} truncated={truncated} reason={reason}".format(
                    episode=result.episode,
                    reward=result.total_reward,
                    length=result.length,
                    reward_name=result.reward_name,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    reason=result.terminated_reason,
                )
            )
    finally:
        env.close()

    _write_results_jsonl(output, results)
    mean_reward = sum(result.total_reward for result in results) / len(results)
    print(f"mean_reward={mean_reward:.3f}")
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
        reward_module=args.reward_module,
        render=args.render,
        action_repeat=args.action_repeat,
        output=args.output,
    )


if __name__ == "__main__":
    main()
