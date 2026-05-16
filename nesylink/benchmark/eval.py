from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .metrics import summarize_suite_metrics, summarize_task_metrics
from .registry import get_task_spec, list_tasks, make_benchmark_env


def evaluate_suite(
    suite_id: str,
    *,
    policy: str,
    episodes: int,
    seed: int,
    json_output: Path | None = None,
) -> dict[str, Any]:
    if policy != "random":
        raise ValueError("only random policy is supported in benchmark v0")

    task_payloads: list[dict[str, Any]] = []
    for offset, task in enumerate(list_tasks(suite_id)):
        task_seed = seed + offset
        env = make_benchmark_env(
            suite_id,
            task.task_id,
            seed=task_seed,
        )
        try:
            episodes_payload = [_run_random_episode(env, task, task_seed + index) for index in range(episodes)]
        finally:
            env.close()
        metrics = summarize_task_metrics(episodes_payload)
        task_payloads.append(
            {
                "task_id": task.task_id,
                "spec": _task_to_payload(get_task_spec(suite_id, task.task_id)),
                "metrics": metrics,
            }
        )

    aggregate = summarize_suite_metrics([payload["metrics"] for payload in task_payloads])
    result = {
        "suite_id": suite_id,
        "policy": policy,
        "episodes": episodes,
        "seed": seed,
        "tasks": task_payloads,
        "aggregate": aggregate,
    }
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _run_random_episode(env, task, seed: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_reward = 0.0
    length = 0
    last_info = info
    terminated = False
    truncated = False
    reward_terms_total: dict[str, float] = {}
    for step_index in range(task.max_episode_steps):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        length = step_index + 1
        reward_terms = dict(last_info.get("reward", {}).get("reward_signals", {}))
        for key, value in reward_terms.items():
            reward_terms_total[key] = reward_terms_total.get(key, 0.0) + float(value)
        if terminated or truncated:
            break
    reward_info = dict(last_info.get("reward", {}))
    terminated_reason = last_info.get("terminal_reason")
    completed = bool(reward_info.get("terminated", False)) and terminated_reason != "agent_dead"
    return {
        "return": total_reward,
        "length": length,
        "completed": completed,
        "dead": bool(last_info.get("game", {}).get("dead", False)),
        "truncated": bool(truncated),
        "terminated_reason": terminated_reason,
        "reward_terms": reward_terms_total,
    }


def _task_to_payload(task) -> dict[str, Any]:
    return {
        "suite_id": task.suite_id,
        "task_id": task.task_id,
        "map_id": task.map_id,
        "map_path": str(task.map_path),
        "difficulty": task.difficulty,
        "max_episode_steps": task.max_episode_steps,
        "reward_id": task.reward_id,
        "reward_module": task.reward_module,
        "observation_mode": task.observation_mode,
        "action_mode": task.action_mode,
        "objective": task.objective,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark suite evaluation for nesylink.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--policy", default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_suite(
        args.suite,
        policy=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        json_output=args.json_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
