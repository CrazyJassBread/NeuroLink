from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def summarize_task_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    reward_terms_totals: dict[str, float] = {}
    terminated_reasons = Counter()
    for episode in episodes:
        for key, value in episode["reward_terms"].items():
            reward_terms_totals[key] = reward_terms_totals.get(key, 0.0) + float(value)
        reason = episode["terminated_reason"]
        if reason is not None:
            terminated_reasons[str(reason)] += 1

    episode_count = len(episodes)
    return {
        "episodes": episode_count,
        "success_rate": _rate(episodes, "success"),
        "failure_rate": _rate(episodes, "failure"),
        "truncation_rate": _rate(episodes, "truncated"),
        "mean_return": mean(episode["return"] for episode in episodes) if episodes else 0.0,
        "mean_episode_length": mean(episode["length"] for episode in episodes) if episodes else 0.0,
        "mean_task_progress": mean(episode["task_progress"] for episode in episodes) if episodes else 0.0,
        "mean_reward_terms": {
            key: value / max(1, episode_count)
            for key, value in reward_terms_totals.items()
        },
        "terminated_reason_counts": dict(terminated_reasons),
    }


def summarize_suite_metrics(task_results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_success_rate": mean(result["success_rate"] for result in task_results) if task_results else 0.0,
        "mean_return": mean(result["mean_return"] for result in task_results) if task_results else 0.0,
        "mean_episode_length": mean(result["mean_episode_length"] for result in task_results) if task_results else 0.0,
    }


def _rate(episodes: list[dict[str, Any]], key: str) -> float:
    if not episodes:
        return 0.0
    return sum(1 for episode in episodes if episode[key]) / len(episodes)
