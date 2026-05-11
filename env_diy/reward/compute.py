from __future__ import annotations

from ..core.types import RewardBreakdown, RuntimeSnapshot, StuckPenaltyConfig
from ..maps.tasks import TaskConfig
from ..rewards.reward_fn import RewardConfig, compute_reward as _canonical_compute_reward


def compute_reward(
    prev_state: RuntimeSnapshot,
    next_state: RuntimeSnapshot,
    engine_result,
    task_config: TaskConfig | None,
    stuck_config: StuckPenaltyConfig,
) -> tuple[float, RewardBreakdown]:
    return _canonical_compute_reward(
        prev_state,
        next_state,
        engine_result,
        task_spec=task_config,
        config=RewardConfig(reward_mode="legacy", stuck_penalty=stuck_config),
    )
