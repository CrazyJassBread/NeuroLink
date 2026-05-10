from __future__ import annotations

from collections.abc import Callable

from rl.config import TrainingConfig


BaselineRunner = Callable[[TrainingConfig], object]


def get_baseline_runner(method: str) -> BaselineRunner:
    normalized = method.lower()
    if normalized == "ppo":
        from .ppo import train

        return train
    if normalized == "dqn":
        from .dqn import train

        return train
    if normalized == "a3c":
        from .a3c import train

        return train
    raise ValueError(f"unsupported RL method '{method}'")


__all__ = ["BaselineRunner", "get_baseline_runner"]
