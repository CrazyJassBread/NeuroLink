from __future__ import annotations

from rl.config import TrainingConfig


def train(config: TrainingConfig) -> object:
    raise NotImplementedError(
        "DQN baseline is registered in the unified training entry, "
        "but its implementation has not been added yet."
    )
