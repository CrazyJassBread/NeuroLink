from __future__ import annotations

from collections.abc import Callable

from rl.config.schema import TrainingConfig


AlgorithmTrainer = Callable[[TrainingConfig], object]
