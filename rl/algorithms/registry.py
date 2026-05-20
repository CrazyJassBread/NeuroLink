from __future__ import annotations

from .a3c.trainer import train as train_a3c
from .base import AlgorithmTrainer
from .dqn.trainer import train as train_dqn
from .ppo.trainer import train as train_ppo


_ALGORITHM_TRAINERS: dict[str, AlgorithmTrainer] = {
    "ppo": train_ppo,
    "dqn": train_dqn,
    "a3c": train_a3c,
}


def get_algorithm_trainer(name: str) -> AlgorithmTrainer:
    normalized = name.lower()
    try:
        return _ALGORITHM_TRAINERS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALGORITHM_TRAINERS))
        raise ValueError(f"unsupported algorithm '{name}', supported: {supported}") from exc


def register_algorithm_trainer(name: str, trainer: AlgorithmTrainer) -> None:
    _ALGORITHM_TRAINERS[name.lower()] = trainer
