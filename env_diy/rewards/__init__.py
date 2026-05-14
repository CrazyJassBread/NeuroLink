from __future__ import annotations

from .base import BaseReward
from .collect_gold import CollectGoldReward
from .collect_key import CollectKeyReward
from .exploration import ExplorationReward
from .kill_monster import KillMonsterReward
from .loader import load_reward, load_reward_module, resolve_reward_module
from .sparse_exit import SparseExitReward

__all__ = [
    "BaseReward",
    "CollectGoldReward",
    "CollectKeyReward",
    "ExplorationReward",
    "KillMonsterReward",
    "SparseExitReward",
    "load_reward",
    "load_reward_module",
    "resolve_reward_module",
]
