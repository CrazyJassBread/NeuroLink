from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DungeonFeaturesExtractor(BaseFeaturesExtractor):
    """Flatten-and-concat extractor for Dict observations used by nesylink PPO."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self._obs_keys = sorted(observation_space.spaces.keys())
        self._flatten = nn.ModuleList(nn.Flatten() for _ in self._obs_keys)
        total_flat = sum(int(np.prod(observation_space.spaces[key].shape)) for key in self._obs_keys)
        self._net = nn.Sequential(nn.Linear(total_flat, features_dim), nn.ReLU())

    def forward(self, observations: dict) -> torch.Tensor:
        parts = [flat(observations[key].float()) for key, flat in zip(self._obs_keys, self._flatten)]
        return self._net(torch.cat(parts, dim=1))
