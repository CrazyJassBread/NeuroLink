from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from .config import ObservationConfig


class ObservationProcessor:
    # Builds observations from raw emulator screen frames.
    # You can change the bucket in the file ---> envs/config.py
    
    def __init__(self, config: ObservationConfig, device: torch.device):
        self.config = config
        self.device = device

        self.bucket_boundaries = torch.tensor(
            list(config.bucket_boundaries), dtype=torch.float32, device=device
        )
        self.bucket_mapping = torch.tensor(
            list(config.bucket_mapping), dtype=torch.uint8, device=device
        )
        self.gaussian_kernel = self._build_gaussian_kernel(
            config.gaussian_size, config.gaussian_sigma
        ).to(device)

    def observation_space(self) -> spaces.Box:
        if self.config.mode == "bucketed":
            return spaces.Box(
                low=0,
                high=255,
                shape=self.config.output_shape,
                dtype=np.uint8,
            )

        h, w = self.config.output_shape
        if self.config.grayscale:
            if self.config.normalize:
                return spaces.Box(low=0.0, high=1.0, shape=(h, w), dtype=np.float32)
            return spaces.Box(low=0, high=255, shape=(h, w), dtype=np.uint8)

        if self.config.normalize:
            return spaces.Box(low=0.0, high=1.0, shape=(h, w, 3), dtype=np.float32)
        return spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

    def process(self, screen: np.ndarray) -> np.ndarray:
        if self.config.mode == "bucketed":
            return self._process_bucketed(screen)
        if self.config.mode == "resized":
            return self._process_resized(screen)
        raise ValueError(f"Unsupported observation mode: {self.config.mode}")

    def _build_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.astype(np.float32)[None, None, :, :]
        return torch.from_numpy(kernel)

    def _process_bucketed(self, screen: np.ndarray) -> np.ndarray:
        raw_screen = screen[:128, :160, 0]
        screen_tensor = torch.from_numpy(raw_screen).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        pooled = F.conv2d(
            screen_tensor,
            self.gaussian_kernel,
            stride=self.config.gaussian_size,
            padding=0,
        ).squeeze(0).squeeze(0)
        pooled = pooled.clamp_(0, 255)

        indices = torch.bucketize(pooled, self.bucket_boundaries)
        mapped = self.bucket_mapping[indices]
        return mapped.cpu().numpy()

    def _process_resized(self, screen: np.ndarray) -> np.ndarray:
        h, w = self.config.output_shape

        if self.config.grayscale:
            channel = screen[:, :, 0]
            tensor = torch.from_numpy(channel).to(
                self.device, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
            out = resized.squeeze(0).squeeze(0)
        else:
            tensor = torch.from_numpy(screen[:, :, :3]).to(self.device, dtype=torch.float32)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            resized = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
            out = resized.squeeze(0).permute(1, 2, 0)

        if self.config.normalize:
            return (out / 255.0).clamp_(0.0, 1.0).cpu().numpy().astype(np.float32)

        return out.clamp_(0, 255).cpu().numpy().astype(np.uint8)
