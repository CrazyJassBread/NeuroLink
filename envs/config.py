from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from pyboy.utils import WindowEvent
import torch


@dataclass(frozen=True)
class MemoryAddresses:
    cur_health: int = 0xDB5A
    max_health: int = 0xDB5B
    rupee: int = 0xDB5E
    room_id: int = 0xDBAE
    keys: int = 0xDBD0


DEFAULT_ADDR = MemoryAddresses()

# Backward-compatible aliases for legacy imports.
ADDR_CUR_HEALTH = DEFAULT_ADDR.cur_health
ADDR_MAX_HEALTH = DEFAULT_ADDR.max_health
ADDR_RUPEE = DEFAULT_ADDR.rupee
ADDR_ROOM_ID = DEFAULT_ADDR.room_id
ADDR_KEYS = DEFAULT_ADDR.keys


DEFAULT_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
]

DEFAULT_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
]


@dataclass
class ObservationConfig:
    mode: str = "bucketed"
    output_shape: tuple[int, int] = (8, 10)
    grayscale: bool = True
    normalize: bool = False
    gaussian_size: int = 16
    gaussian_sigma: float = 4.0
    bucket_boundaries: Sequence[float] = field(default_factory=lambda: [30, 90, 190])
    bucket_mapping: Sequence[int] = field(default_factory=lambda: [0, 1, 2, 3])


@dataclass
class ZeldaEnvConfig:
    max_steps: int = 1000
    press_ticks: int = 10
    release_ticks: int = 10
    outside_max_steps: int = 100
    debug: bool = False
    addresses: MemoryAddresses = field(default_factory=MemoryAddresses)
    observation: ObservationConfig = field(default_factory=ObservationConfig)


def resolve_device(device: torch.device | str | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
