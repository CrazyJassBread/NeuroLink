from __future__ import annotations

from pathlib import Path

import numpy as np
from pyboy import PyBoy


class EmulatorController:
    """Thin wrapper around PyBoy lifecycle and low-level I/O."""

    def __init__(self, game_file: str, save_file: str, render_mode: str | None = None):
        self.game_file = game_file
        self.save_file = save_file
        self.render_mode = render_mode

        window_mode = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(game_file, sound_emulated=False, window=window_mode)

        self.try_load_state(save_file)

    def try_load_state(self, state_path: str | Path) -> bool:
        path = Path(state_path)
        if not path.exists():
            return False
        with path.open("rb") as handle:
            self.pyboy.load_state(handle)
        return True

    def load_state(self, state_path: str | Path) -> None:
        with Path(state_path).open("rb") as handle:
            self.pyboy.load_state(handle)

    def tick(self, steps: int = 1) -> None:
        self.pyboy.tick(steps)

    def send_press_release(
        self,
        press_event: int,
        release_event: int,
        press_ticks: int,
        release_ticks: int,
    ) -> None:
        self.pyboy.send_input(press_event)
        self.pyboy.tick(press_ticks)
        self.pyboy.send_input(release_event)
        self.pyboy.tick(release_ticks)

    def read_memory(self, address: int) -> int:
        return int(self.pyboy.memory[address])

    def screen(self) -> np.ndarray:
        return self.pyboy.screen.ndarray

    def game_area(self) -> np.ndarray:
        return self.pyboy.game_area()

    def sprite_pos(self, sprite_index: int = 2) -> tuple[int, int]:
        sprite = self.pyboy.get_sprite(sprite_index)
        return int(sprite.x), int(sprite.y)

    def render(self, mode: str | None) -> np.ndarray | None:
        if mode == "rgb_array":
            return self.pyboy.screen.ndarray
        return None

    def close(self) -> None:
        self.pyboy.stop()
