from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure local project modules resolve first when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.config import ObservationConfig, resolve_device
from envs.emulator import EmulatorController
from envs.observation import ObservationProcessor


GAME_FILE = "game_state/Link's awakening.gb"
SAVE_STATE = "game_state/Room58_task2.state"
MAX_FRAMES = 100000

# Use the same config as your RL env observation pipeline.
OBS_CONFIG = ObservationConfig(
    mode="bucketed",          # Change to "resized" if needed
    output_shape=(8, 10),
    grayscale=True,
    normalize=False,
)


def _build_annotations(ax, height: int, width: int):
    annotations: list[list] = []
    for i in range(height):
        row_annotations = []
        for j in range(width):
            text = ax.text(
                j,
                i,
                "",
                ha="center",
                va="center",
                color="red",
                fontsize=8,
            )
            row_annotations.append(text)
        annotations.append(row_annotations)
    return annotations


def main() -> None:
    device = resolve_device(None)
    emulator = EmulatorController(GAME_FILE, SAVE_STATE, render_mode=None)
    processor = ObservationProcessor(OBS_CONFIG, device)

    running = True

    def on_key(event):
        nonlocal running
        if event.key and event.key.lower() == "q":
            running = False

    plt.ion()
    fig, (ax_raw, ax_proc) = plt.subplots(1, 2, figsize=(10, 4))
    fig.canvas.mpl_connect("key_press_event", on_key)

    raw_img = ax_raw.imshow(np.zeros((128, 160), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
    ax_raw.set_title("Raw 128x160")
    ax_raw.axis("off")

    obs_shape = OBS_CONFIG.output_shape
    proc_img = ax_proc.imshow(np.zeros(obs_shape, dtype=np.float32), cmap="gray")
    ax_proc.set_title(f"Processed ({OBS_CONFIG.mode})")
    ax_proc.axis("off")

    # Numeric overlay is useful in bucketed mode because the map is small.
    annotations = None
    if OBS_CONFIG.mode == "bucketed":
        annotations = _build_annotations(ax_proc, obs_shape[0], obs_shape[1])

    plt.tight_layout()
    plt.show()

    try:
        for _ in range(MAX_FRAMES):
            if not running:
                break

            emulator.tick(1)
            screen = emulator.screen()
            obs = processor.process(screen)

            raw_img.set_data(screen[:128, :160, 0])
            proc_img.set_data(obs)

            if annotations is not None:
                for i in range(obs.shape[0]):
                    for j in range(obs.shape[1]):
                        annotations[i][j].set_text(str(int(obs[i, j])))

            plt.draw()
            plt.pause(0.001)
    finally:
        emulator.close()
        plt.ioff()


if __name__ == "__main__":
    main()
