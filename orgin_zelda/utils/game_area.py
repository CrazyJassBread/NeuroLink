from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyboy.utils import WindowEvent

# Ensure local project modules resolve first when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from orgin_zelda.envs.config import ObservationConfig, resolve_device
from orgin_zelda.envs.emulator import EmulatorController
from orgin_zelda.envs.observation import ObservationProcessor


GAME_FILE = "asserts/game_state/Link's awakening.gb"
SAVE_STATE = "asserts/game_state/Room_51.state"
MAX_FRAMES = 100000

# Rendering every frame with matplotlib is expensive and can stall the emulator.
UI_UPDATE_EVERY = 3
TEXT_UPDATE_EVERY = 6

KEY_TO_EVENTS = {
    "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    "z": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    "x": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
}

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
    pressed_keys: set[str] = set()

    def on_key_press(event):
        nonlocal running
        if not event.key:
            return
        key = event.key.lower()
        if key == "q":
            running = False
            return

        mapped = KEY_TO_EVENTS.get(key)
        if mapped is None or key in pressed_keys:
            return
        pressed_keys.add(key)
        emulator.send_input(mapped[0])

    def on_key_release(event):
        if not event.key:
            return
        key = event.key.lower()
        mapped = KEY_TO_EVENTS.get(key)
        if mapped is None or key not in pressed_keys:
            return
        pressed_keys.remove(key)
        emulator.send_input(mapped[1])

    plt.ion()
    fig, (ax_raw, ax_proc) = plt.subplots(1, 2, figsize=(10, 4))
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    raw_img = ax_raw.imshow(np.zeros((128, 160), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
    ax_raw.set_title("Raw 128x160")
    ax_raw.axis("off")

    obs_shape = OBS_CONFIG.output_shape
    proc_img = ax_proc.imshow(
        np.zeros(obs_shape, dtype=np.float32),
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    ax_proc.set_title(f"Processed ({OBS_CONFIG.mode})")
    ax_proc.axis("off")

    # Numeric overlay is useful in bucketed mode because the map is small.
    annotations = None
    last_obs_int = None
    if OBS_CONFIG.mode == "bucketed":
        annotations = _build_annotations(ax_proc, obs_shape[0], obs_shape[1])
        last_obs_int = np.full(obs_shape, -1, dtype=np.int16)

    plt.tight_layout()
    plt.show()

    try:
        for frame_idx in range(MAX_FRAMES):
            if not running:
                break

            emulator.tick(1)
            screen = emulator.screen()
            if OBS_CONFIG.mode != "bucketed":
                # Show continuous pooled values before bucketization for debugging.
                obs = processor.process_bucketed_prebucket(screen)
            else:
                obs = processor.process(screen)

            # Throttle figure redraw frequency to reduce UI overhead.
            should_render = (frame_idx % UI_UPDATE_EVERY) == 0
            if should_render:
                raw_img.set_data(screen[:128, :160, 0])
                proc_img.set_data(obs)

                if annotations is not None and last_obs_int is not None and (frame_idx % TEXT_UPDATE_EVERY) == 0:
                    obs_int = np.rint(obs).astype(np.int16, copy=False)
                    changed = obs_int != last_obs_int
                    if np.any(changed):
                        rows, cols = np.where(changed)
                        for i, j in zip(rows, cols):
                            annotations[i][j].set_text(f"{float(obs[i, j]):.1f}")
                        last_obs_int[:, :] = obs_int

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                # Keep keyboard events responsive even when not redrawing.
                plt.pause(0.0001)
    finally:
        for key in list(pressed_keys):
            mapped = KEY_TO_EVENTS.get(key)
            if mapped is not None:
                emulator.send_input(mapped[1])
        emulator.close()
        plt.ioff()


if __name__ == "__main__":
    main()
