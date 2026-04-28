from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from pyboy import PyBoy


class _StateHotkeys:
    """Handles optional global hotkeys for save/quit while the game is running."""

    def __init__(self, save_key: str, quit_key: str):
        self.save_key = save_key.lower()
        self.quit_key = quit_key.lower()
        self.save_requested = False
        self.quit_requested = False
        self._listener = None
        self._last_save_state = False

    def start(self) -> None:
        try:
            from pynput import keyboard  # Optional dependency.
        except Exception:
            print("[WARN] pynput not installed, hotkeys are disabled.")
            print("       Install with: pip install pynput")
            print("       Or quit with Ctrl+C in terminal.")
            return

        def on_press(key):
            char = getattr(key, "char", None)
            if not char:
                return
            c = char.lower()
            if c == self.save_key and not self._last_save_state:
                self.save_requested = True
                self._last_save_state = True
            elif c == self.quit_key:
                self.quit_requested = True
                return False

        def on_release(key):
            char = getattr(key, "char", None)
            if not char:
                return
            if char.lower() == self.save_key:
                self._last_save_state = False

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


def _default_paths(project_root: Path) -> tuple[Path, Path]:
    rom = project_root / "game_state" / "Link's awakening.gb"
    state = project_root / "game_state" / "manual.state"
    return rom, state


def _resolve_path(path: str | None, fallback: Path) -> Path:
    if not path:
        return fallback
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _save_state(pyboy: PyBoy, save_path: Path, use_timestamp: bool) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    final_path = save_path
    if use_timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")

    with final_path.open("wb") as handle:
        pyboy.save_state(handle)
    return final_path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    default_rom, default_save = _default_paths(project_root)

    parser = argparse.ArgumentParser(
        description="Play Link's Awakening with optional state load/save hotkeys."
    )
    parser.add_argument("--rom", type=str, default=str(default_rom), help="Path to ROM file")
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Optional state file to load before starting",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        default=str(default_save),
        help="Where to save state when save key is pressed",
    )
    parser.add_argument(
        "--save-key",
        type=str,
        default="x",
        help="Global hotkey to save state (default: x)",
    )
    parser.add_argument(
        "--quit-key",
        type=str,
        default="q",
        help="Global hotkey to quit (default: q)",
    )
    parser.add_argument(
        "--tick-sleep",
        type=float,
        default=0.01,
        help="Sleep between ticks to reduce CPU usage",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=300,
        help="Print Link room/position every N ticks; <=0 disables print",
    )
    parser.add_argument(
        "--timestamp-save",
        action="store_true",
        help="Append timestamp to save filename each time",
    )
    args = parser.parse_args()

    rom_path = _resolve_path(args.rom, default_rom)
    load_state_path = _resolve_path(args.state, default_save) if args.state else None
    save_state_path = _resolve_path(args.save_state, default_save)

    if not rom_path.exists():
        print(f"[ERROR] ROM not found: {rom_path}")
        return 1

    if len(args.save_key) != 1 or len(args.quit_key) != 1:
        print("[ERROR] --save-key and --quit-key must be single characters")
        return 1

    print(f"[INFO] ROM: {rom_path}")
    if load_state_path:
        print(f"[INFO] Load state: {load_state_path}")
    print(f"[INFO] Save state: {save_state_path}")
    print(f"[INFO] Hotkeys: save='{args.save_key}', quit='{args.quit_key}'")

    pyboy = PyBoy(str(rom_path), sound_emulated=False, window="SDL2")
    hotkeys = _StateHotkeys(save_key=args.save_key, quit_key=args.quit_key)

    try:
        if load_state_path:
            if load_state_path.exists():
                with load_state_path.open("rb") as handle:
                    pyboy.load_state(handle)
                print(f"[INFO] Loaded state: {load_state_path}")
            else:
                print(f"[WARN] State file not found, skip load: {load_state_path}")

        hotkeys.start()
        tick_count = 0
        while True:
            if hotkeys.quit_requested:
                print("[INFO] Quit requested")
                break

            pyboy.tick()
            tick_count += 1

            if hotkeys.save_requested:
                target = _save_state(pyboy, save_state_path, use_timestamp=args.timestamp_save)
                print(f"[INFO] State saved: {target}")
                hotkeys.save_requested = False

            if args.print_every > 0 and tick_count % args.print_every == 0:
                link = pyboy.get_sprite(2)
                room = int(pyboy.memory[0xDBAE])
                print(f"[INFO] tick={tick_count} room={room} link=({link.x}, {link.y})")

            if args.tick_sleep > 0:
                time.sleep(args.tick_sleep)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        hotkeys.stop()
        pyboy.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
