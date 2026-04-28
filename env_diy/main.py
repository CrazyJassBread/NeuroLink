from __future__ import annotations

import argparse
from pathlib import Path

from .game import ZeldaLikeGame


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-resolution Zelda-style pygame prototype")
    default_room_file = (
        Path(__file__).resolve().parent / "map_data" / "dungeons" / "prototype" / "dungeon.json"
    )
    parser.add_argument(
        "--rooms",
        type=str,
        default=str(default_room_file),
        help="Path to dungeon definition JSON",
    )
    args = parser.parse_args()

    game = ZeldaLikeGame(room_file=args.rooms)
    game.run()


if __name__ == "__main__":
    main()
