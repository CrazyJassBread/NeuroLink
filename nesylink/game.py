from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from .core.constants import TARGET_FPS, WINDOW_HEIGHT, WINDOW_WIDTH
from .core.input import HumanInputState
from .env import make_env


class ZeldaLikeGame:
    def __init__(self, room_file: str | Path):
        pygame.init()
        pygame.display.set_caption("NesyLink Dungeon")

        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.env = make_env(room_file, api="gym", render_mode="rgb_array", auto_reset_on_step=True)
        self.env.reset()
        self.input_state = HumanInputState()
        self.running = True

    def _draw(self) -> None:
        frame = self.env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        self.display_surface.blit(scaled, (0, 0))
        pygame.display.flip()

    def run(self) -> None:
        game_over = False
        victory = False
        while self.running:
            self.clock.tick(TARGET_FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.type == pygame.KEYDOWN and (game_over or victory):
                    self.env.reset()
                    game_over = False
                    victory = False
                elif event.type == pygame.KEYDOWN:
                    self.input_state.handle_keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.input_state.handle_keyup(event.key)

            if self.running and not game_over and not victory:
                frame_action = self.input_state.resolve_action()
                _, _, terminated, _, info = self.env.step(frame_action)
                if terminated and info.get("terminal_reason") == "agent_dead":
                    game_over = True
                elif terminated and info.get("terminal_reason") == "world_completed":
                    victory = True

            self._draw()
            if game_over or victory:
                self._draw_overlay("GAME OVER - Press any key" if game_over else "VICTORY - Press any key")

        self.env.close()
        pygame.quit()

    def _draw_overlay(self, text: str) -> None:
        font = pygame.font.SysFont(None, 28)
        surface = font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.display_surface.blit(surface, rect)
        pygame.display.flip()


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
