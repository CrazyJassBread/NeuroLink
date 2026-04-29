from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from .constants import (
    COLOR_HUD_TEXT,
    COLOR_HUD_TEXT_DIM,
    HUD_PIXEL_Y,
    SCALE_FACTOR,
    TARGET_FPS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from .env import DungeonEnv
from .input_state import HumanInputState


class ZeldaLikeGame:
    def __init__(self, room_file: str | Path):
        pygame.init()
        pygame.display.set_caption("NeuroLink DIY Dungeon")

        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_dim = pygame.font.SysFont("Consolas", 13)

        self.env = DungeonEnv(room_file=room_file, render_mode="rgb_array", auto_reset_on_step=True)
        self.env.reset()
        self.input_state = HumanInputState()
        self.running = True

    def _draw(self) -> None:
        frame = self.env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        self.display_surface.blit(scaled, (0, 0))

        line_1, line_2 = self.env.hud_lines()
        hud_scale = SCALE_FACTOR
        base_y = HUD_PIXEL_Y * hud_scale + 10
        text_1 = self.font.render(line_1, True, COLOR_HUD_TEXT)
        text_2 = self.font_dim.render(line_2, True, COLOR_HUD_TEXT_DIM)
        self.display_surface.blit(text_1, (18, base_y))
        self.display_surface.blit(text_2, (18, base_y + 30))
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self.clock.tick(TARGET_FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.input_state.handle_keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.input_state.handle_keyup(event.key)

            if self.running:
                frame_action = self.input_state.resolve_action()
                self.env.step(frame_action)

            self._draw()

        self.env.close()
        pygame.quit()
