from __future__ import annotations

from pathlib import Path

import pygame

from .constants import (
    COLOR_BG,
    COLOR_DIALOG_BG,
    COLOR_DIALOG_TEXT,
    COLOR_GRID,
    COLOR_HUD,
    COLOR_WALL,
    DIALOGUE_SECONDS,
    IFRAMES_SECONDS,
    INTERNAL_HEIGHT,
    INTERNAL_WIDTH,
    MONSTER_SIZE,
    SCALE_FACTOR,
    TARGET_FPS,
    TILE_SIZE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from .entities import Player, clamp
from .input_state import InputState
from .monsters import Bounds
from .observation import downsampled_observation
from .room import RoomDefinition, RoomManager


class ZeldaLikeGame:
    def __init__(self, room_file: str | Path):
        pygame.init()
        pygame.display.set_caption("Dual-Resolution Zelda Prototype")

        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.internal_surface = pygame.Surface((INTERNAL_WIDTH, INTERNAL_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Consolas", 10)
        self.input_state = InputState()

        self.room_manager = RoomManager(room_file)
        self.room_coord = self.room_manager.start_room
        self.room = self.room_manager.get_room(self.room_coord)

        spawn_x, spawn_y = self.room.player_spawn
        self.player = Player(spawn_x * TILE_SIZE, spawn_y * TILE_SIZE)

        self.dialogue_text = ""
        self.dialogue_timer = 0.0
        self.running = True

    def get_observation(self):
        return downsampled_observation(self.room, self.player)

    def check_proximity(self) -> None:
        for chest in self.room.chests:
            if chest.try_open(self.player):
                return

        for npc in self.room.npcs:
            text = npc.try_talk(self.player)
            if text is not None:
                self.dialogue_text = text
                self.dialogue_timer = DIALOGUE_SECONDS
                return

    def _update_player(self, dt: float) -> None:
        movement = self.input_state.movement_vector()
        self.player.update_movement(movement, dt, self.room.wall_rects())

        if self.input_state.action_a_pressed():
            self.check_proximity()

        if self.input_state.action_b_pressed():
            print("Button B pressed. Reserved for future skills.")

    def _update_monsters(self, dt: float) -> None:
        bounds: Bounds = (
            0.0,
            float(INTERNAL_WIDTH - MONSTER_SIZE),
            0.0,
            float(INTERNAL_HEIGHT - MONSTER_SIZE),
        )
        wall_rects = self.room.wall_rects()
        for monster in self.room.monsters:
            monster.update(self.player, dt, bounds, wall_rects)

    def _resolve_combat(self) -> None:
        for monster in self.room.monsters:
            if monster.rect.colliderect(self.player.rect):
                damaged = self.player.take_damage(1, iframes_seconds=IFRAMES_SECONDS)
                if damaged:
                    print(f"Player hit! HP: {self.player.hp}")
                break

    def _transition_room(self, direction: str) -> None:
        offsets = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, -1),
            "down": (0, 1),
        }
        dx, dy = offsets[direction]
        self.room_coord = (self.room_coord[0] + dx, self.room_coord[1] + dy)
        self.room = self.room_manager.get_room(self.room_coord)

        if direction == "left":
            self.player.x = float(INTERNAL_WIDTH - self.player.size)
        elif direction == "right":
            self.player.x = 0.0
        elif direction == "up":
            self.player.y = float(INTERNAL_HEIGHT - self.player.size)
        elif direction == "down":
            self.player.y = 0.0

        self.player.x = clamp(self.player.x, 0.0, float(INTERNAL_WIDTH - self.player.size))
        self.player.y = clamp(self.player.y, 0.0, float(INTERNAL_HEIGHT - self.player.size))

        print(f"Transitioned to room {self.room_coord}")

    def _handle_room_edges(self) -> None:
        if self.player.x < 0.0:
            self._transition_room("left")
            return
        if self.player.x > float(INTERNAL_WIDTH):
            self._transition_room("right")
            return
        if self.player.y < 0.0:
            self._transition_room("up")
            return
        if self.player.y > float(INTERNAL_HEIGHT):
            self._transition_room("down")

    def _draw_room(self) -> None:
        self.internal_surface.fill(COLOR_BG)

        for wall in self.room.wall_rects():
            pygame.draw.rect(self.internal_surface, COLOR_WALL, wall)

        for chest in self.room.chests:
            chest.draw(self.internal_surface)
        for npc in self.room.npcs:
            npc.draw(self.internal_surface)
        for monster in self.room.monsters:
            monster.draw(self.internal_surface)

        self.player.draw(self.internal_surface)

        # Faint 16x16 grid overlay for visualizing downsample cell boundaries.
        for x in range(0, INTERNAL_WIDTH + 1, TILE_SIZE):
            pygame.draw.line(self.internal_surface, COLOR_GRID, (x, 0), (x, INTERNAL_HEIGHT), 1)
        for y in range(0, INTERNAL_HEIGHT + 1, TILE_SIZE):
            pygame.draw.line(self.internal_surface, COLOR_GRID, (0, y), (INTERNAL_WIDTH, y), 1)

        self._draw_hud()
        self._draw_dialogue()

    def _draw_hud(self) -> None:
        hud = self.font.render(
            f"Room: {self.room_coord}  HP: {self.player.hp}",
            True,
            COLOR_HUD,
        )
        self.internal_surface.blit(hud, (4, 3))

    def _draw_dialogue(self) -> None:
        if self.dialogue_timer <= 0.0:
            return

        panel_rect = pygame.Rect(2, INTERNAL_HEIGHT - 24, INTERNAL_WIDTH - 4, 22)
        pygame.draw.rect(self.internal_surface, COLOR_DIALOG_BG, panel_rect)

        text = self.font.render(self.dialogue_text[:54], True, COLOR_DIALOG_TEXT)
        self.internal_surface.blit(text, (6, INTERNAL_HEIGHT - 18))

    def _present(self) -> None:
        scaled = pygame.transform.scale(
            self.internal_surface,
            (INTERNAL_WIDTH * SCALE_FACTOR, INTERNAL_HEIGHT * SCALE_FACTOR),
        )
        self.display_surface.blit(scaled, (0, 0))
        pygame.display.flip()

    def tick(self, dt: float) -> None:
        pressed = pygame.key.get_pressed()
        self.input_state.update(pressed)

        self.player.update_timers(dt)
        self._update_player(dt)
        self._update_monsters(dt)
        self._resolve_combat()
        self._handle_room_edges()

        self.dialogue_timer = max(0.0, self.dialogue_timer - dt)

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(TARGET_FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            self.tick(dt)
            self._draw_room()
            self._present()

        pygame.quit()
