from __future__ import annotations

from dataclasses import dataclass

from .constants import MESSAGE_DEFAULT
from .state import PlayerState, tile_from_position_px
from .world.rooms import RoomManager, RoomState
from .types import RuntimeSnapshot


@dataclass
class RuntimeState:
    room_manager: RoomManager
    room_coord: tuple[int, int]
    room: RoomState
    player: PlayerState
    episode: int = 0
    step_count: int = 0
    pending_reset: bool = False
    last_message: str = MESSAGE_DEFAULT
    no_progress_steps: int = 0
    seed: int | None = None

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            room_id=self.room.room_id,
            room_coord=self.room.coord,
            player_position_px=self.player.position_px,
            player_tile=tile_from_position_px(self.player.position_px, self.player.size_px),
            health=self.player.health,
            gold=self.player.gold,
            keys=self.player.keys,
            items=tuple(self.player.items),
            no_progress_steps=self.no_progress_steps,
            step_count=self.step_count,
            episode_id=self.episode,
            seed=self.seed,
        )
